from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor


# ======================================================================
# Linear pipeline: load ONE split (or the whole dataset) and build graph
# ======================================================================
def load_multiomics(
    *,
    feature_csvs: List[str],
    labels_csv: Optional[str] = None,
    featname_csvs: Optional[List[str]] = None,
    mode: Literal["train", "val", "test"] = "train",
    num_classes: Optional[int] = None,
    pre_transform: Optional[Any] = None,
    target_pre_transform: Optional[Any] = None,
    edge_per_node: int = 10,
    metric: Literal["cosine"] = "cosine",
    eps: float = 1e-8,
    equal_weight: bool = False,
    ref: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    if rng is None:
        rng = np.random.default_rng(12345)

    y_np = None
    if labels_csv is not None:
        y_np = _load_labels(labels_csv)

    X_list_np, featnames_all = [], []
    for mi, fpath in enumerate(feature_csvs):
        Xi = np.loadtxt(fpath, delimiter=",")
        if pre_transform is not None:
            Xi = pre_transform(Xi)
        X_list_np.append(Xi)
        if featname_csvs and mi < len(featname_csvs) and featname_csvs[mi]:
            names = pd.read_csv(featname_csvs[mi], header=None).iloc[:, 0].astype(str).tolist()
        else:
            names = [f"m{mi}_f{j}" for j in range(Xi.shape[1])]
        featnames_all.append(names)

    nset = {X.shape[0] for X in X_list_np}
    if len(nset) != 1:
        raise ValueError(f"All modalities must have same sample count; got {nset}")
    N = next(iter(nset))

    if y_np is not None and y_np.shape[0] != N:
        raise ValueError(f"labels length ({y_np.shape[0]}) != samples ({N})")

    if y_np is not None and target_pre_transform is not None:
        y_np = target_pre_transform(y_np)

    if y_np is not None:
        y_t = torch.as_tensor(y_np)
        if y_t.ndim == 1 and (num_classes is not None):
            y_t = F.one_hot(y_t.to(torch.long), num_classes=num_classes).float()
    else:
        y_t = None

    sim_thresholds: List[float] = []
    data_list: List[Data] = []
    allow_fit = (mode == "train")
    applied_ref = (ref is not None and "sim_thresholds" in ref and isinstance(ref["sim_thresholds"], list))

    for mi, X_np in enumerate(X_list_np):
        X = torch.as_tensor(X_np, dtype=torch.float32)

        if allow_fit:
            S = _pairwise_cosine(X, eps=eps)
            S_no_diag = S.clone(); S_no_diag.fill_diagonal_(float("-inf"))
            thr = _fit_global_cutoff(S_no_diag, edge_per_node)
        else:
            if applied_ref and mi < len(ref["sim_thresholds"]):
                thr = float(ref["sim_thresholds"][mi])
            else:
                S = _pairwise_cosine(X, eps=eps)
                S_no_diag = S.clone(); S_no_diag.fill_diagonal_(float("-inf"))
                thr = _fit_global_cutoff(S_no_diag, edge_per_node)

        sim_thresholds.append(thr)

        S = _pairwise_cosine(X, eps=eps)
        M = (S >= thr).float()
        M.fill_diagonal_(0.0)
        A = _symmetrize_max(S * M)
        A = _add_I_and_row_normalize(A, eps=eps)

        Asp = A.to_sparse()
        ei = Asp.indices()           # <-- fixed
        ew = Asp.values()

        if y_t is not None and mode == "train":
            if y_t.ndim == 2:
                labels_train = torch.argmax(y_t, dim=1).cpu().numpy()
                n_cls = y_t.shape[1]
            else:
                labels_train = y_t.to(torch.long).cpu().numpy()
                n_cls = int(labels_train.max()) + 1 if labels_train.size > 0 else (num_classes or 0)
            sw = _sample_weight(labels_train, n_cls, equal_weight)
            train_sample_weight = torch.as_tensor(sw, dtype=torch.float32)
        else:
            train_sample_weight = None

        data = Data(
            x=X,
            y=y_t,
            mode=mode,
            feat_names=np.asarray(featnames_all[mi], dtype=object),
            edge_index=ei,
            edge_weight=ew,
            adj_t=SparseTensor(row=ei[0], col=ei[1], value=ew, sparse_sizes=(N, N)),
        )
        if train_sample_weight is not None:
            data.train_sample_weight = train_sample_weight

        data_list.append(data)

    return {"data_list": data_list, "fit_": {"sim_thresholds": sim_thresholds}}


# =========================
# ---- Helper functions ----
# =========================
def _load_labels(path: str) -> np.ndarray:
    """Load labels CSV to (N,) or (N,C). If a single column, squeeze to 1D int."""
    y = np.loadtxt(path, delimiter=",")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    return y.astype(int) if y.ndim == 1 else y

def _pairwise_cosine(X: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Dense cosine similarity between rows of X (N x F) -> (N x N) with diag=1."""
    Xn = F.normalize(X, p=2, dim=1, eps=eps)
    return Xn @ Xn.T

def _fit_global_cutoff(S_no_diag: torch.Tensor, k: int) -> float:
    """
    Pick a SINGLE global cutoff across the (N x N) similarity matrix (diagonal masked -inf)
    such that we keep ~k neighbors per node on average:
    - Conceptually: take the global (k*N)-th largest off-diagonal similarity.
    - Implementation: kthvalue on the negated flattened similarities.
    """
    N = S_no_diag.shape[0]
    k_eff = max(min(k, max(N - 1, 1)), 1)        # 1 <= k_eff <= N-1
    flat = S_no_diag.reshape(-1)
    valid = flat.isfinite()
    flat = flat[valid]
    if flat.numel() == 0:
        return float("inf")                       # degenerate case (N<=1)
    pos = max(min(k_eff * N - 1, flat.numel() - 1), 0)
    val = torch.kthvalue(-flat, k=pos + 1).values
    return float(-val.item())

def _symmetrize_max(A: torch.Tensor) -> torch.Tensor:
    """Symmetrize adjacency by elementwise maximum (undirected graph)."""
    return torch.maximum(A, A.T)

def _add_I_and_row_normalize(A: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Add self-loops and row-normalize so each row sums to 1."""
    N = A.shape[0]
    A = A + torch.eye(N, dtype=A.dtype, device=A.device)
    row_sum = A.sum(dim=1, keepdim=True).clamp_min(eps)
    return A / row_sum

def _sample_weight(labels: np.ndarray, num_classes: int, equal_weight: bool) -> np.ndarray:
    """
    Per-sample weight vector:
      - If equal_weight=True: uniform 1/N
      - Else: inverse-frequency per class (as in original code spirit)
    """
    if labels.size == 0:
        return np.array([], dtype=np.float32)
    if equal_weight:
        return np.ones(len(labels), dtype=np.float32) / len(labels)
    count = np.bincount(labels, minlength=num_classes if num_classes else labels.max() + 1)
    return (count[labels] / np.sum(count)).astype(np.float32)