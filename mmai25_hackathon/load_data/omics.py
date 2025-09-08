from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor


# ======================================================================
# load_multiomics: build per-modality sample-graph(s) from CSV features
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
    """
    Build one PYG `Data` per omics modality by:
      1) Loading feature matrices (CSV; delimiter=",") into numpy arrays.
      2) (Optional) Applying `pre_transform` to each X (e.g., standardization).
      3) Computing pairwise similarities (currently cosine) between samples.
      4) Fitting a *global* similarity cutoff so each node keeps ~k neighbors.
      5) Thresholding, symmetrizing (max), adding self-loops, row-normalizing.
      6) Packaging `edge_index`, `edge_weight`, dense `x`, and `SparseTensor adj_t`.

    Labels (optional):
      - If `labels_csv` is provided, loads (N,) or (N,C) integer labels.
      - If 1D labels and `num_classes` is set, converts to one-hot (float).
      - If `mode=="train"`, computes `train_sample_weight` per sample.

    Parameters
    ----------
    feature_csvs : list[str]
        Paths to feature CSVs. Each is expected to be shape (N, F_m) after reading via
        `np.loadtxt(..., delimiter=",")`. (Make sure your file really is comma-separated.)
    labels_csv : str | None
        Optional path to labels CSV (comma-separated). Shape (N,) or (N, C).
        If (N,1) it will be squeezed to (N,).
    featname_csvs : list[str] | None
        Optional per-modality feature-name CSVs (no header). If omitted, names become
        `m{mi}_f{j}`.
    mode : {"train","val","test"}
        Drives whether to *fit* thresholds (train) or *reuse* from `ref` if present.
    num_classes : int | None
        If labels are 1D ints and this is provided, labels are one-hot encoded.
    pre_transform : callable | None
        Function applied to each X (numpy). Signature: `X -> X_transformed`.
    target_pre_transform : callable | None
        Function applied to `y` (numpy). Useful for label re-mapping.
    edge_per_node : int
        Desired avg number of neighbors per node (k) used to fit a global cutoff.
    metric : {"cosine"}
        Similarity metric. Currently only "cosine" is implemented.
    eps : float
        Numerical epsilon for normalization / division safeguards.
    equal_weight : bool
        If True, assigns uniform sample weights (1/N). Otherwise class-frequency based.
        (See NOTE in `_sample_weight` docstring.)
    ref : dict | None
        If provided, may contain pre-fitted thresholds: `{"sim_thresholds": [thr_m0, ...]}`.
        When `mode!="train"` we try to reuse `ref["sim_thresholds"][mi]` for each modality.
        If missing, we refit on the fly.
    rng : np.random.Generator | None
        Reserved for future stochastic steps (currently unused). Defaults to a fixed seed.

    Returns
    -------
    out : dict
        {
          "data_list": [Data_m0, Data_m1, ...],   # one PYG Data per modality
          "fit_": {"sim_thresholds": [thr_m0, thr_m1, ...]}  # fitted/used thresholds
        }

    Graph construction details
    --------------------------
    - Similarity: S = cosine(X_i, X_j). Diagonal is 1.
    - Fit cutoff:
        * Mask diag to -inf, flatten off-diagonals, pick global (k*N)-th largest.
        * This yields a *single* threshold per modality.
    - Adjacency:
        * Mask: M = (S >= thr), zero diag, A0 = S * M
        * Symmetrize: A = max(A0, A0^T)
        * Add self-loops: A <- A + I
        * Row-normalize: A[i,:] /= sum(A[i,:])

    Saved on each `Data` object
    ---------------------------
    x : torch.FloatTensor [N, F_m]
    y : torch.FloatTensor [N] or [N, C] or None
    mode : str
    feat_names : np.ndarray[object]  # feature names list for modality m
    edge_index : torch.LongTensor [2, E]
    edge_weight : torch.FloatTensor [E]
    adj_t : torch_sparse.SparseTensor (N x N)
    train_sample_weight : torch.FloatTensor [N]  (only when mode=="train")

    Notes & gotchas
    ---------------
    - All modalities must share the same sample count N (checked).
    - Labels (if provided) must also have length N.
    - `np.loadtxt` is strict; malformed CSVs (headers/strings) will fail. Use pandas
      to read & convert to numeric if your inputs are not purely numeric.
    - The global cutoff approach keeps *on average* k neighbors per node; some nodes
      may have more/less depending on the similarity distribution and symmetrization.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    # -----------------------------
    # (1) Load labels, if present
    # -----------------------------
    y_np = None
    if labels_csv is not None:
        y_np = _load_labels(labels_csv)

    # -----------------------------------------------
    # (2) Load features + optional pre_transform/name
    # -----------------------------------------------
    X_list_np, featnames_all = [], []
    for mi, fpath in enumerate(feature_csvs):
        # Expect pure numeric CSV with comma delimiter
        Xi = np.loadtxt(fpath, delimiter=",")
        if pre_transform is not None:
            Xi = pre_transform(Xi)  # e.g., StandardScaler().fit_transform(Xi)
        X_list_np.append(Xi)

        # Feature names
        if featname_csvs and mi < len(featname_csvs) and featname_csvs[mi]:
            names = pd.read_csv(featname_csvs[mi], header=None).iloc[:, 0].astype(str).tolist()
        else:
            names = [f"m{mi}_f{j}" for j in range(Xi.shape[1])]
        featnames_all.append(names)

    # Consistency: N must be same across modalities
    nset = {X.shape[0] for X in X_list_np}
    if len(nset) != 1:
        raise ValueError(f"All modalities must have same sample count; got {nset}")
    N = next(iter(nset))

    # Labels length must match N
    if y_np is not None and y_np.shape[0] != N:
        raise ValueError(f"labels length ({y_np.shape[0]}) != samples ({N})")

    # Optional label pre-transform (e.g., relabeling, one-hot already, etc.)
    if y_np is not None and target_pre_transform is not None:
        y_np = target_pre_transform(y_np)

    # Convert labels to torch; one-hot if requested
    if y_np is not None:
        y_t = torch.as_tensor(y_np)
        if y_t.ndim == 1 and (num_classes is not None):
            y_t = F.one_hot(y_t.to(torch.long), num_classes=num_classes).float()
    else:
        y_t = None

    # ---------------------------------------------
    # (3) Build per-modality adjacency + Data objs
    # ---------------------------------------------
    sim_thresholds: List[float] = []
    data_list: List[Data] = []

    allow_fit = (mode == "train")
    applied_ref = (ref is not None and "sim_thresholds" in ref and isinstance(ref["sim_thresholds"], list))

    for mi, X_np in enumerate(X_list_np):
        X = torch.as_tensor(X_np, dtype=torch.float32)

        # ---- Fit or reuse threshold for this modality ----
        if allow_fit:
            S = _pairwise_cosine(X, eps=eps)
            S_no_diag = S.clone(); S_no_diag.fill_diagonal_(float("-inf"))
            thr = _fit_global_cutoff(S_no_diag, edge_per_node)
        else:
            if applied_ref and mi < len(ref["sim_thresholds"]):
                thr = float(ref["sim_thresholds"][mi])
            else:
                # Fallback: compute from current data (useful for standalone val/test)
                S = _pairwise_cosine(X, eps=eps)
                S_no_diag = S.clone(); S_no_diag.fill_diagonal_(float("-inf"))
                thr = _fit_global_cutoff(S_no_diag, edge_per_node)

        sim_thresholds.append(thr)

        # ---- Build row-normalized adjacency ----
        S = _pairwise_cosine(X, eps=eps)
        M = (S >= thr).float()
        M.fill_diagonal_(0.0)
        A = _symmetrize_max(S * M)              # undirected
        A = _add_I_and_row_normalize(A, eps=eps)  # add self-loops + normalize rows

        # Convert to (edge_index, edge_weight)
        Asp = A.to_sparse()
        ei = Asp.indices()           # [2, E]
        ew = Asp.values()            # [E]

        # ---- Sample weights (train only) ----
        if y_t is not None and mode == "train":
            if y_t.ndim == 2:  # one-hot
                labels_train = torch.argmax(y_t, dim=1).cpu().numpy()
                n_cls = y_t.shape[1]
            else:              # integer
                labels_train = y_t.to(torch.long).cpu().numpy()
                n_cls = int(labels_train.max()) + 1 if labels_train.size > 0 else (num_classes or 0)
            sw = _sample_weight(labels_train, n_cls, equal_weight)
            train_sample_weight = torch.as_tensor(sw, dtype=torch.float32)
        else:
            train_sample_weight = None

        # ---- Package PYG Data object for this modality ----
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
    """
    Load labels from a comma-separated CSV using `np.loadtxt`.

    Returns
    -------
    y : np.ndarray
        - If shape is (N,1), squeezed to (N,) and cast to int.
        - If shape is (N,C), returned as-is (e.g., multi-label or already one-hot).

    Tip
    ---
    If your label file has headers or non-numeric tokens, use pandas to read and
    then extract a numeric array instead of `np.loadtxt`.
    """
    y = np.loadtxt(path, delimiter=",")
    if y.ndim == 2 and y.shape[1] == 1:
        y = y.reshape(-1)
    return y.astype(int) if y.ndim == 1 else y


def _pairwise_cosine(X: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """
    Dense cosine similarity between rows of X (N x F) -> (N x N) with diag=1.

    Implementation
    --------------
    - L2-normalize rows, then return Xn @ Xn^T.
    """
    Xn = F.normalize(X, p=2, dim=1, eps=eps)
    return Xn @ Xn.T


def _fit_global_cutoff(S_no_diag: torch.Tensor, k: int) -> float:
    """
    Choose a *single* similarity cutoff so each node keeps ~k neighbors on average.

    Parameters
    ----------
    S_no_diag : torch.Tensor [N, N]
        Similarity matrix with diagonal pre-set to -inf.
    k : int
        Target avg neighbors per node. Internally clipped to [1, N-1].

    Method
    ------
    - Flatten off-diagonal entries, take the global (k*N)-th largest.
      Implemented as kthvalue on the negated array.

    Returns
    -------
    thr : float
        Scalar similarity threshold.
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
    """
    Add self-loops and row-normalize so each row sums to ~1.

    Returns
    -------
    A_norm : torch.Tensor [N, N]
        Row-stochastic adjacency.
    """
    N = A.shape[0]
    A = A + torch.eye(N, dtype=A.dtype, device=A.device)
    row_sum = A.sum(dim=1, keepdim=True).clamp_min(eps)
    return A / row_sum


def _sample_weight(labels: np.ndarray, num_classes: int, equal_weight: bool) -> np.ndarray:
    """
    Per-sample weight vector for (imbalanced) classification.

    Behavior
    --------
    - If `equal_weight=True`: uniform weights 1/N.
    - Else: weights are **proportional to class frequency** (NOT inverse):
        w_i = count[label_i] / sum(count)
      This gives larger weights to majority classes.

    NOTE
    ----
    If what you want is *inverse* frequency (common choice to rebalance):
        inv = 1.0 / np.maximum(count, 1)
        w = inv[labels]
        w = w / w.sum()
    Replace the body accordingly if desired.
    """
    if labels.size == 0:
        return np.array([], dtype=np.float32)
    if equal_weight:
        return np.ones(len(labels), dtype=np.float32) / len(labels)
    count = np.bincount(labels, minlength=num_classes if num_classes else labels.max() + 1)
    return (count[labels] / np.sum(count)).astype(np.float32)
