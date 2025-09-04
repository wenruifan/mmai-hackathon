# MMAI '25 Hackathon Base Source Code

## Overview

This repository provides the base source code for the MMAI '25 workshop Hackathon. It is designed to help participants get started quickly with a pre-configured Python environment and essential dependencies for development and testing.

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12 (other versions are not supported)
- Git



## Installation

### Prerequisite

Before installing other dependencies, install pykale with all optional dependencies (full extras) from git:

```bash
pip install "pykale@git+https://github.com/pykale/pykale@main[full]"
```

You can set up your development environment using one of the following methods: `venv`, `conda`, or `uv`.

### Main Installation Steps

1. **Clone the repository:**
	```bash
	git clone https://github.com/pykale/mmai-hackathon.git
	cd mmai-hackathon
	```
2. **Set up a virtual environment (recommended):**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
3. **Install dependencies:**
	```bash
	pip install --upgrade pip
	# Install pykale with all optional dependencies (full extras) from git first
	pip install "pykale@git+https://github.com/pykale/pykale@main[full]"
	pip install -e .
	```

#### Installing torch-geometric (pyg) and its extensions

To install torch-geometric (`pyg`) and its required extensions (such as `torch-scatter`, `torch-sparse`, etc.), use the following command with the appropriate URL for your PyTorch and CUDA version:

```bash
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

Replace the URL with the one matching your PyTorch and CUDA version. For more details and the latest URLs, see the official torch-geometric installation guide: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

---

You can also use the following environment-specific guides:

### Using conda (Anaconda/Miniconda)

1. **Create and activate a conda environment:**
	```bash
	conda create -n mmai-hackathon python=3.10
	conda activate mmai-hackathon
	```
2. **Install dependencies:**
	```bash
	pip install -e .
	```


### Using uv (Ultra-fast Python package manager)

Assuming `uv` is already installed:

1. **Create and activate a uv virtual environment:**
	```bash
	uv venv .venv
	source .venv/bin/activate
	```
2. **Install dependencies:**
	```bash
	uv pip install -e .
	```

---

1. **Clone the repository:**
	```bash
	git clone https://github.com/pykale/mmai-hackathon.git
	cd mmai-hackathon
	```

2. **Set up a virtual environment (recommended):**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```

3. **Install dependencies:**
	```bash
	pip install --upgrade pip
	pip install -e .
	```

4. **(Optional) Install pre-commit hooks:**
	```bash
	pre-commit install
	```

5. **Run tests:**
	```bash
	pytest
	```

## Notes

- The project restricts Python versions to 3.9–3.12 as specified in `.python-version` and `pyproject.toml`.
- For more information about the dependencies, see `pyproject.toml`.

## Authors

- Shuo Zhou (<shuo.zhou@sheffield.ac.uk>)
- Xianyuan Liu (<xianyuan.liu@sheffield.ac.uk>)
- Wenrui Fan (<wenrui.fan@sheffield.ac.uk>)
- Mohammod N. I. Suvon (<m.suvon@sheffield.ac.uk>)
- L. M. Riza Rizky (<l.m.rizky@sheffield.ac.uk>)
