# MMAI '25 Hackathon Base Source Code

## Overview

This repository provides the base source code for the MMAI '25 workshop Hackathon. It is designed to help participants get started quickly with a pre-configured Python environment and essential dependencies for development and testing.

## Requirements

- Python 3.9, 3.10, 3.11, or 3.12 (other versions are not supported)
- Git


## Installation (Development)

You can set up your development environment using one of the following methods: `venv`, `conda`, or `uv`.

### Using venv (Standard Python Virtual Environment)

1. **Create and activate a virtual environment:**
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
2. **Install dependencies:**
	```bash
	pip install --upgrade pip
	pip install -e .
	```

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

