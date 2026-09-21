# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UPP (Umami PreProcessing) is a modular preprocessing pipeline for jet tagging in the ATLAS high-energy physics collaboration. It selects and resamples jets from HDF5 input files to produce balanced training datasets for b-tagging classifiers.

## Commands

```bash
# Install in development mode
python -m pip install -e ".[dev]"
pre-commit install

# Run full test suite with coverage
pytest -v --junitxml=coverage.xml --cov-report=term-missing:skip-covered --cov=upp tests/

# Run a single test file
pytest tests/unit/test_resampling.py -v

# Run all code quality checks
pre-commit run --all-files

# Run the pipeline
preprocess --config upp/configs/GN3EPCMV01/GN3EPCMV01.yaml --split train --prep --resample

# Validate input samples before preprocessing
check_input_samples --config <config.yaml>

# Build documentation
pip install -r docs/requirements.txt && mkdocs serve
```

## Architecture

### Pipeline Stages

The pipeline is orchestrated in [upp/main.py](upp/main.py) and executes these stages in order, each selectively enabled via CLI flags:

1. **SplitContainers** (`--prep`) — Splits input HDF5 containers into per-component chunks
2. **Reweight** — Applies reweighting if configured
3. **Hist** (`--resample`) — Builds PDF histograms used for jet selection
4. **Resampling** (`--resample`) — Selects jets to match target distributions
5. **Merging** (`--merge`) — Combines components into output HDF5 files (one per split: train/val/test)
6. **Normalisation** (`--norm`) — Computes shift/scale factors from training split only
7. **Plotting** (`--plot`) — Visualizes input and output distributions

### Core Abstractions

**Component** ([upp/classes/components.py](upp/classes/components.py)) — A `(sample, flavour, region)` tuple representing one slice of the data. The `Components` collection manages all combinations and their I/O setup.

**PreprocessingConfig** ([upp/classes/preprocessing_config.py](upp/classes/preprocessing_config.py)) — Loads and validates the YAML config. Passed to every stage. Config files support inclusion via `yamlinclude`.

**Resampling methods** ([upp/stages/resampling.py](upp/stages/resampling.py)):
- `pdf` — Estimates PDFs from histograms, samples to match target distribution
- `countup` — Statistical method using available statistics
- `none` — Pass-through (no resampling)

### I/O Strategy

- Source files are wrapped in **HDF5 virtual datasets** — no data duplication
- All processing is batch-based with configurable batch sizes
- Data stored as **structured arrays** for compact disk storage
- H5Reader/H5Writer abstractions handle all I/O

### Configuration

YAML configs in [upp/configs/](upp/configs/) follow a hierarchy:
- Split definitions live in `upp/configs/splits/`
- Predefined configs: `GN3EPCMV01/`, `GN3V00/`, `GN3V01/`, `xbb*.yaml`, `xtautau*.yaml`
- `--split train/val/test` selects which split to process

### Key Data Flow

```
YAML config
  → PreprocessingConfig
  → Components (sample × flavour × region)
  → Stages (SplitContainers → Resample → Merge → Norm → Plot)
  → Output HDF5 files per split
```

## Code Quality

- **Linting/formatting**: Ruff
- **Type checking**: mypy
- **Docstrings**: pydoclint
- **YAML**: yamllint
- All enforced via pre-commit hooks and CI
