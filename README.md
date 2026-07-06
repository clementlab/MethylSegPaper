# 20260624_methylseg

This repository contains the working analysis code for the MethylSeg benchmark
project. It is organized around a few reproducible workflow directories under
`analysis/`, with shared path constants in `repo_paths.py`, figure notebooks in
`figures/`, and heavy result outputs written through the repo-local `results/`
path.

## Environment

The committed `environment.yaml` is an export of the live
`jt_wgbs_analysis` conda environment used for this project.

Create the environment with:

```bash
conda env create -f environment.yaml
conda activate jt_wgbs_analysis
```

If the environment already exists and you want to refresh it:

```bash
conda env update -n jt_wgbs_analysis -f environment.yaml --prune
```

## Repository layout

- `analysis/01_region_calling_analysis/`: PMD caller comparison workflow and
  aggregation.
- `analysis/02_synthetic_analysis/`: synthetic PMD injection and recovery
  benchmark pipeline.
- `analysis/03_chromatin_analysis/`: chromatin overlap and deepTools analyses.
- `analysis/04_lad_analysis/`: LAD overlap, laminB1 signal, and profile
  analyses.
- `analysis/05_tcga_classification_analysis/`: TCGA segmentation and downstream
  machine-learning workflows.
- `figures/`: figure notebooks and shared plotting helpers.
- `data/`: local data staging and reference assets.
- `get_data/`: helper code for fetching project inputs.
- `repo_paths.py`: canonical project, data, figure, and results paths.

## Results and path conventions

This checkout keeps a stable repo-local `results/` path, but in practice that
path is a symlink to scratch storage on CHPC. Scripts should use
`repo_paths.py` or the repo-local `results/` path instead of hard-coding a
scratch location.

Most workflows are written so that:

- code lives in the repo,
- large intermediate files land under `results/`,
- notebooks and scripts can share the same output tree.

## Main workflow entrypoints

### 1. Region calling benchmark

Submit the comparator array plus aggregation:

```bash
cd analysis/01_region_calling_analysis/slurm_code
./run_slurm.sh
```

The sample/config manifest lives in `analysis/01_region_calling_analysis/slurm_code/configs.txt`.

### 2. Synthetic recovery benchmark

Run the synthetic PMD pipeline:

```bash
cd analysis/02_synthetic_analysis/slurm_code
./run_slurm.sh
```

This workflow has a more detailed local guide in
`analysis/02_synthetic_analysis/slurm_code/README.md`.

### 3. Chromatin analysis

Run the chromatin analysis against comparator outputs:

```bash
python analysis/03_chromatin_analysis/run_chromatin.py --help
```

### 4. LAD analysis

Run the LAD overlap and signal workflow:

```bash
python analysis/04_lad_analysis/01_run_lad.py --help
```

### 5. TCGA classification

The Slurm entrypoints and defaults are documented in:

- `analysis/05_tcga_classification_analysis/slurm_code/README_TCGA_ML.md`

Typical entrypoints are:

```bash
cd analysis/05_tcga_classification_analysis/slurm_code
./run_PMD_detection.sh
./run_tcga_ml_slurm.sh
```

## Figures and notebooks

The `figures/` directory contains the publication-oriented notebooks and shared
helpers in `figures/utils/figures_utils.py`. The numbered notebooks generally
mirror the numbered workflow areas in `analysis/`.

For workflow debugging and development, several analysis folders also contain
test or exploratory notebooks alongside the Python entrypoints.

TODO: move figure outputs out of results folder and into figures/out

Fix bug in methylseg 450k where it will call a region over the centromere with almost no data