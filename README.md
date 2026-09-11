# MethylSegPaper

This repository contains the analysis workflows and figure notebooks used to
benchmark [MethylSeg](https://github.com/clementlab/MethylSeg) and study its
downstream biological applications. It is a companion analysis repository,
not the MethylSeg Python package itself. For installation and usage of the
software, see the [MethylSeg documentation](https://clementlab.github.io/MethylSeg/).

The repository covers PMD-caller comparisons, synthetic recovery benchmarks,
chromatin and lamina-associated domain (LAD) analyses, TCGA classification,
and the notebooks used to assemble the resulting figures.

## Repository contents

| Location | Purpose |
| --- | --- |
| [`analysis/01_region_calling_analysis/`](analysis/01_region_calling_analysis/) | Compare MethylSeg with other PMD callers and aggregate their outputs. |
| [`analysis/02_synthetic_analysis/`](analysis/02_synthetic_analysis/) | Generate synthetic PMDs and measure caller recovery. |
| [`analysis/03_chromatin_analysis/`](analysis/03_chromatin_analysis/) | Analyze chromatin overlap and deepTools profiles. |
| [`analysis/04_lad_analysis/`](analysis/04_lad_analysis/) | Analyze LAD overlap and laminB1 signal. |
| [`analysis/05_tcga_classification_analysis/`](analysis/05_tcga_classification_analysis/) | Run TCGA segmentation and downstream classification. |
| [`figures/`](figures/) | Executed publication-oriented notebooks and shared plotting helpers. |
| [`repo_paths.py`](repo_paths.py) | Shared repository, data, results, and figure-output paths. |
| [`environment.yaml`](environment.yaml) | Export of the software environment used for the analyses. |

## Environment

The committed [`environment.yaml`](environment.yaml) is an export of the
`jt_wgbs_analysis` Conda environment used for this project. It includes the
Python and command-line dependencies used by the workflows, including the
version of MethylSeg used for these analyses.

Create the environment with:

```bash
conda env create -f environment.yaml
conda activate jt_wgbs_analysis
```

To update an existing environment from the export:

```bash
conda env update -n jt_wgbs_analysis -f environment.yaml --prune
```

## Data and results

Large input datasets, intermediate files, and generated figure files are not
stored in Git. Before running a workflow, stage the required files under the
appropriate repository-local directory:

- `data/methylation_data/` for WGBS and array methylation inputs;
- `data/chromatin_data/` for chromatin signal and interval tracks;
- `data/reference_data/` for genome and analysis reference files;
- `data/tcga_samples/` for TCGA inputs.

Analysis outputs are written below `results/`. This can be an ordinary
directory or a symlink to larger scratch storage:

```bash
mkdir -p results

# Alternatively, from a fresh clone:
ln -s /path/to/scratch/results results
```

Figure notebooks write exported assets below `figures/out/`. The notebooks in
[`figures/`](figures/) retain their executed outputs so that their results can
be inspected on GitHub, but re-executing them requires the corresponding local
data and analysis results.

> [!IMPORTANT]
> The tracked configurations and Slurm scripts reflect the original CHPC
> environment and include checkout-specific paths and CHPC scheduler settings.
> Users running elsewhere must adapt those paths, partitions, accounts, and
> resource requests to their system. Portability changes are intentionally
> outside the scope of this repository snapshot.

## Main workflows

Run commands from the repository root after activating the environment and
staging the required inputs.

### Region-calling benchmark

Submit the caller-comparison and aggregation jobs:

```bash
./analysis/01_region_calling_analysis/slurm_code/run_slurm.sh
```

The sample list is defined in
[`configs.txt`](analysis/01_region_calling_analysis/slurm_code/configs.txt),
with individual YAML configurations in the adjacent `configs/` directory.

### Synthetic recovery benchmark

Submit the synthetic PMD workflow:

```bash
./analysis/02_synthetic_analysis/slurm_code/run_slurm.sh
```

See the
[`Synthetic Slurm Pipeline` guide](analysis/02_synthetic_analysis/slurm_code/README.md)
for output structure, reuse rules, and targeted reruns.

### Chromatin analysis

Inspect the available options or submit the CHPC workflow:

```bash
python analysis/03_chromatin_analysis/run_chromatin.py --help
./analysis/03_chromatin_analysis/slurm_code/run_slurm.sh --help
```

### LAD analysis

Inspect the local runner or Slurm options:

```bash
python analysis/04_lad_analysis/01_run_lad.py --help
./analysis/04_lad_analysis/slurm_code/run_slurm.sh --help
```

### TCGA classification

The TCGA workflow separates PMD detection from downstream machine-learning
submission:

```bash
./analysis/05_tcga_classification_analysis/slurm_code/run_PMD_detection.sh --help
./analysis/05_tcga_classification_analysis/slurm_code/run_tcga_ml_slurm.sh --help
```

## Citation

If you use this repo, please cite the software using the
[CITATION.cff](https://github.com/clementlab/MethylSeg/blob/main/CITATION.cff)
metadata. On GitHub, select **Cite this repository** to copy the citation in APA
or BibTeX format.

A manuscript describing MethylSeg is in preparation. Its citation will be added
when available.

## License

This repository is available under the
[BSD 3-Clause License](LICENSE).
