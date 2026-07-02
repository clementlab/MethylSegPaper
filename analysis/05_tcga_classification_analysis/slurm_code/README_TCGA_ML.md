# TCGA Classification Slurm Pipeline

This folder contains the batch entrypoints for the migrated TCGA PMD detection
and machine-learning analysis. The reusable implementation lives in:

- `../utils/tcga_segmentation_workflow.py`
- `../utils/tcga_ml_pipeline.py`

## Results Layout

The Slurm wrappers write into:

- `results/05_tcga_classification_analysis/segmentation`
- `results/05_tcga_classification_analysis/ml_outputs`
- `results/05_tcga_classification_analysis/logs`

The ML wrapper reuses the cached segmentation outputs in `segmentation/` and
does not rerun PMD calling unless you explicitly opt back into that behavior.

## Run On Slurm

```bash
cd /uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/analysis/05_tcga_classification_analysis/slurm_code
```

Submit only the PMD-detection array job:

```bash
./run_PMD_detection.sh
```

Delete existing MethylSeg PMD outputs and rerun segmentation:

```bash
./run_PMD_detection.sh --force
```

Submit the ML analysis against the current cached segmentation results:

```bash
./run_tcga_ml_slurm.sh
```

Useful environment overrides:

```bash
ARRAY_TASK_COUNT=40 ./run_PMD_detection.sh

CASE_SETS=brca,pan_cancer \
N_VALUES=1,3,5,10,50,100,500,1000 \
CV_SPLITS=5 \
N_ESTIMATORS=500 \
./run_tcga_ml_slurm.sh
```

For quick debug jobs, add `MAX_SAMPLES_PER_CLASS=10` and small `N_VALUES`.

## ML Defaults

- `CALL_MISSING_PMDS=0` by default, so the ML job uses the existing segmentation
  cache and fails clearly if required PMD files are missing.
- Set `CALL_MISSING_PMDS=1` only if you want the ML pipeline to backfill missing
  PMDs with MethylSeg.
- `ALLOW_MISSING_FEATURE_SETS=0` by default, so the ML job fails if PMD or
  Random feature generation breaks.
- `SAVE_FEATURE_MATRICES=1` by default.
- Random regions avoid high-confidence PMDs from the training fold, with default
  `HIGH_CONFIDENCE_PMD_MIN_FRACTION=0.5`.
- Random region bounds default to
  `RANDOM_REGION_MIN_LENGTH_BP=150000` and
  `RANDOM_REGION_MAX_LENGTH_BP=20000000`.

## Run Locally

```bash
python run_tcga_ml_pipeline.py run \
  --out-dir /uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results/05_tcga_classification_analysis/ml_outputs_debug \
  --segmentation-root /uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results/05_tcga_classification_analysis/segmentation \
  --case-sets brca \
  --n-values 1,3 \
  --cv-splits 2 \
  --n-estimators 20 \
  --max-samples-per-class 5 \
  --no-call-missing-pmds
```

## Recalculate Metrics

After a run finishes, metrics and plots can be regenerated from saved
predictions without retraining:

```bash
python run_tcga_ml_pipeline.py recalculate-metrics \
  --out-dir /uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results/05_tcga_classification_analysis/ml_outputs
```
