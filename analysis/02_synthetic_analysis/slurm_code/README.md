# Synthetic Slurm Pipeline

This folder runs the synthetic PMD benchmark from the notebook workflow:

1. Call PMDs on healthy source samples.
2. Remove likely pre-existing healthy PMDs to make clean backgrounds.
3. Inject synthetic PMDs into those backgrounds.
4. Run PMD callers again.
5. Compute per-sample recovery metric shards.
6. Merge metric shards into final tables and plots.

`run_slurm.sh` defaults to:

```bash
/uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results/02_synthetic_analysis
```

In this checkout, `results/` is a symlink into scratch, so the workflow still writes heavy artifacts on scratch while keeping the repo-local results path as the stable entrypoint.

## Full Run

Submit the pipeline from this folder:

```bash
./run_slurm.sh
```

The production run uses all 12 source samples listed in `pipeline_config.py`.

By default, `run_slurm.sh` archives old non-synthetic outputs and keeps validated injected synthetic samples. This lets you rerun recovery and metric aggregation without rebuilding synthetic methylation tables.

To rebuild synthetic samples from scratch:

```bash
./run_slurm.sh --force_recreate
```

To write to another output root:

```bash
./run_slurm.sh --out-root /scratch/general/vast/$USER/slurm_jobs/my_synthetic_run
```

## Output Layout

- `healthy_pmds/`: first-pass PMD caller outputs on healthy samples.
- `healthy_background_references/`: normalized healthy backgrounds and removed-PMD truth BEDs.
- `injected_pmd_samples/`: injected methylation tables, synthetic truth BEDs, and injected manifest.
- `synthetic_recovery/tool_results/`: second-pass PMD caller outputs on injected samples.
- `synthetic_recovery/metrics/`: per-sample metrics, per-tool summaries, sample/tool status tables, false-PMD beta tables.
- `synthetic_recovery/metrics/shards/`: per-sample metric shard files written by the metric array stage.
- `synthetic_recovery/plots/`: HTML plots built from the saved metric tables.
- `archives/`: timestamped old outputs moved aside by `run_slurm.sh`.

## Reuse Rules

Injected synthetic samples are reused only when `injected_pmd_samples/synthetic_sample_manifest.tsv` exists and every referenced `synthetic_file` and `truth_bed` exists.

If that validation fails, the submitter stops and asks for `--force_recreate`. This avoids quietly mixing partial synthetic outputs with fresh recovery results.

## Rerun Only Metrics

After recovery outputs exist, metrics can be recalculated without rerunning PMD callers:

```bash
python aggregate_metrics.py \
  --out-root /uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results/02_synthetic_analysis \
  --boundary-tolerance-bp 10000 \
  --n-sample-procs 8
```

Change or add metric logic in `aggregate_metrics.py` or `synthetic_analysis_helpers.py`, then rerun the command above.

If per-sample metric shards already exist and you only want to rebuild the merged metric tables and plots:

```bash
python aggregate_metrics.py \
  --out-root /uufs/chpc.utah.edu/common/home/u0914269/clement/projects/20260624_methylseg/results/02_synthetic_analysis \
  --boundary-tolerance-bp 10000 \
  --from-shards
```

Aggregation uses the injected manifest as the expected sample list, but computes summaries from samples with at least one parseable tool output. Missing or failed samples are written to `synthetic_recovery/metrics/metric_sample_status.tsv`.

To rerun only the metric array stage plus the final metric/plot aggregation:

```bash
./rerun_metrics_only.sh
```

To rerun only selected failed array indices:

```bash
./rerun_metrics_only.sh --array 3,7,11
```

## Debug Run

Use `../09_slurm_pipeline_debug.ipynb` for a one-sample local run. It writes to `../debug_slurm_pipeline_run` and is intended for short checks before submitting Slurm jobs.

## Memory Notes

The one-sample debug run in this folder recorded very large peak memory for `MethylSeekR` and `mmseekr`, so the healthy and recovery array jobs run the comparator with `--n-jobs 1` even though the Slurm task requests multiple CPUs. The extra CPUs remain available for tool-internal work, but the pipeline avoids launching multiple heavy callers at once inside a single sample job.
