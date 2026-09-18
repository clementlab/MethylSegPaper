# Historical MethylSeg parameter development

This directory archives the random parameter sweep used during development of
the MethylSeg defaults. The sweep was run on April 20, 2026, before the
40 kb and 450 kb window defaults were introduced.

## Contents

- `methylseg_parameter_sweeps.ipynb` generated 25 seeded random
  configurations and ran the original benchmark workflow.
- `methylsegtest_utils.py` contains the historical sweep, execution, and
  summary helpers.
- `methylseg_sweep_figure_browser.ipynb` is the supported entry point for
  inspecting the archived tables, rankings, and figures.

The cached archive is stored outside Git at
`results/00_methylseg_development/random_parallel_sweep/`. Run the browser
notebook from either the repository root or this directory; it discovers the
repository and result root automatically.

## Sweep design

The sweep used random seed 42 to generate 25 configurations. Each configuration
sampled one to four shared WGBS/HM450K window sizes between 5 kb and 2 Mb, an
HM450K continuous-time HMM holding-time guess, and three region-cleaning
parameters. Configurations were evaluated separately for WGBS and HM450K-like
outputs using synthetic base-pair precision/recall, H3K36me2 chromatin
enrichment, and LAD end-window overlap.

Twenty configurations completed all summaries. `random_17`, `random_19`,
`random_20`, `random_22`, and `random_25` failed because ESO26 HM450K produced
no retained regions after cleaning. The archive retains those failures and
their tracebacks.

The browser reports independent rankings for synthetic recovery, chromatin,
and LAD overlap; the historical code did not calculate a single overall
winner. `random_05` sampled 41,828 bp and 469,418 bp windows, which were later
simplified to the 40 kb and 450 kb defaults.

## Compatibility limitation

The sweep runner is preserved for provenance, not as a currently supported
workflow. It imports legacy comparator, synthetic-preparation, chromatin, LAD,
and MethylSeg interfaces that have changed or moved. Its repository and output
paths have been adapted to this checkout, but it may not execute end to end
without reconstructing the historical software environment and dependencies.

The figure-browser notebook does not run MethylSeg. It reads the archived TSV
tables and saved HTML/PNG figures and is expected to work with the current
paper-repository environment.

## Archived and omitted results

The ignored result archive retains:

- the sweep manifest, status table, and sweep summary;
- all 25 configuration snapshots and configuration hashes;
- primary, chromatin, LAD, deepTools, and synthetic summary tables;
- synthetic/run manifests and the original failure tracebacks;
- all 100 HTML figures and 40 PNG profiles used by the browser.

The archive intentionally omits repeated methylation preparation files,
synthetic methylation matrices and normalized inputs, copied reference tracks,
segmentation BED collections, deepTools matrices and sorted-region files,
logs, caches, and bytecode. Paths that identify those omitted intermediates
are retained in relocated repository-relative form for provenance and are
listed as omitted in the result archive's `archive_inventory.tsv`.
