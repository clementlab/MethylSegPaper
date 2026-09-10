import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ANALYSIS_DIR.parents[1]
for import_path in (PROJECT_ROOT, ANALYSIS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from utils import tcga_ml_pipeline as pipeline  # noqa: E402
from slurm_code.run_tcga_segmentation_array import (  # noqa: E402
    assert_segmentation_chunk_complete,
)


def _sample_id(patient_number: int, sample_suffix: str = "01A") -> str:
    return f"TCGA-AA-{patient_number:04d}-{sample_suffix}"


def _balanced_grouped_inputs(n_patients_per_class: int = 10):
    tumor_samples = [_sample_id(i, "01A") for i in range(n_patients_per_class)]
    normal_samples = [
        _sample_id(100 + i, "11A") for i in range(n_patients_per_class)
    ]
    sample_ids = np.asarray(tumor_samples + normal_samples, dtype=object)
    y = np.asarray(
        [1] * n_patients_per_class + [0] * n_patients_per_class,
        dtype=int,
    )
    groups = {
        sample_id: pipeline.extract_tcga_patient_id(sample_id)
        for sample_id in sample_ids
    }
    return sample_ids, y, groups


@pytest.mark.parametrize(
    ("sample_id", "expected"),
    [
        ("TCGA-AB-1234-01A", "TCGA-AB-1234"),
        ("TCGA-AB-1234-01A-01D-A1B2-05", "TCGA-AB-1234"),
    ],
)
def test_extract_tcga_patient_id(sample_id, expected):
    assert pipeline.extract_tcga_patient_id(sample_id) == expected


@pytest.mark.parametrize(
    "sample_id",
    ["", "AB-1234-01A", "TCGA-AB-123-01A", "TCGA-A-1234-01A", "TCGA-AB-1234"],
)
def test_extract_tcga_patient_id_rejects_malformed_barcodes(sample_id):
    with pytest.raises(ValueError, match="Expected a TCGA sample barcode"):
        pipeline.extract_tcga_patient_id(sample_id)


def test_make_patient_groups_rejects_duplicate_samples():
    samples = pd.DataFrame(
        {
            "sample": ["TCGA-AA-0001-01A", "TCGA-AA-0001-01A"],
            "sample_type": ["Primary Tumor", "Primary Tumor"],
        }
    )
    with pytest.raises(ValueError, match="Duplicate TCGA sample barcodes"):
        pipeline.make_patient_groups(samples)


def test_grouped_splits_are_deterministic_disjoint_and_stratified():
    sample_ids, y, groups = _balanced_grouped_inputs()
    # Add a normal specimen for one tumor participant to exercise a mixed-label group.
    mixed_normal = _sample_id(0, "11A")
    sample_ids = np.append(sample_ids, mixed_normal)
    y = np.append(y, 0)
    groups[mixed_normal] = "TCGA-AA-0000"

    first = pipeline.build_repeated_stratified_group_splits(
        sample_ids,
        y,
        groups,
        n_splits=5,
        n_repeats=10,
        random_seed=42,
    )
    second = pipeline.build_repeated_stratified_group_splits(
        sample_ids,
        y,
        groups,
        n_splits=5,
        n_repeats=10,
        random_seed=42,
    )

    assert len(first) == 50
    assert [job["split_number"] for job in first] == list(range(50))
    assert [job["repeat_seed"] for job in first[::5]] == list(range(42, 52))
    for job_a, job_b in zip(first, second):
        np.testing.assert_array_equal(job_a["train_idx"], job_b["train_idx"])
        np.testing.assert_array_equal(job_a["test_idx"], job_b["test_idx"])
        train_groups = {groups[sample_ids[idx]] for idx in job_a["train_idx"]}
        test_groups = {groups[sample_ids[idx]] for idx in job_a["test_idx"]}
        assert train_groups.isdisjoint(test_groups)
        assert set(y[job_a["train_idx"]]) == {0, 1}
        assert set(y[job_a["test_idx"]]) == {0, 1}


def test_grouped_splits_require_complete_exact_patient_mapping():
    sample_ids, y, groups = _balanced_grouped_inputs()
    missing = dict(groups)
    missing.pop(sample_ids[0])
    with pytest.raises(ValueError, match="assignments are missing"):
        pipeline.build_repeated_stratified_group_splits(
            sample_ids, y, missing, n_splits=5, n_repeats=1, random_seed=42
        )

    null_group = dict(groups)
    null_group[sample_ids[0]] = None
    with pytest.raises(ValueError, match="cannot be missing or empty"):
        pipeline.build_repeated_stratified_group_splits(
            sample_ids, y, null_group, n_splits=5, n_repeats=1, random_seed=42
        )

    wrong_group = dict(groups)
    wrong_group[sample_ids[0]] = "TCGA-AA-9999"
    with pytest.raises(ValueError, match="first three TCGA barcode fields"):
        pipeline.build_repeated_stratified_group_splits(
            sample_ids, y, wrong_group, n_splits=5, n_repeats=1, random_seed=42
        )


def test_grouped_splits_reject_duplicate_sample_ids():
    sample_ids, y, groups = _balanced_grouped_inputs()
    duplicated = np.append(sample_ids, sample_ids[0])
    duplicated_y = np.append(y, y[0])
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        pipeline.build_repeated_stratified_group_splits(
            duplicated,
            duplicated_y,
            groups,
            n_splits=5,
            n_repeats=1,
            random_seed=42,
        )


def test_cohort_manifest_uses_patient_counts_and_reports_mixed_groups():
    rows = []
    for patient_number in range(5):
        rows.append(
            {
                "sample": _sample_id(patient_number, "01A"),
                "sample_type": "Primary Tumor",
                "project_id": "TCGA-TEST",
            }
        )
    for patient_number in range(100, 105):
        rows.append(
            {
                "sample": _sample_id(patient_number, "11A"),
                "sample_type": "Solid Tissue Normal",
                "project_id": "TCGA-TEST",
            }
        )
    rows.append(
        {
            "sample": _sample_id(0, "11A"),
            "sample_type": "Solid Tissue Normal",
            "project_id": "TCGA-TEST",
        }
    )
    manifest = pipeline.build_per_cancer_cohort_manifest(
        pd.DataFrame(rows), min_splits=5
    )
    cohort = manifest.iloc[0]
    assert cohort["n_samples"] == 11
    assert cohort["n_patients"] == 10
    assert cohort["n_tumor_patients"] == 5
    assert cohort["n_normal_patients"] == 6
    assert cohort["n_mixed_label_patients"] == 1


def test_cohort_eligibility_requires_distinct_patients_per_class():
    rows = []
    for cohort_id, normal_patient_numbers in (
        ("TCGA-GOOD", range(100, 105)),
        ("TCGA-BAD", [200, 200, 200, 200, 200]),
    ):
        for patient_number in range(5):
            rows.append(
                {
                    "sample": _sample_id(patient_number + (300 if cohort_id.endswith("BAD") else 0)),
                    "sample_type": "Primary Tumor",
                    "project_id": cohort_id,
                }
            )
        for sample_number, patient_number in enumerate(normal_patient_numbers):
            rows.append(
                {
                    "sample": _sample_id(patient_number, f"11{chr(65 + sample_number)}"),
                    "sample_type": "Solid Tissue Normal",
                    "project_id": cohort_id,
                }
            )

    manifest = pipeline.build_per_cancer_cohort_manifest(
        pd.DataFrame(rows), min_splits=5
    )
    assert manifest["cohort_id"].tolist() == ["TCGA-GOOD"]


def test_training_observed_cpg_filter_handles_zero_partial_and_complete_coverage():
    meth_data = pd.DataFrame(
        {
            "CpG_chrm": ["chr1", "chr1", "chr1"],
            "CpG_beg": [100, 200, 300],
            "CpG_end": [101, 201, 301],
            "TCGA-AA-0001-01A": [np.nan, 0.2, 0.3],
            "TCGA-AA-0002-01A": [np.nan, np.nan, 0.4],
        }
    )
    lookup = pipeline.build_training_observed_cpg_position_lookup(
        meth_data,
        ["TCGA-AA-0001-01A", "TCGA-AA-0002-01A"],
    )
    regions = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "start": [90, 190, 190],
            "end": [110, 210, 310],
            "name": ["zero", "partial", "complete"],
        }
    )
    filtered = pipeline.filter_regions_by_training_cpg_coverage(regions, lookup)
    assert filtered["name"].tolist() == ["partial", "complete"]
    assert filtered["training_observed_cpgs"].tolist() == [1, 2]

    stricter_lookup = pipeline.build_training_observed_cpg_position_lookup(
        meth_data,
        ["TCGA-AA-0001-01A", "TCGA-AA-0002-01A"],
        min_observed_samples=2,
    )
    assert stricter_lookup["chr1"].tolist() == [300]


def test_distinct_region_sampling_fails_instead_of_replacing():
    duplicated_regions = pd.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "start": [100, 100],
            "end": [200, 200],
        }
    )
    with pytest.raises(ValueError, match="1 distinct eligible regions, but 2 were requested"):
        pipeline._sample_region_rows(
            duplicated_regions,
            2,
            seed=42,
            feature_set_name="Random PMD",
        )


def test_score_feature_set_rejects_all_missing_training_regions():
    meth_data = pd.DataFrame(
        {
            "CpG_chrm": ["chr1"],
            "CpG_beg": [100],
            "CpG_end": [101],
            "TCGA-AA-0001-01A": [np.nan],
            "TCGA-AA-0002-11A": [0.2],
        }
    )
    regions = pd.DataFrame({"chrom": ["chr1"], "start": [90], "end": [110]})
    with pytest.raises(ValueError, match="no observed methylation values"):
        pipeline.score_feature_set(
            ["TCGA-AA-0001-01A"],
            ["TCGA-AA-0002-11A"],
            np.asarray([1]),
            np.asarray([0]),
            regions,
            meth_data,
            random_state=42,
        )


def test_incomplete_segmentation_chunk_fails_before_dependent_ml():
    result_df = pd.DataFrame(
        {
            "sample_id": ["TCGA-AA-0001-01A"],
            "status": ["failed"],
            "summary_exists": [False],
            "error": ["missing output"],
        }
    )
    with pytest.raises(RuntimeError, match="1 incomplete samples"):
        assert_segmentation_chunk_complete(result_df)


def test_small_end_to_end_run_persists_patient_and_cv_metadata(tmp_path, monkeypatch):
    sample_ids, y, _ = _balanced_grouped_inputs(n_patients_per_class=6)
    samples_info = pd.DataFrame(
        {
            "sample": sample_ids,
            "sample_type": np.where(y == 1, "Primary Tumor", "Solid Tissue Normal"),
            "project_id": "TCGA-SMOKE",
        }
    )
    meth_data = pd.DataFrame(
        {
            "CpG_chrm": ["chr1", "chr1"],
            "CpG_beg": [100, 200],
            "CpG_end": [101, 201],
            **{
                sample_id: [0.8, 0.7] if label == 1 else [0.2, 0.3]
                for sample_id, label in zip(sample_ids, y)
            },
        }
    )
    pmds_per_sample = {
        sample_id: pd.DataFrame(
            {"chrom": ["chr1"], "start": [90], "end": [110], "type": ["PMD"]}
        )
        for sample_id in sample_ids
    }
    cgis = pd.DataFrame(
        {"chrom": ["chr1"], "start": [190], "end": [210], "name": ["cgi"]}
    )

    def fixed_feature_sets(n_features, training_pmds, cgis, **_kwargs):
        assert n_features == 1
        pmd = training_pmds.head(1).copy()
        cgi = cgis.head(1).copy()
        return {
            "PMD": pmd,
            "Random PMD": pmd,
            "CGI": cgi,
            "Random long": pmd,
            "Random short": cgi,
        }

    monkeypatch.setattr(pipeline, "load_tcga_samples", lambda _cohort: samples_info)
    monkeypatch.setattr(pipeline, "load_methylation_data", lambda _samples: meth_data)
    monkeypatch.setattr(pipeline, "load_pmds_per_sample", lambda _samples: pmds_per_sample)
    monkeypatch.setattr(pipeline, "load_cgis", lambda: cgis)
    monkeypatch.setattr(pipeline, "pick_features", fixed_feature_sets)

    outputs = pipeline.run_cohort_feature_classification(
        cohort_id="TCGA-SMOKE",
        feature_count=1,
        n_splits=2,
        n_repeats=1,
        n_feature_draws=1,
        random_seed=42,
        max_split_workers=2,
        out_root=tmp_path,
    )
    predictions = pd.read_csv(outputs["fold_predictions"], sep="\t")
    run_config = json.loads(outputs["run_config"].read_text())

    assert len(pd.read_csv(outputs["classification_results"], sep="\t")) == 12
    assert "patient_id" in predictions.columns
    assert predictions["patient_id"].str.match(r"^TCGA-AA-\d{4}$").all()
    assert run_config["cv_strategy"] == "repeated_stratified_group_k_fold"
    assert run_config["patient_id_rule"] == pipeline.TCGA_PATIENT_ID_RULE
    assert run_config["repeat_seeds"] == [42]
    assert run_config["sample_summary"]["n_patients"] == 12
    assert run_config["classifier"]["n_estimators"] == 500
    assert run_config["classifier"]["class_weight"] == "balanced"
    assert run_config["imputation"]["fit_scope"] == "training_partition_only"
    assert run_config["feature_definition"]["training_data_only"] is True
    assert run_config["feature_definition"]["insufficient_eligible_features"] == "raise_error"
    assert run_config["software_versions"]["scikit-learn"]
