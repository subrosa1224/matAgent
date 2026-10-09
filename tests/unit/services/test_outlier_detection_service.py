"""Unit tests for OutlierDetectionService — pure functions."""

from __future__ import annotations

import pytest

from materials_screening.models import MaterialRecord
from materials_screening.services.outlier_detection_service import (
    _deterministic_sample,
    detect_multivariate_outliers,
    detect_property_outliers,
)
from materials_screening.sub_agents.outlier_detection.models import (
    MultivariateOutlierReport,
    PropertyOutlierReport,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _rec(
    material_id: str = "mp-1",
    formula: str = "TiO2",
    band_gap: float | None = 2.0,
    formation_energy: float | None = -3.0,
    hull: float | None = 0.05,
    density: float | None = 4.0,
) -> MaterialRecord:
    return MaterialRecord(
        source="mock",
        material_id=material_id,
        formula_pretty=formula,
        elements=("Ti", "O"),
        band_gap_ev=band_gap,
        formation_energy_ev_atom=formation_energy,
        energy_above_hull_ev_atom=hull,
        density_g_cm3=density,
    )


# ── detect_property_outliers — Z-score ──────────────────────────────────────


class TestPropertyOutliersZscore:
    def test_small_sample_warning_and_formula_id_label(self) -> None:
        records = [
            _rec("mp-1", "TiO2", band_gap=1.0),
            _rec("mp-2", "SiO2", band_gap=5.0),
        ]

        report = detect_property_outliers(records, "band_gap_ev", method="zscore")

        assert report.records[0].material_label == "TiO2 (mp-1)"
        assert any("Small sample size" in warning for warning in report.warnings)

    def test_basic_zscore_detection(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=1.0),
            _rec("mp-2", "B", band_gap=1.2),
            _rec("mp-3", "C", band_gap=1.1),
            _rec("mp-4", "D", band_gap=1.3),
            _rec("mp-5", "E", band_gap=0.9),
            _rec("mp-6", "F", band_gap=1.15),
            _rec("mp-7", "G", band_gap=1.05),
            _rec("mp-8", "H", band_gap=1.25),
            _rec("mp-9", "I", band_gap=0.95),
            _rec("mp-10", "X", band_gap=10.0),  # clear outlier
        ]
        report = detect_property_outliers(records, "band_gap_ev", method="zscore")
        assert isinstance(report, PropertyOutlierReport)
        assert report.property_name == "band_gap_ev"
        assert report.method == "zscore"
        assert report.distribution.count == 10

        outliers = [r for r in report.records if r.is_outlier]
        assert len(outliers) == 1
        assert outliers[0].material_id == "mp-10"
        assert outliers[0].direction == "high"
        assert outliers[0].z_score is not None and outliers[0].z_score > 2.0

    def test_no_outliers_in_uniform_data(self) -> None:
        records = [_rec(f"mp-{i}", f"X{i}", band_gap=1.5) for i in range(10)]
        report = detect_property_outliers(records, "band_gap_ev")
        assert all(not r.is_outlier for r in report.records)

    def test_custom_threshold(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=1.0),
            _rec("mp-2", "B", band_gap=2.0),  # mild outlier with zscore default
        ]
        # Default threshold 2.0 — may or may not flag
        report = detect_property_outliers(records, "band_gap_ev")
        # Low threshold should flag more
        report_low = detect_property_outliers(
            records, "band_gap_ev", threshold=0.5
        )
        assert sum(1 for r in report_low.records if r.is_outlier) >= sum(
            1 for r in report.records if r.is_outlier
        )

    def test_distribution_stats(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=1.0),
            _rec("mp-2", "B", band_gap=2.0),
            _rec("mp-3", "C", band_gap=3.0),
        ]
        report = detect_property_outliers(records, "band_gap_ev")
        dist = report.distribution
        assert dist.count == 3
        assert dist.mean == pytest.approx(2.0)
        assert dist.min_value == 1.0
        assert dist.max_value == 3.0

    def test_low_outlier_detected(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=5.0),
            _rec("mp-2", "B", band_gap=5.2),
            _rec("mp-3", "C", band_gap=4.8),
            _rec("mp-4", "D", band_gap=5.1),
            _rec("mp-5", "E", band_gap=4.9),
            _rec("mp-6", "F", band_gap=5.15),
            _rec("mp-7", "G", band_gap=5.05),
            _rec("mp-8", "H", band_gap=4.85),
            _rec("mp-9", "I", band_gap=5.2),
            _rec("mp-10", "X", band_gap=0.1),  # low outlier
        ]
        report = detect_property_outliers(records, "band_gap_ev")
        low = [r for r in report.records if r.is_outlier and r.direction == "low"]
        assert len(low) == 1
        assert low[0].material_id == "mp-10"

    def test_both_high_and_low_outliers(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=5.0),
            _rec("mp-2", "B", band_gap=5.1),
            _rec("mp-3", "C", band_gap=4.9),
            _rec("mp-4", "D", band_gap=0.5),  # low
            _rec("mp-5", "E", band_gap=10.0),  # high
        ]
        report = detect_property_outliers(records, "band_gap_ev", threshold=1.0)
        directions = {r.direction for r in report.records if r.is_outlier}
        assert directions == {"high", "low"}


# ── detect_property_outliers — IQR ──────────────────────────────────────────


class TestPropertyOutliersIqr:
    def test_basic_iqr_detection(self) -> None:
        records = [
            _rec("mp-1", "A", density=4.0),
            _rec("mp-2", "B", density=4.2),
            _rec("mp-3", "C", density=3.9),
            _rec("mp-4", "D", density=4.1),
            _rec("mp-5", "E", density=4.05),
            _rec("mp-6", "F", density=3.95),
            _rec("mp-7", "G", density=4.15),
            _rec("mp-8", "H", density=3.85),
            _rec("mp-9", "I", density=4.1),
            _rec("mp-10", "X", density=20.0),  # clear outlier
        ]
        report = detect_property_outliers(records, "density_g_cm3", method="iqr")
        outliers = [r for r in report.records if r.is_outlier]
        assert len(outliers) == 1
        assert outliers[0].material_id == "mp-10"
        assert outliers[0].z_score is None  # IQR doesn't produce z-scores

    def test_iqr_distribution_stats(self) -> None:
        records = [_rec(f"mp-{i}", f"X{i}", density=float(i)) for i in range(1, 6)]
        report = detect_property_outliers(records, "density_g_cm3", method="iqr")
        dist = report.distribution
        assert dist.q3 > dist.q1
        assert dist.iqr == pytest.approx(dist.q3 - dist.q1)


# ── Missing values ──────────────────────────────────────────────────────────


class TestMissingValues:
    def test_none_values_excluded_with_warning(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=2.0),
            _rec("mp-2", "B", band_gap=None),
            _rec("mp-3", "C", band_gap=2.2),
        ]
        report = detect_property_outliers(records, "band_gap_ev")
        assert report.distribution.count == 2
        assert len(report.warnings) >= 1
        assert any("mp-2" in w for w in report.warnings)

    def test_all_none_returns_empty(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=None),
            _rec("mp-2", "B", band_gap=None),
        ]
        report = detect_property_outliers(records, "band_gap_ev")
        assert report.distribution.count == 0
        assert len(report.records) == 0

    def test_single_value_no_std(self) -> None:
        records = [_rec("mp-1", "A", band_gap=2.5)]
        report = detect_property_outliers(records, "band_gap_ev")
        assert report.distribution.std == 0.0


# ── Property validation ─────────────────────────────────────────────────────


class TestPropertyValidation:
    def test_invalid_property_raises(self) -> None:
        with pytest.raises(ValueError, match="not in allowed set"):
            detect_property_outliers([], "color")

    def test_invalid_univariate_method_raises(self) -> None:
        records = [_rec()]
        with pytest.raises(ValueError, match="not allowed"):
            detect_property_outliers(records, "band_gap_ev", method="median")

    def test_invalid_multivariate_method_raises(self) -> None:
        records = [_rec(), _rec("mp-2", "B"), _rec("mp-3", "C")]
        with pytest.raises(ValueError, match="not allowed"):
            detect_multivariate_outliers(
                records, ("band_gap_ev", "density_g_cm3"), method="pca"
            )


# ── detect_multivariate_outliers — Mahalanobis ──────────────────────────────


class TestMultivariateMahalanobis:
    def test_basic_detection(self) -> None:
        # 9 normal + 1 clear outlier — enough data to avoid masking effect
        records = [
            _rec("mp-1", "A", band_gap=2.0, density=4.0, hull=0.05),
            _rec("mp-2", "B", band_gap=1.5, density=4.5, hull=0.08),
            _rec("mp-3", "C", band_gap=2.5, density=3.5, hull=0.02),
            _rec("mp-4", "D", band_gap=1.8, density=3.8, hull=0.06),
            _rec("mp-5", "E", band_gap=2.3, density=4.2, hull=0.04),
            _rec("mp-7", "F", band_gap=1.9, density=4.1, hull=0.07),
            _rec("mp-8", "G", band_gap=2.2, density=3.9, hull=0.03),
            _rec("mp-9", "H", band_gap=1.7, density=4.3, hull=0.09),
            _rec("mp-10", "I", band_gap=2.4, density=3.7, hull=0.01),
            _rec("mp-6", "X", band_gap=9.0, density=1.5, hull=5.0),  # outlier
        ]
        report = detect_multivariate_outliers(
            records,
            ("band_gap_ev", "density_g_cm3", "energy_above_hull_ev_atom"),
            method="mahalanobis",
        )
        assert isinstance(report, MultivariateOutlierReport)
        assert report.method == "mahalanobis"
        assert report.threshold is not None
        assert report.threshold > 0
        assert len(report.records) == 10
        # Outlier should have the highest anomaly score and be flagged
        scores = [r.anomaly_score for r in report.records]
        max_idx = scores.index(max(scores))
        assert report.records[max_idx].material_id == "mp-6"
        assert report.records[max_idx].is_outlier is True
        # Normal records should not be flagged
        normal = [r for r in report.records if r.material_id != "mp-6"]
        assert all(not r.is_outlier for r in normal)

    def test_output_structure(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=1.0, density=4.0, hull=0.05),
            _rec("mp-2", "B", band_gap=1.2, density=3.8, hull=0.06),
            _rec("mp-3", "C", band_gap=1.1, density=4.2, hull=0.04),
        ]
        report = detect_multivariate_outliers(
            records,
            ("band_gap_ev", "density_g_cm3"),
            method="mahalanobis",
        )
        for r in report.records:
            assert r.material_label
            assert r.anomaly_score >= 0
            assert isinstance(r.is_outlier, bool)
            assert isinstance(r.outlier_reason, str)
            assert len(r.outlier_reason) > 0

    def test_too_few_properties_raises(self) -> None:
        records = [_rec(), _rec("mp-2", "B"), _rec("mp-3", "C")]
        with pytest.raises(ValueError, match="at least 2"):
            detect_multivariate_outliers(records, ("band_gap_ev",))

    def test_too_many_properties_raises(self) -> None:
        records = [_rec(), _rec("mp-2", "B"), _rec("mp-3", "C")]
        with pytest.raises(ValueError, match="at most 4"):
            detect_multivariate_outliers(
                records,
                ("band_gap_ev", "density_g_cm3", "formation_energy_ev_atom",
                 "energy_above_hull_ev_atom", "extra"),
            )


# ── detect_multivariate_outliers — Isolation Forest ─────────────────────────


class TestMultivariateIsolationForest:
    def test_subsampling_is_without_replacement_and_seeded(self) -> None:
        first = _deterministic_sample(300, 256, seed=17)
        second = _deterministic_sample(300, 256, seed=17)
        assert first == second
        assert len(first) == 256
        assert len(set(first)) == 256
        assert max(first) >= 256

    def test_basic_detection(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=2.0, density=4.0, hull=0.05),
            _rec("mp-2", "B", band_gap=2.1, density=3.9, hull=0.04),
            _rec("mp-3", "C", band_gap=1.9, density=4.1, hull=0.06),
            _rec("mp-4", "D", band_gap=2.0, density=4.0, hull=0.05),
            _rec("mp-5", "E", band_gap=2.2, density=3.8, hull=0.03),
            _rec("mp-6", "X", band_gap=5.0, density=1.0, hull=2.0),
        ]
        report = detect_multivariate_outliers(
            records,
            ("band_gap_ev", "density_g_cm3", "energy_above_hull_ev_atom"),
            method="isolation_forest",
        )
        assert report.method == "isolation_forest"
        assert len(report.records) == 6
        # All scores should be between 0 and 1
        for r in report.records:
            assert 0.0 <= r.anomaly_score <= 1.0
            assert isinstance(r.is_outlier, bool)
        # mp-6 (outlier) should have the highest score and be flagged
        scores = [r.anomaly_score for r in report.records]
        max_idx = scores.index(max(scores))
        assert report.records[max_idx].material_id == "mp-6"
        assert report.records[max_idx].is_outlier is True

    def test_deterministic(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=2.0, density=4.0),
            _rec("mp-2", "B", band_gap=2.1, density=3.9),
            _rec("mp-3", "C", band_gap=1.9, density=4.1),
        ]
        r1 = detect_multivariate_outliers(
            records, ("band_gap_ev", "density_g_cm3"), method="isolation_forest"
        )
        r2 = detect_multivariate_outliers(
            records, ("band_gap_ev", "density_g_cm3"), method="isolation_forest"
        )
        for a, b in zip(r1.records, r2.records, strict=True):
            assert a.anomaly_score == b.anomaly_score


# ── Missing values in multivariate ──────────────────────────────────────────


class TestMultivariateMissing:
    def test_missing_values_excluded(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=2.0, density=4.0),
            _rec("mp-2", "B", band_gap=None, density=3.9),  # excluded — missing
            _rec("mp-3", "C", band_gap=1.9, density=4.1),
            _rec("mp-4", "D", band_gap=2.3, density=3.6),
            _rec("mp-5", "E", band_gap=1.7, density=4.3),
        ]
        report = detect_multivariate_outliers(
            records, ("band_gap_ev", "density_g_cm3")
        )
        assert len(report.records) == 4
        assert len(report.warnings) >= 1

    def test_insufficient_complete_cases(self) -> None:
        records = [
            _rec("mp-1", "A", band_gap=None, density=4.0),
            _rec("mp-2", "B", band_gap=2.0, density=None),
        ]
        report = detect_multivariate_outliers(
            records, ("band_gap_ev", "density_g_cm3")
        )
        assert len(report.records) == 0
        assert any("insufficient" in w.lower() for w in report.warnings)
