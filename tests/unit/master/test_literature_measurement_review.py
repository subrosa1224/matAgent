from __future__ import annotations

import json
from pathlib import Path

import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.literature_evidence_trial import (
    LiteratureEvidenceTrialService,
)
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.user_report import (
    LiteratureUserReport,
    UserPaperReport,
)


def _group(group_id: str = "g1") -> ExperimentalGroup:
    return ExperimentalGroup(
        group_id=group_id,
        document_id="d1",
        label="Graphene/ZnONR",
        role="treatment",
        material="Graphene/ZnONR",
        variables={},
        conditions={},
        source_quote="Graphene/ZnONR",
        chunk_id="c1",
        page_from=4,
        page_to=4,
        source_text_sha256="a" * 64,
    )


def _measurement(
    identifier: str,
    metric: str,
    value: float,
    quote: str,
    *,
    group_id: str = "g1",
    unit: str | None = "AW⁻¹",
    chunk_id: str = "c1",
    page: int = 4,
) -> ExperimentalMeasurement:
    return ExperimentalMeasurement(
        measurement_id=identifier,
        document_id="d1",
        group_id=group_id,
        metric=metric,
        value_text=str(value),
        numeric_value=value,
        unit=unit,
        source_quote=quote,
        chunk_id=chunk_id,
        page_from=page,
        page_to=page,
        source_text_sha256="a" * 64,
    )


def _prepare(tmp_path: Path, rows, groups=None):
    report = LiteratureUserReport(
        report_id="user-lit-review",
        topic="UV detector",
        papers=(
            UserPaperReport(
                document_id="d1",
                title="UV paper",
                dossier_status="pending",
                evidence=(),
                low_risk_count=0,
                medium_risk_count=0,
                excluded_high_risk_count=0,
                excluded_unsafe_count=0,
            ),
        ),
    )

    class Reports:
        def find_latest(self, topic):
            return report

    class Matrices:
        def load_matrix(self, document_id, *, status="approved"):
            return (
                (groups or [_group()], rows, [], [], [])
                if status == "pending"
                else ([], [], [], [], [])
            )

    datasets = DatasetStore(tmp_path / "datasets")
    handoff = LiteratureEvidenceTrialService(
        report_store=Reports(),
        matrix_store=Matrices(),
        dataset_store=datasets,
    ).prepare(topic=report.topic)
    return handoff, datasets


def test_exact_duplicate_aliases_keep_all_provenance_without_replicate_inflation(
    tmp_path,
):
    quote = "Responsivity R is 113 AW − 1."
    rows = [
        _measurement("m1", "Responsivity (R)", 113, quote),
        _measurement("m2", "responsivity", 113, quote),
    ]
    handoff, datasets = _prepare(tmp_path, rows)
    assert handoff.record_count == 1
    row = datasets.load_dataframe(handoff.dataset_id).iloc[0]
    assert row["metric"] == "responsivity"
    assert set(json.loads(row["source_measurement_ids"])) == {"m1", "m2"}
    assert json.loads(row["source_quotes"]) == [quote, quote]
    assert any("不是独立实验重复数" in warning for warning in handoff.warnings)
    assert rows[0].metric == "Responsivity (R)"


def test_table_reference_proves_repeated_summary_not_two_experiments(tmp_path):
    body = (
        "The R and G are estimated to be 113 AW − 1 and 385, respectively. "
        "These two values are summarized in Table 1."
    )
    table = (
        "Table 1. R [AW⁻¹] G Graphene/ZnONR Schottky junction "
        "365 0.7 3.6 113 385 This work"
    )
    rows = [
        _measurement("r1", "Responsivity (R)", 113, body),
        _measurement("r2", "responsivity", 113, table, chunk_id="c2", page=5),
        _measurement("g1", "Gain (G)", 385, body, unit=None),
        _measurement("g2", "gain", 385, table, unit=None, chunk_id="c2", page=5),
    ]
    handoff, datasets = _prepare(tmp_path, rows)
    assert handoff.record_count == 2
    frame = datasets.load_dataframe(handoff.dataset_id)
    assert set(frame.metric) == {"responsivity", "gain"}
    assert all(frame.evidence_count == 2)
    assert all(frame.duplicate_resolution == "explicit_table_reference")
    assert {"c1", "c2"}.issubset(handoff.evidence_ids)


def test_table_reference_does_not_prove_a_value_outside_the_explicit_rg_pair(tmp_path):
    body = (
        "A different device had R 113 AW⁻¹. The R and G are estimated to be "
        "130 AW⁻¹ and 400, respectively. These two values are in Table 1."
    )
    table = "Table 1. R [AW⁻¹] G Graphene/ZnONR 113 385 This work"
    handoff, _ = _prepare(
        tmp_path,
        [
            _measurement("m1", "responsivity", 113, body),
            _measurement("m2", "Responsivity (R)", 113, table, chunk_id="c2"),
        ],
    )
    assert handoff.dataset_id is None
    assert any("疑似重复" in warning for warning in handoff.warnings)


def test_a_literature_reference_table_is_not_the_own_study_summary(tmp_path):
    body = (
        "The R and G are estimated to be 113 AW⁻¹ and 385, respectively. "
        "Compare these two values with Table 1."
    )
    handoff, _ = _prepare(
        tmp_path,
        [
            _measurement("m1", "responsivity", 113, body),
            _measurement(
                "m2",
                "Responsivity (R)",
                113,
                "Table 1. Reference [20] R 113",
                chunk_id="c2",
            ),
        ],
    )
    assert handoff.dataset_id is None


def test_equal_numbers_without_shared_evidence_are_only_suspected_duplicates(tmp_path):
    rows = [
        _measurement("m1", "responsivity", 113, "Experiment A: R 113 AW⁻¹."),
        _measurement(
            "m2", "Responsivity (R)", 113, "Experiment B: R 113 AW⁻¹.", chunk_id="c2"
        ),
    ]
    handoff, _ = _prepare(tmp_path, rows)
    assert handoff.dataset_id is None
    assert any("疑似重复" in warning for warning in handoff.warnings)
    assert "m1" in handoff.literature_markdown and "m2" in handoff.literature_markdown


def test_same_quote_with_multiple_experiments_does_not_prove_duplicates(tmp_path):
    quote = "Experiment A had R 113 AW⁻¹; experiment B had R 113 AW⁻¹."
    handoff, _ = _prepare(
        tmp_path,
        [
            _measurement("m1", "responsivity", 113, quote),
            _measurement("m2", "Responsivity (R)", 113, quote),
        ],
    )
    assert handoff.dataset_id is None
    assert any("疑似重复" in warning for warning in handoff.warnings)


def test_identical_quote_with_different_source_hash_does_not_prove_duplicates(tmp_path):
    first = _measurement("m1", "responsivity", 113, "R is 113 AW⁻¹.")
    second = first.model_copy(
        update={"measurement_id": "m2", "source_text_sha256": "b" * 64}
    )
    handoff, _ = _prepare(tmp_path, [first, second])
    assert handoff.dataset_id is None


@pytest.mark.parametrize("difference", ["group", "unit", "uncertainty", "sample_size"])
def test_different_groups_units_or_measurement_metadata_are_not_merged(
    tmp_path, difference
):
    quote = "Responsivity was 113 AW⁻¹ and 113 mAW⁻¹."
    first = _measurement("m1", "responsivity", 113, quote)
    changes = {
        "group": {"group_id": "g2"},
        "unit": {"unit": "mAW⁻¹"},
        "uncertainty": {"uncertainty_text": "±2"},
        "sample_size": {"sample_size": 3},
    }
    second = first.model_copy(update={"measurement_id": "m2", **changes[difference]})
    handoff, _ = _prepare(tmp_path, [first, second], [_group(), _group("g2")])
    assert handoff.record_count == 2


def test_conflicting_response_time_order_is_visible_but_not_analyzed(tmp_path):
    table = "Table 1. Graphene/ZnONR rise/fall time: 0.7/3.6 ms This work"
    body = "The fall and rise time are derived to be 0.7 and 3.6 ms, respectively."
    rows = [
        _measurement("rise", "response_time_rise", 0.7, table, unit="ms"),
        _measurement("fall", "response_time_fall", 3.6, table, unit="ms"),
        _measurement(
            "fall2", "time_response_fall", 3.6, body, unit="ms", chunk_id="c2"
        ),
        _measurement("r", "responsivity", 113, "R is 113 AW⁻¹."),
    ]
    handoff, datasets = _prepare(tmp_path, rows)
    assert handoff.record_count == 1
    assert datasets.load_dataframe(handoff.dataset_id).metric.tolist() == [
        "responsivity"
    ]
    assert any("冲突" in warning for warning in handoff.warnings)
    assert "0.7" in handoff.literature_markdown and "3.6" in handoff.literature_markdown
    assert "fall and rise" in handoff.literature_markdown
    assert (
        "rise" in handoff.literature_markdown and "fall2" in handoff.literature_markdown
    )


def test_consistent_response_time_order_is_retained(tmp_path):
    quote = "The rise and fall time are 0.7 and 3.6 ms, respectively."
    handoff, _ = _prepare(
        tmp_path,
        [
            _measurement("rise", "response_time_rise", 0.7, quote, unit="ms"),
            _measurement("fall", "response_time_fall", 3.6, quote, unit="ms"),
        ],
    )
    assert handoff.record_count == 2
    assert not any("冲突" in warning for warning in handoff.warnings)


def test_visible_light_seconds_do_not_conflict_with_uv_milliseconds(tmp_path):
    quote = (
        "The rise and fall time are 0.7 and 3.6 ms, respectively. "
        "Under visible light the rise and fall time are 0.32 and 1.5 s, respectively."
    )
    handoff, _ = _prepare(
        tmp_path,
        [
            _measurement("rise", "response_time_rise", 0.7, quote, unit="ms"),
            _measurement("fall", "response_time_fall", 3.6, quote, unit="ms"),
        ],
    )
    assert handoff.record_count == 2
    assert not any("冲突" in warning for warning in handoff.warnings)


def test_conflicting_values_are_not_averaged(tmp_path):
    handoff, _ = _prepare(
        tmp_path,
        [
            _measurement("m1", "responsivity", 113, "R is 113 AW⁻¹."),
            _measurement(
                "m2", "Responsivity (R)", 130, "R is 130 AW⁻¹.", chunk_id="c2"
            ),
        ],
    )
    assert handoff.dataset_id is None
    assert any("冲突" in warning for warning in handoff.warnings)
    assert "113" in handoff.literature_markdown and "130" in handoff.literature_markdown
