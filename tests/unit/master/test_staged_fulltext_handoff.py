import pytest

from materials_screening.data_analysis.dataset_store import DatasetStore
from materials_screening.master.staged_fulltext_engine import StagedFulltextEngine
from materials_screening.master.staged_fulltext_handoff import (
    render_staged_result,
    staged_trial_handoff,
)
from materials_screening.master.staged_fulltext_store import StagedExtractionStore
from tests.unit.master.test_staged_fulltext_engine import Provider, calls, inputs


def run(tmp_path):
    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path / "stages"), model_profile="offline/unit"
    )
    refs = []
    record = engine.run_document(
        **inputs(), calls=calls(Provider()), on_checkpoint=refs.append
    )
    return engine, record, refs[-1], DatasetStore(tmp_path / "datasets")


def test_composite_handoff_retains_two_sources_and_unknown_conditions(tmp_path):
    _, record, ref, datasets = run(tmp_path)
    handoff = staged_trial_handoff(
        record,
        reference=ref,
        chunks=inputs()["chunks"],
        dataset_factory=lambda: datasets,
    )
    frame = datasets.load_dataframe(handoff.dataset_id)
    assert len(frame) == handoff.record_count == 1
    row = frame.iloc[0]
    assert row["numeric_value"] == 82
    assert row["value_text"] == "around 82"
    assert row["sample_source_chunk_id"] != row["source_chunk_id"]
    assert row["review_status"] == "pending"
    assert not row["independent_replicates_confirmed"]
    assert '"unknown"' in row["condition_status"]
    assert '"can_rank": false' in row["comparison_policy"]
    second = staged_trial_handoff(
        record,
        reference=ref,
        chunks=inputs()["chunks"],
        dataset_factory=lambda: datasets,
        existing=handoff,
    )
    assert second == handoff


def test_user_summary_shows_scientific_boundaries_not_engineer_details(tmp_path):
    _, record, ref, datasets = run(tmp_path)
    handoff = staged_trial_handoff(
        record,
        reference=ref,
        chunks=inputs()["chunks"],
        dataset_factory=lambda: datasets,
    )
    text = render_staged_result(record, handoff)
    assert "NTO3" in text and "约82" in text
    assert "UV-visible" in text and "电阻率" in text
    assert "无法直接排名" in text
    assert "尚待核对" in text
    assert not any(
        t in text
        for t in (
            "staged-",
            "snapshot-",
            "measurement-",
            "FULLTEXT_",
            "trial",
            "pending",
        )
    )


def test_source_change_blocks_handoff_before_dataset_creation(tmp_path):
    from dataclasses import replace

    _, record, ref, _ = run(tmp_path)
    chunks = inputs()["chunks"]
    with pytest.raises(ValueError):
        staged_trial_handoff(
            record,
            reference=ref,
            chunks=(chunks[0], replace(chunks[1], text="changed")),
            dataset_factory=lambda: pytest.fail("Must not create a tampered dataset"),
        )


@pytest.mark.parametrize("existing", [False, True])
def test_saved_identity_condition_cannot_reach_dataset_or_cached_handoff(
    tmp_path, existing
):
    from uuid import uuid4

    from materials_screening.master.fulltext_conditions import (
        AttributeBinding,
        ConditionPlan,
    )

    engine, record, ref, datasets = run(tmp_path)
    old = (
        staged_trial_handoff(
            record,
            reference=ref,
            chunks=inputs()["chunks"],
            dataset_factory=lambda: datasets,
        )
        if existing
        else None
    )
    metric = record.measurements[0]
    bad = AttributeBinding(
        measurement_ids=(metric.measurement_id,),
        kind="condition",
        key="sample_type",
        value_text="NTO3",
        value_source=metric.source,
        applicability="per_sample",
        applicability_source=metric.source,
    )
    record = record.model_copy(
        update={
            "record_id": "staged-" + uuid4().hex,
            "condition_plans": (
                *record.condition_plans,
                ConditionPlan(bindings=(bad,)),
            ),
        }
    )
    ref = engine.store.save(record)
    if old:
        old = old.model_copy(update={"reference": ref})
    with pytest.raises(ValueError, match="identity"):
        engine._revalidate(record, inputs()["chunks"])
    with pytest.raises(ValueError, match="identity"):
        staged_trial_handoff(
            record,
            reference=ref,
            chunks=inputs()["chunks"],
            existing=old,
            dataset_factory=lambda: pytest.fail(
                "Identity conditions must be rejected before dataset access"
            ),
        )


@pytest.mark.parametrize(
    "literal", ["more than 85", "above 85", "below 85", "up to 74", "80-82"]
)
def test_verified_bounds_and_ranges_remain_visible_without_scalar_statistics(
    tmp_path, literal
):
    from dataclasses import replace

    from materials_screening.master.staged_fulltext_evidence import (
        MetricProposal,
        MetricsProposal,
    )
    from tests.unit.master.test_staged_fulltext_evidence import source

    args = inputs()
    args["chunks"] = (
        args["chunks"][0],
        replace(
            source(
                f"NTO3 transmittance was {literal}% in the visible region.", "bound"
            ),
            document_id=args["document_id"],
        ),
    )

    class BoundProvider(Provider):
        def generate_structured(self, **kwargs):
            if kwargs["output_model"] is not MetricsProposal:
                return super().generate_structured(**kwargs)
            import json

            from materials_screening.llm.base import StructuredProviderResponse

            payload = json.loads(kwargs["user_text"])
            return StructuredProviderResponse(
                parsed=MetricsProposal(
                    measurements=(
                        MetricProposal(
                            sample_id=payload["samples"][0]["sample_id"],
                            metric="transmittance",
                            value_text=literal,
                            unit="%",
                            source_id=next(iter(payload["evidence"])),
                        ),
                    )
                ),
                provider="offline",
                model="unit",
                latency_ms=0,
                request_id=None,
                input_tokens=1,
                output_tokens=1,
                reasoning_tokens=0,
                raw_output_sha256=None,
            )

    engine = StagedFulltextEngine(
        store=StagedExtractionStore(tmp_path / "stages"), model_profile="offline/unit"
    )
    refs = []
    record = engine.run_document(
        **args, calls=calls(BoundProvider()), on_checkpoint=refs.append
    )
    handoff = staged_trial_handoff(
        record,
        reference=refs[-1],
        chunks=args["chunks"],
        dataset_factory=lambda: pytest.fail(
            "A bound/range must not form a scalar dataset"
        ),
    )
    assert handoff.record_count == 0 and len(handoff.isolated_measurement_ids) == 1
    text = render_staged_result(record, handoff)
    expected = {
        "more than 85": "大于85%",
        "above 85": "高于85%",
        "below 85": "低于85%",
        "up to 74": "up to 74%",
        "80-82": "80-82%",
    }
    assert expected[literal] in text
    assert "不按单一数值进行描述统计" in text
    assert "原文提到可见光区，具体波长范围尚待核对" in text
    if literal.startswith("up to"):
        assert "原文限定表述“up to 74%”" in text
        assert "不能直接解释为精确值或明确上/下界" in text
