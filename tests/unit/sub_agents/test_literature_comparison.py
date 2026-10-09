from __future__ import annotations

import hashlib

import pytest

from materials_screening.sub_agents.literature.comparison import calculate_comparison
from materials_screening.sub_agents.literature.matrix import validate_matrix_evidence
from materials_screening.sub_agents.literature.models import (
    ExperimentalGroup,
    ExperimentalMeasurement,
)
from materials_screening.sub_agents.literature.rag import ChunkRecord


def _measurement(
    measurement_id: str,
    group_id: str,
    value: float,
    *,
    unit: str = "%",
) -> ExperimentalMeasurement:
    quote = f"conversion efficiency {value:g} {unit}"
    return ExperimentalMeasurement(
        measurement_id=measurement_id,
        group_id=group_id,
        document_id="doc-1",
        metric="conversion efficiency",
        value_text=f"{value:g}",
        numeric_value=value,
        unit=unit,
        source_quote=quote,
        chunk_id="chunk-1",
        page_from=6,
        page_to=6,
        source_text_sha256="a" * 64,
    )


def test_calculated_comparison_preserves_absolute_and_relative_change() -> None:
    result = calculate_comparison(
        _measurement("m-control", "g-control", 1.1),
        _measurement("m-treated", "g-treated", 1.3),
    )
    assert result.absolute_change == pytest.approx(0.2)
    assert result.relative_change_percent == pytest.approx(18.181818)
    assert result.direction == "increase"
    assert result.provenance_type == "calculated"


def test_comparison_rejects_different_units() -> None:
    with pytest.raises(ValueError, match="units differ"):
        calculate_comparison(
            _measurement("m-1", "g-1", 1.0, unit="%"),
            _measurement("m-2", "g-2", 1.0, unit="V"),
        )


class ChunkStore:
    def __init__(self, chunk: ChunkRecord) -> None:
        self.chunk = chunk

    def get_chunks(self, chunk_ids: tuple[str, ...]) -> list[ChunkRecord]:
        return [self.chunk] if self.chunk.chunk_id in chunk_ids else []


def test_matrix_requires_every_variable_and_value_in_exact_evidence() -> None:
    text = "Nb 1.5 mol % produced conversion efficiency 1.3 %."
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=6,
        page_to=6,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    group = ExperimentalGroup(
        group_id="g-treated",
        document_id="doc-1",
        label="Nb 1.5",
        role="treatment",
        material="Nb-doped TiO2",
        variables={"Nb doping": "1.5 mol %"},
        source_quote=text,
        chunk_id="chunk-1",
        page_from=6,
        page_to=6,
        source_text_sha256=chunk.text_sha256,
    )
    measurement = ExperimentalMeasurement(
        measurement_id="m-treated",
        group_id=group.group_id,
        document_id="doc-1",
        metric="conversion efficiency",
        value_text="1.3",
        numeric_value=1.3,
        unit="%",
        source_quote=text,
        chunk_id="chunk-1",
        page_from=6,
        page_to=6,
        source_text_sha256=chunk.text_sha256,
    )
    validate_matrix_evidence(
        ChunkStore(chunk),  # type: ignore[arg-type]
        document_id="doc-1",
        groups=(group,),
        measurements=(measurement,),
    )
    bad = group.model_copy(update={"variables": {"Nb doping": "9.9 mol %"}})
    with pytest.raises(ValueError, match="variable or condition is absent"):
        validate_matrix_evidence(
            ChunkStore(chunk),  # type: ignore[arg-type]
            document_id="doc-1",
            groups=(bad,),
            measurements=(measurement,),
        )


def test_matrix_rejects_forged_hash_pages_and_units() -> None:
    text = "Sample A at 25 °C reached strength 12.0 ± 0.4 MPa."
    chunk = ChunkRecord(
        chunk_id="chunk-1",
        document_id="doc-1",
        paper_id=None,
        page_from=4,
        page_to=4,
        text=text,
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )
    group = ExperimentalGroup(
        group_id="g-1",
        document_id="doc-1",
        label="Sample A",
        role="treatment",
        material="ceramic",
        conditions={"temperature": "25 °C"},
        source_quote=text,
        chunk_id="chunk-1",
        page_from=4,
        page_to=4,
        source_text_sha256=chunk.text_sha256,
    )
    measurement = ExperimentalMeasurement(
        measurement_id="m-1",
        group_id="g-1",
        document_id="doc-1",
        metric="strength",
        value_text="12.0",
        numeric_value=12.0,
        unit="MPa",
        uncertainty_text="± 0.4",
        source_quote=text,
        chunk_id="chunk-1",
        page_from=4,
        page_to=4,
        source_text_sha256=chunk.text_sha256,
    )
    validate_matrix_evidence(
        ChunkStore(chunk),  # type: ignore[arg-type]
        document_id="doc-1",
        groups=(group,),
        measurements=(measurement,),
    )
    with pytest.raises(ValueError, match="hash"):
        validate_matrix_evidence(
            ChunkStore(chunk),  # type: ignore[arg-type]
            document_id="doc-1",
            groups=(group.model_copy(update={"source_text_sha256": "b" * 64}),),
            measurements=(measurement,),
        )
    with pytest.raises(ValueError, match="measurement unit is absent from evidence"):
        validate_matrix_evidence(
            ChunkStore(chunk),  # type: ignore[arg-type]
            document_id="doc-1",
            groups=(group,),
            measurements=(measurement.model_copy(update={"unit": "GPa"}),),
        )
