"""Tests for the dedicated Materials Database Agent UI formatting."""

import pytest

pytest.importorskip("gradio")

from materials_screening.materials_database_ui_gradio import (  # noqa: E402
    _format_chemical_formulas,
)


def test_formula_column_uses_subscripts_without_changing_numeric_fields() -> None:
    answer = (
        "| 材料 ID | 化学式 | 带隙 (eV) |\n"
        "| --- | --- | --- |\n"
        "| mp-1234 | Li2FeO3 | 2.15 |\n"
    )

    rendered = _format_chemical_formulas(answer)

    assert "Li₂FeO₃" in rendered
    assert "mp-1234" in rendered
    assert "2.15" in rendered


def test_labeled_formula_uses_subscripts() -> None:
    answer = "化学式：SiO2\n密度：2.33 g/cm3"

    rendered = _format_chemical_formulas(answer)

    assert "化学式：SiO₂" in rendered
    assert "2.33 g/cm3" in rendered
