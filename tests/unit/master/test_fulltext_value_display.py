"""Display formatting must not alter reported source values."""

import pytest

from materials_screening.master.fulltext_analysis import _measurement_value_cell


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        ("113 AW−1", "AW−1", "113 AW−1"),
        ("113 AW − 1", "AW−1", "113 AW − 1"),
        ("113", "AW−1", "113 AW−1"),
        ("385", None, "385"),
        ("385", "", "385"),
        ("113 mAW−1", "AW−1", "113 mAW−1 AW−1"),
        ("3.4 eV", "eV", "3.4 eV"),
        ("a|b\nc", None, "a\\|b c"),
    ],
)
def test_measurement_value_display(value, unit, expected):
    row = {"value_text": value, "unit": unit}
    original = row.copy()
    assert _measurement_value_cell(row) == expected
    assert row == original
