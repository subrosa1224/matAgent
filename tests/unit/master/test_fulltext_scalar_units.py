"""Only exact declared unit suffixes may accompany trial scalars."""

import pytest

from materials_screening.master.fulltext_handoff import _reported_scalar


@pytest.mark.parametrize(
    "text,unit,value",
    [
        ("113 AW−1", "AW−1", 113),
        ("113 AW − 1", "AW−1", 113),
        ("12.0 MPa", "MPa", 12),
        ("about 12.0 MPa", "MPa", 12),
        ("385", None, 385),
        ("−1 V", "V", -1),
    ],
)
def test_scalar_with_matching_declared_unit(text, unit, value):
    assert _reported_scalar(text, unit) == value


@pytest.mark.parametrize(
    "text,unit",
    [
        ("113 A/W", "AW−1"),
        ("113 mA/W", "A/W"),
        ("113 AW−1", None),
        ("12-14 MPa", "MPa"),
        (">12 MPa", "MPa"),
        ("12 ± 1 MPa", "MPa"),
        ("12 MPa extra", "MPa"),
        ("12 mpa", "MPa"),
        ("1e3 MPa", "MPa"),
    ],
)
def test_non_scalars_or_conflicting_units_rejected(text, unit):
    with pytest.raises(ValueError):
        _reported_scalar(text, unit)
