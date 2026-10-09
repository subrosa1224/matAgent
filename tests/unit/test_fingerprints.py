"""Unit tests for request fingerprints (M2)."""

from materials_screening.fingerprints import request_fingerprint
from materials_screening.models import FloatRange, ScreeningRequest


def test_fingerprint_is_sha256_hex() -> None:
    fingerprint = request_fingerprint(ScreeningRequest())
    assert len(fingerprint) == 64
    int(fingerprint, 16)


def test_field_order_does_not_matter() -> None:
    first = ScreeningRequest.model_validate(
        {
            "required_elements": ["Fe"],
            "band_gap_ev": {"min": 1.2, "max": 2.0},
            "limit": 20,
        }
    )
    second = ScreeningRequest.model_validate(
        {
            "limit": 20,
            "band_gap_ev": {"max": 2.0, "min": 1.2},
            "required_elements": ["Fe"],
        }
    )
    assert request_fingerprint(first) == request_fingerprint(second)


def test_element_input_order_does_not_matter() -> None:
    first = ScreeningRequest(required_elements=("O", "Fe", "Li"))
    second = ScreeningRequest(required_elements=("Li", "O", "Fe"))
    assert request_fingerprint(first) == request_fingerprint(second)


def test_different_thresholds_produce_different_fingerprints() -> None:
    first = ScreeningRequest(band_gap_ev=FloatRange(min=1.2, max=2.0))
    second = ScreeningRequest(band_gap_ev=FloatRange(min=1.3, max=2.0))
    assert request_fingerprint(first) != request_fingerprint(second)


def test_none_and_absent_fields_are_equivalent() -> None:
    explicit_none = ScreeningRequest(chemsys=None, formula=None)
    omitted = ScreeningRequest()
    assert request_fingerprint(explicit_none) == request_fingerprint(omitted)


def test_explicit_defaults_equivalent_to_omitted() -> None:
    explicit = ScreeningRequest(limit=10, is_metal=False, spacegroup_numbers=())
    omitted = ScreeningRequest()
    assert request_fingerprint(explicit) == request_fingerprint(omitted)


def test_empty_excluded_elements_equivalent_to_omitted() -> None:
    explicit = ScreeningRequest(excluded_elements=())
    omitted = ScreeningRequest()
    assert request_fingerprint(explicit) == request_fingerprint(omitted)
