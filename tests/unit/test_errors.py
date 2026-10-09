"""Unit tests for the project exception hierarchy (M2)."""

import pytest

from materials_screening.errors import InvalidRequestError, MaterialsScreeningError


def test_invalid_request_error_is_project_error() -> None:
    assert issubclass(InvalidRequestError, MaterialsScreeningError)


def test_base_error_is_exception() -> None:
    assert issubclass(MaterialsScreeningError, Exception)


def test_invalid_request_error_carries_message() -> None:
    with pytest.raises(InvalidRequestError, match="bad request"):
        raise InvalidRequestError("bad request")
