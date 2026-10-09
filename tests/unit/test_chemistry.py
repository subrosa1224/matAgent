"""Unit tests for chemistry helpers (M2)."""

import pytest

from materials_screening.chemistry import (
    normalize_chemsys,
    normalize_element_list,
    normalize_element_symbol,
)


class TestNormalizeElementSymbol:
    def test_lowercase_symbol(self) -> None:
        assert normalize_element_symbol("pb") == "Pb"

    def test_uppercase_symbol(self) -> None:
        assert normalize_element_symbol("PB") == "Pb"

    def test_canonical_symbol_unchanged(self) -> None:
        assert normalize_element_symbol("Fe") == "Fe"

    def test_whitespace_stripped(self) -> None:
        assert normalize_element_symbol("  li  ") == "Li"

    def test_invalid_symbol_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_element_symbol("Xx")

    def test_empty_symbol_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_element_symbol("")


class TestNormalizeElementList:
    def test_dedupes_and_sorts(self) -> None:
        assert normalize_element_list(["O", "Li", "O", "Fe", "FE"]) == ("Fe", "Li", "O")

    def test_empty_list(self) -> None:
        assert normalize_element_list([]) == ()

    def test_invalid_element_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_element_list(["Fe", "Xx"])


class TestNormalizeChemsys:
    def test_normalizes_and_sorts(self) -> None:
        assert normalize_chemsys("O-Li-Fe") == "Fe-Li-O"

    def test_whitespace_between_parts(self) -> None:
        assert normalize_chemsys(" Li - Fe - O ") == "Fe-Li-O"

    def test_none_returns_none(self) -> None:
        assert normalize_chemsys(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert normalize_chemsys("") is None

    def test_whitespace_only_returns_none(self) -> None:
        assert normalize_chemsys("   ") is None

    def test_duplicate_elements_deduped(self) -> None:
        assert normalize_chemsys("Fe-Fe-O") == "Fe-O"

    def test_invalid_element_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_chemsys("Fe-Xx")

    def test_empty_part_raises(self) -> None:
        with pytest.raises(ValueError):
            normalize_chemsys("Fe--O")
