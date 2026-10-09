"""Unit tests for planner unit conversions (D2-M2)."""

from decimal import Decimal

import pytest

from materials_screening.planner.models import (
    DensityUnit,
    EnergyUnit,
    HullUnit,
)
from materials_screening.planner.unit_conversion import (
    UnitConversionError,
    density_to_g_per_cm3,
    energy_to_ev,
    hull_to_ev_per_atom,
)


class TestEnergyToEv:
    def test_ev_passthrough(self) -> None:
        assert energy_to_ev(Decimal("1.2"), EnergyUnit.EV) == Decimal("1.2")

    @pytest.mark.parametrize(
        ("mev", "expected"),
        [
            (Decimal("1000"), Decimal("1")),
            (Decimal("50"), Decimal("0.05")),
            (Decimal("1"), Decimal("0.001")),
            (Decimal("123.456"), Decimal("0.123456")),
            (Decimal("0"), Decimal("0")),
            (Decimal("0.001"), Decimal("0.000001")),
        ],
    )
    def test_mev_to_ev(self, mev: Decimal, expected: Decimal) -> None:
        result = energy_to_ev(mev, EnergyUnit.MEV)
        assert result == expected
        assert isinstance(result, Decimal)

    def test_negative_value_converts_exactly(self) -> None:
        assert energy_to_ev(Decimal("-500"), EnergyUnit.MEV) == Decimal("-0.5")

    def test_unspecified_rejected(self) -> None:
        with pytest.raises(UnitConversionError, match="specified"):
            energy_to_ev(Decimal("1"), EnergyUnit.UNSPECIFIED)

    def test_input_not_mutated(self) -> None:
        value = Decimal("1000")
        energy_to_ev(value, EnergyUnit.MEV)
        assert value == Decimal("1000")


class TestHullToEvPerAtom:
    def test_ev_per_atom_passthrough(self) -> None:
        assert hull_to_ev_per_atom(Decimal("0.05"), HullUnit.EV_PER_ATOM) == Decimal(
            "0.05"
        )

    @pytest.mark.parametrize(
        ("mev", "expected"),
        [
            (Decimal("1000"), Decimal("1")),
            (Decimal("50"), Decimal("0.05")),
            (Decimal("1"), Decimal("0.001")),
        ],
    )
    def test_mev_per_atom_to_ev_per_atom(self, mev: Decimal, expected: Decimal) -> None:
        assert hull_to_ev_per_atom(mev, HullUnit.MEV_PER_ATOM) == expected

    def test_unspecified_rejected(self) -> None:
        with pytest.raises(UnitConversionError, match="specified"):
            hull_to_ev_per_atom(Decimal("50"), HullUnit.UNSPECIFIED)


class TestDensityToGPerCm3:
    def test_g_per_cm3_passthrough(self) -> None:
        assert density_to_g_per_cm3(Decimal("5.2"), DensityUnit.G_PER_CM3) == Decimal(
            "5.2"
        )

    @pytest.mark.parametrize(
        ("kg_m3", "expected"),
        [
            (Decimal("1000"), Decimal("1")),
            (Decimal("500"), Decimal("0.5")),
            (Decimal("1234"), Decimal("1.234")),
            (Decimal("1"), Decimal("0.001")),
            (Decimal("0"), Decimal("0")),
        ],
    )
    def test_kg_per_m3_to_g_per_cm3(self, kg_m3: Decimal, expected: Decimal) -> None:
        assert density_to_g_per_cm3(kg_m3, DensityUnit.KG_PER_M3) == expected

    def test_unspecified_rejected(self) -> None:
        with pytest.raises(UnitConversionError, match="specified"):
            density_to_g_per_cm3(Decimal("1000"), DensityUnit.UNSPECIFIED)


class TestDecimalPrecision:
    def test_no_float_rounding(self) -> None:
        result = energy_to_ev(Decimal("1"), EnergyUnit.MEV)
        assert result == Decimal("0.001")
        assert result.as_tuple().exponent == -3

    def test_exact_decimal_division(self) -> None:
        assert energy_to_ev(Decimal("1001"), EnergyUnit.MEV) == Decimal("1.001")
