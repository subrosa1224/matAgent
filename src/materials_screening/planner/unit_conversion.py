"""Deterministic unit conversions for the planner (D2-M2).

Conversions are performed in Python with ``Decimal``; the model never
converts units. Unspecified units are rejected here and left to the
Resolver to assign a default or ask for clarification.
"""

from decimal import Decimal

from materials_screening.planner.models import (
    DensityUnit,
    EnergyUnit,
    HullUnit,
)

_KILO = Decimal(1000)


class UnitConversionError(ValueError):
    """Raised when a unit is unspecified or unsupported."""


def energy_to_ev(value: Decimal, unit: EnergyUnit) -> Decimal:
    """Convert an energy value to eV."""
    if unit is EnergyUnit.EV:
        return value
    if unit is EnergyUnit.MEV:
        return value / _KILO
    raise UnitConversionError(f"energy unit must be specified, got {unit!r}")


def hull_to_ev_per_atom(value: Decimal, unit: HullUnit) -> Decimal:
    """Convert an energy-above-hull value to eV/atom."""
    if unit is HullUnit.EV_PER_ATOM:
        return value
    if unit is HullUnit.MEV_PER_ATOM:
        return value / _KILO
    raise UnitConversionError(f"hull unit must be specified, got {unit!r}")


def density_to_g_per_cm3(value: Decimal, unit: DensityUnit) -> Decimal:
    """Convert a density value to g/cm3."""
    if unit is DensityUnit.G_PER_CM3:
        return value
    if unit is DensityUnit.KG_PER_M3:
        return value / _KILO
    raise UnitConversionError(f"density unit must be specified, got {unit!r}")
