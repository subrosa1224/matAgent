"""Chemistry helpers: element and chemsys normalization (M2)."""

from collections.abc import Iterable

from pymatgen.core.periodic_table import Element


def normalize_element_symbol(symbol: str) -> str:
    """Return the canonical element symbol, e.g. ``pb`` -> ``Pb``."""
    cleaned = symbol.strip()
    if not cleaned:
        raise ValueError("Element symbol must not be empty")
    try:
        element = Element(cleaned.capitalize())
    except ValueError as exc:
        raise ValueError(f"Invalid element symbol: {symbol!r}") from exc
    return element.symbol


def normalize_element_list(elements: Iterable[str]) -> tuple[str, ...]:
    """Normalize, deduplicate and sort element symbols."""
    return tuple(sorted({normalize_element_symbol(symbol) for symbol in elements}))


def normalize_chemsys(chemsys: str | None) -> str | None:
    """Normalize a dash-separated chemsys, e.g. ``O-Li-Fe`` -> ``Fe-Li-O``."""
    if chemsys is None:
        return None
    cleaned = chemsys.strip()
    if not cleaned:
        return None
    parts = [part.strip() for part in cleaned.split("-")]
    if any(not part for part in parts):
        raise ValueError("Invalid chemsys: empty element part")
    return "-".join(normalize_element_list(parts))
