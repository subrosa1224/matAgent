"""Stable rule codes for the planner resolver (D2-M2)."""

from enum import StrEnum


class StabilityRule(StrEnum):
    """Codes for rules around the word ``stable``."""

    STABLE_AS_IS_STABLE = "stable_as_is_stable"
    STABLE_NOT_HULL_ZERO = "stable_not_hull_zero"


class AmbiguityCode(StrEnum):
    """Stable codes for ambiguous requirements."""

    UNIT_UNSPECIFIED_ASSUMED = "unit_unspecified_assumed"
    UNIT_UNSPECIFIED_IMPLAUSIBLE = "unit_unspecified_implausible"
    APPLICATION_GOAL = "application_goal"
    TOXIC_OR_RARE_ELEMENTS = "toxic_or_rare_elements"
    STABLE_NOT_HULL_ZERO = "stable_not_hull_zero"
    LLM_FLAGGED = "llm_flagged_ambiguity"


class ConflictCode(StrEnum):
    """Stable codes for contradictory requirements."""

    REQUIRED_AND_EXCLUDED_ELEMENT = "required_and_excluded_element"
    BAND_GAP_MIN_ABOVE_MAX = "band_gap_min_above_max"
    HULL_MIN_ABOVE_MAX = "hull_min_above_max"
    DENSITY_MIN_ABOVE_MAX = "density_min_above_max"
    TARGET_OUTSIDE_BAND_GAP = "target_outside_band_gap"
    LLM_FLAGGED = "llm_flagged_conflict"


class UnsupportedCode(StrEnum):
    """Stable codes for unsupported tasks."""

    PREDICT_MATERIALS = "predict_materials"
    PREDICT_PROPERTIES = "predict_properties"
    DFT_COMPUTATION = "dft_computation"
    SYNTHESIS_ADVICE = "synthesis_advice"
    EXPERIMENTAL_DATA = "experimental_data"
    LLM_FLAGGED = "llm_flagged_unsupported"


class InvalidCode(StrEnum):
    """Stable codes for invalid extractions."""

    EMPTY_QUERY = "empty_query"
    INVALID_ELEMENT = "invalid_element"
    INVALID_CHEMSYS = "invalid_chemsys"
    NEGATIVE_BAND_GAP = "negative_band_gap"
    NEGATIVE_HULL = "negative_hull"
    NON_POSITIVE_DENSITY = "non_positive_density"
    NEGATIVE_TARGET_BAND_GAP = "negative_target_band_gap"
    INVALID_SPACEGROUP = "invalid_spacegroup"
    INVALID_CRYSTAL_SYSTEM = "invalid_crystal_system"
    INVALID_LIMIT = "invalid_limit"
    CONFLICTING_REQUIREMENTS = "conflicting_requirements"
