"""Planner-layer errors (D2-M3)."""


class PlannerError(Exception):
    """Base exception for the planner layer."""


class PlannerQueryError(PlannerError, ValueError):
    """Raised when a planner query is empty or exceeds the length limit."""
