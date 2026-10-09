"""Stable errors for research-project contracts and persistence."""

from __future__ import annotations


class ResearchProjectError(ValueError):
    """A deterministic, user-correctable research project error."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class ProjectConflictError(ResearchProjectError):
    """The requested write conflicts with persisted project state."""


class ProjectTransitionError(ResearchProjectError):
    """The requested project state transition is not allowed."""
