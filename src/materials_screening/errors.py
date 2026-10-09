"""Project exception hierarchy (M2: request validation)."""


class MaterialsScreeningError(Exception):
    """Base exception for all project errors."""


class InvalidRequestError(MaterialsScreeningError):
    """Raised when a screening request fails validation."""


class ConfigurationError(MaterialsScreeningError):
    """Raised when the environment or configuration is invalid."""


class RepositoryError(MaterialsScreeningError):
    """Raised when a repository query fails."""


class RepositoryAuthenticationError(RepositoryError):
    """Raised when the Materials Project API rejects credentials."""


class RepositoryRateLimitError(RepositoryError):
    """Raised when the Materials Project API rate limit is exceeded."""


class RepositoryTimeoutError(RepositoryError):
    """Raised when a Materials Project request times out."""


class RepositoryMappingError(RepositoryError):
    """Raised when a repository document cannot be mapped to a MaterialRecord."""


class ExportError(MaterialsScreeningError):
    """Raised when exporting screening results fails."""


class ExportConflictError(ExportError):
    """Raised when an export would overwrite different existing content."""


class ValidationFailedError(MaterialsScreeningError):
    """Raised when final validation of a screening result fails."""
