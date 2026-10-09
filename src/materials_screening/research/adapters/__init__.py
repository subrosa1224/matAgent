"""Read-only normalization adapters for existing domain agents."""

from materials_screening.research.adapters.data_analysis import DataAnalysisReadAdapter
from materials_screening.research.adapters.literature import (
    LiteratureEvidenceReadAdapter,
)
from materials_screening.research.adapters.materials_database import (
    MaterialsDatabaseCandidateSource,
    MaterialsDatabaseReadAdapter,
)

__all__ = [
    "DataAnalysisReadAdapter",
    "LiteratureEvidenceReadAdapter",
    "MaterialsDatabaseCandidateSource",
    "MaterialsDatabaseReadAdapter",
]
