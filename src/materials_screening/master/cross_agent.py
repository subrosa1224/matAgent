"""Typed MA-4 literature-to-database coordination boundary."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from materials_screening.services.material_database_service import (
    MaterialDatabaseService,
)
from materials_screening.sub_agents.literature.dossier import PaperDossierStore
from materials_screening.sub_agents.materials_database.models import (
    FilterOperator,
    PropertyFilter,
    SearchMaterialsInput,
    SortRule,
)

_PERIODIC_TABLE = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe "
    "Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In "
    "Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf "
    "Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm "
    "Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
)
_ELEMENTS = frozenset(_PERIODIC_TABLE.split())
_FORMULA_PATTERN = re.compile(r"(?<![A-Za-z])(?:[A-Z][a-z]?\d*){2,}(?![A-Za-z])")


class MaterialClue(BaseModel):
    """Evidence-bound material composition clue from one paper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    clue_id: str = Field(pattern=r"^clue-[0-9a-f]{24}$")
    document_id: str = Field(pattern=r"^doc-[0-9a-f]{24}$")
    paper_title: str = Field(min_length=1, max_length=1000)
    material_label: str = Field(min_length=1, max_length=1000)
    required_elements: tuple[str, ...] = Field(min_length=1, max_length=12)
    formula_candidates: tuple[str, ...] = Field(default=(), max_length=20)
    source_quote: str = Field(min_length=1, max_length=4000)
    evidence_page: int = Field(ge=1)
    evidence_chunk_id: str = Field(min_length=1)
    review_status: Literal["approved", "pending"]
    confidence: float = Field(ge=0, le=1)
    inference_notes: tuple[str, ...] = ()

    @field_validator("required_elements")
    @classmethod
    def validate_elements(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)) or any(
            item not in _ELEMENTS for item in values
        ):
            raise ValueError("required_elements must be unique element symbols")
        return values


class CrossAgentCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    clue_id: str
    query_id: str
    material_id: str
    formula_pretty: str
    elements: tuple[str, ...] = ()
    is_stable: bool | None = None
    energy_above_hull_ev_atom: float | None = None
    band_gap_ev: float | None = None


class CrossAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["completed", "partial", "failed"]
    clues: tuple[MaterialClue, ...] = ()
    candidates: tuple[CrossAgentCandidate, ...] = ()
    query_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class LiteratureMaterialClueExtractor:
    """Extract conservative clues from evidence-grounded paper dossiers."""

    def __init__(
        self,
        approved_root: Path = Path("data/literature_dossiers"),
        pending_root: Path = Path("data/literature_dossier_candidates"),
    ) -> None:
        self._approved = PaperDossierStore(approved_root)
        self._pending = PaperDossierStore(pending_root)

    def extract(self, document_id: str) -> MaterialClue | None:
        dossier = self._approved.load(document_id)
        review_status: Literal["approved", "pending"] = "approved"
        if dossier is None:
            dossier = self._pending.load(document_id)
            review_status = "pending"
        if dossier is None or dossier.review_status == "rejected":
            return None
        material_items = [
            item for item in dossier.items if item.category == "materials"
        ]
        if not material_items:
            return None
        supporting = [
            item
            for item in dossier.items
            if item.category == "materials"
            or (
                item.category == "preparation"
                and re.search(
                    r"molar composition|composition\s*:|mol\s*%",
                    item.source_quote,
                    re.IGNORECASE,
                )
            )
        ]
        formulas = tuple(
            dict.fromkeys(
                formula
                for item in supporting
                for formula in _valid_formulas(item.source_quote + " " + item.summary)
            )
        )
        elements = tuple(
            sorted(
                {
                    element
                    for formula in formulas
                    for element in _formula_elements(formula)
                }
            )
        )
        notes: list[str] = []
        label_text = " ".join(item.summary for item in material_items)
        if "TCP" in label_text.upper():
            elements = tuple(sorted(set(elements) | {"Ca", "P", "O"}))
            notes.append("TCP alias was expanded to the Ca-P-O element system.")
        if re.search(
            r"(?:bioactive glass|\bBG(?![A-Za-z]))", label_text, re.IGNORECASE
        ):
            elements = tuple(sorted(set(elements) | {"Si", "Ca", "P", "O"}))
            notes.append("Bioactive-glass label was expanded to the Si-Ca-P-O system.")
        if not elements:
            return None
        evidence = material_items[0]
        identity = f"{document_id}|{'-'.join(elements)}|{evidence.chunk_id}"
        return MaterialClue(
            clue_id="clue-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            document_id=document_id,
            paper_title=dossier.title,
            material_label=evidence.summary[:1000],
            required_elements=elements,
            formula_candidates=formulas,
            source_quote=evidence.source_quote[:4000],
            evidence_page=evidence.page,
            evidence_chunk_id=evidence.chunk_id,
            review_status=review_status,
            confidence=evidence.confidence,
            inference_notes=tuple(notes),
        )


class LiteratureDatabaseCoordinator:
    """Convert typed literature clues into typed database searches."""

    def __init__(
        self,
        database_service: MaterialDatabaseService,
        extractor: LiteratureMaterialClueExtractor | None = None,
    ) -> None:
        self._database = database_service
        self._extractor = extractor or LiteratureMaterialClueExtractor()

    def run(
        self, document_ids: list[str], *, limit_per_clue: int = 10
    ) -> CrossAgentResult:
        clues: list[MaterialClue] = []
        candidates: list[CrossAgentCandidate] = []
        query_ids: list[str] = []
        warnings: list[str] = []
        for document_id in dict.fromkeys(document_ids):
            clue = self._extractor.extract(document_id)
            if clue is None:
                warnings.append(f"{document_id}: no validated material clue")
                continue
            clues.append(clue)
            if clue.review_status != "approved":
                warnings.append(f"{document_id}: clue comes from a pending dossier")
            try:
                result = self._database.search(_query_from_clue(clue, limit_per_clue))
            except Exception as exc:
                warnings.append(
                    f"{document_id}: database query failed ({type(exc).__name__})"
                )
                continue
            query_id = str(result["query_id"])
            query_ids.append(query_id)
            for row in result.get("materials", []):
                candidates.append(_candidate(clue.clue_id, query_id, row))
        status: Literal["completed", "partial", "failed"]
        if clues and candidates and not warnings:
            status = "completed"
        elif clues or candidates:
            status = "partial"
        else:
            status = "failed"
        return CrossAgentResult(
            status=status,
            clues=tuple(clues),
            candidates=tuple(candidates),
            query_ids=tuple(query_ids),
            warnings=tuple(warnings),
        )


def _query_from_clue(clue: MaterialClue, limit: int) -> SearchMaterialsInput:
    """Typed conversion; formula candidates are intentionally not exact filters."""

    return SearchMaterialsInput(
        required_elements=clue.required_elements,
        filters=(
            PropertyFilter(field="is_stable", operator=FilterOperator.EQ, value=True),
        ),
        sort=(SortRule(field="energy_above_hull_ev_atom", direction="asc"),),
        limit=limit,
        fields=(
            "material_id",
            "formula_pretty",
            "elements",
            "is_stable",
            "energy_above_hull_ev_atom",
            "band_gap_ev",
        ),
    )


def _candidate(clue_id: str, query_id: str, row: dict[str, Any]) -> CrossAgentCandidate:
    return CrossAgentCandidate(
        clue_id=clue_id,
        query_id=query_id,
        material_id=str(row.get("material_id") or "unknown"),
        formula_pretty=str(row.get("formula_pretty") or "unknown"),
        elements=tuple(str(item) for item in row.get("elements") or ()),
        is_stable=row.get("is_stable"),
        energy_above_hull_ev_atom=row.get("energy_above_hull_ev_atom"),
        band_gap_ev=row.get("band_gap_ev"),
    )


def _valid_formulas(text: str) -> tuple[str, ...]:
    return tuple(
        match.group(0)
        for match in _FORMULA_PATTERN.finditer(text)
        if _formula_elements(match.group(0))
        and (
            any(char.isdigit() for char in match.group(0))
            or len(_formula_elements(match.group(0))) > 1
        )
    )


def _formula_elements(formula: str) -> tuple[str, ...]:
    tokens = tuple(re.findall(r"[A-Z][a-z]?", formula))
    return (
        tuple(dict.fromkeys(tokens))
        if tokens and all(item in _ELEMENTS for item in tokens)
        else ()
    )
