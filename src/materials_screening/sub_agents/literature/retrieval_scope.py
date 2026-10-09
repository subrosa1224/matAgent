"""Conservative metadata scope: alternatives are OR, composite components are AND."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from pymatgen.core import Composition

from .models import LiteratureSearchInput
from .topic_scope import explicit_candidate_formulas, scientific_topic

_FORMULA = r"(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*){2,}"
_PHASES = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "g": "graphitic",
}
_PHASE = "|".join((*_PHASES, *_PHASES.values()))
_COMPONENT = rf"(?:(?:g|{_PHASE})[- ]?)?{_FORMULA}"
_CONDITIONS = (
    ("spray drying", r"喷雾干燥|\bspray[- ](?:drying|dried|dry)\b"),
    ("annealing", r"退火|\banneal\w*\b"),
    ("visible light", r"可见光|\bvisible[- ]light\b"),
    ("tetracycline", r"四环素|\btetracycline\b"),
    (
        "transparent conducting",
        r"透明(?:导电|电极)|\btransparent[- ](?:conduct(?:ing|ive|ors?)|electrodes?)\b"
        r"|(?:transmittance|transparency|transmission)[\s\S]*?"
        r"(?:resistivity|conductivity|sheet resistance|carrier mobility)"
        r"|(?:resistivity|conductivity|sheet resistance|carrier mobility)[\s\S]*?"
        r"(?:transmittance|transparency|transmission)",
    ),
    ("thin film", r"薄膜|\b(?:thin[- ]films?|films?|coatings?)\b"),
)


def search_subject(request: LiteratureSearchInput) -> str:
    """Keep the literature clause, not MP filters or negative sample caveats."""
    source = scientific_topic(request.research_question or request.topic)
    if request.research_question:
        match = re.search(
            r"(?:检索|查找|搜索|搜集)([^。；;\n]*?(?:文献|论文|实验研究))", source
        )
        if match:
            return match.group(1)
    return source


def _formulas(source: str) -> tuple[str, ...]:
    source = source.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    result = []
    for match in re.finditer(rf"(?<![A-Za-z0-9]){_FORMULA}(?![A-Za-z0-9])", source):
        value = match.group()
        try:
            composition = Composition(value, strict=True)
        except ValueError:
            continue
        if len(composition.elements) >= 2 and value not in result:
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class RetrievalScope:
    # Each alternative is a conjunction of complete chemical components.
    material_alternatives: tuple[tuple[str, ...], ...] = ()
    phases: tuple[tuple[str, str], ...] = ()
    conditions: tuple[tuple[str, str], ...] = ()
    query_labels: tuple[str, ...] = ()

    @property
    def query_materials(self) -> tuple[str, ...]:
        if (
            self.phases
            and len(self.material_alternatives) == 1
            and len(self.material_alternatives[0]) == 1
        ):
            phase, formula = self.phases[0]
            return (f"{phase}-{formula}",)
        return self.query_labels or tuple(
            "/".join(group) for group in self.material_alternatives
        )


@lru_cache(maxsize=256)
def retrieval_scope(request: LiteratureSearchInput) -> RetrievalScope:
    source = search_subject(request)
    candidates = explicit_candidate_formulas(request.topic)
    compounds = []
    labels = []
    for match in re.finditer(rf"{_COMPONENT}(?:\s*[/@]\s*{_COMPONENT})+", source):
        components = _formulas(match.group())
        if len(components) > 1 and components not in compounds:
            compounds.append(components)
            labels.append("/".join(re.findall(_COMPONENT, match.group())))
    formulas = _formulas(source)
    alternative = bool(re.search(r"\bor\b|或|候选化学式", source, re.I))
    if not compounds and re.search(
        r"\bcomposit\w*\b|\bheterojunction\b|复合|异质结", source, re.I
    ):
        for match in re.finditer(
            rf"{_COMPONENT}(?:\s*(?:\band\b|与|和)\s*{_COMPONENT})+", source
        ):
            components = _formulas(match.group())
            if len(components) > 1 and components not in compounds:
                compounds.append(components)
                labels.append("/".join(re.findall(_COMPONENT, match.group())))
    if len(compounds) == 1 or (compounds and alternative):
        groups = tuple(compounds)
    elif not compounds and candidates:
        groups = tuple((formula,) for formula in candidates)
    elif not compounds and (len(formulas) == 1 or alternative):
        groups = tuple((formula,) for formula in formulas)
    else:
        # Do not invent conjunction/disjunction for ambiguous lists.
        groups = ()
    phases = []
    for match in re.finditer(
        rf"(?P<phase>{_PHASE})[- ]?(?P<formula>{_FORMULA})", source
    ):
        value = (match["phase"], match["formula"])
        if _formulas(match["formula"]) and value not in phases:
            phases.append(value)
    conditions = tuple(
        (name, pattern)
        for name, pattern in _CONDITIONS
        if re.search(pattern, source, re.I)
    )
    return RetrievalScope(
        groups, tuple(phases), conditions, tuple(labels) if groups and compounds else ()
    )


def phase_missing(title: str, abstract: str, phase: str, formula: str) -> bool:
    """Title-only wrong phases cannot be rescued by incidental abstract mentions."""
    greek = next((k for k, v in _PHASES.items() if v == phase), phase)
    english = _PHASES.get(greek, phase)
    translated_title = title.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    text = (
        translated_title
        + " "
        + abstract.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    )
    formula_names = re.escape(formula)
    if formula == "Fe2O3":
        formula_names += "|iron oxides?"
    if greek == "g" and formula == "C3N4":
        formula_names += "|carbon nitride"
    desired = rf"(?:{greek}|\b{english})[- ]?(?:{formula_names})"
    if not re.search(desired, text, re.I):
        return True
    other = tuple(p for p in (*_PHASES, *_PHASES.values()) if p not in {greek, english})
    wrong_title = re.search(
        rf"(?:{'|'.join(other)})[- ]?(?:{formula_names})",
        translated_title,
        re.I,
    )
    return bool(wrong_title and not re.search(desired, translated_title, re.I))
