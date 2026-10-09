"""Conservative scientific-topic projection; original requests remain auditable."""

from __future__ import annotations

import re


def explicit_candidate_formulas(topic: str) -> tuple[str, ...]:
    """Read only the bounded, complete master-owned queue, not free text guesses."""
    match = re.search(
        r"候选化学式\s*[：:]\s*(.*?)[。\n]\s*查询快照\s*[：:]\s*(query-[\w-]+)",
        topic,
        re.DOTALL,
    )
    if match is None:
        return ()
    values = tuple(value.strip() for value in re.split(r"[；;,、]", match[1]))
    if not 1 <= len(values) <= 30 or not all(
        re.fullmatch(r"(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*){2,}", value)
        for value in values
    ):
        return ()
    from pymatgen.core import Composition

    try:
        if any(len(Composition(value, strict=True).elements) < 2 for value in values):
            return ()
    except ValueError:
        return ()
    return tuple(dict.fromkeys(values))


def candidate_handoff_projection(message: str):
    """Preserve the original question and all explicitly handed-off candidates."""
    candidates = explicit_candidate_formulas(message)
    match = re.search(
        r"原始科研问题\s*[：:]\s*(.*?)\s*候选化学式\s*[：:]", message, re.DOTALL
    )
    query = re.search(r"查询快照\s*[：:]\s*(query-[\w-]+)", message)
    if not candidates or match is None or query is None:
        return None
    question = match[1].strip()
    if not 0 < len(question) <= 4000:
        return None
    clause = re.search(
        r"(?:检索|查找|搜索|搜集)([^。；;\n]*?(?:文献|论文|实验研究))", question
    )
    subject = clause[1] if clause else question
    topic = (
        f"原始科研问题：{subject}。候选化学式：{'；'.join(candidates)}。"
        f"查询快照：{query[1]}。"
    )
    if len(topic) > 1000:
        return None
    return question, candidates, topic


_ION_PATTERNS = {
    "lithium-ion battery": r"\b(?:lithium|li)[ -]?ion\b|锂离子电池",
    "sodium-ion battery": r"\b(?:sodium|na)[ -]?ion\b|钠离子电池",
    "lithium-sulfur battery": r"\b(?:lithium|li)[ -](?:sulfur|sulphur|s)\b|锂硫电池",
    "lithium-oxygen battery": (
        r"\b(?:lithium|li)[ -](?:oxygen|air|o2)\b|锂(?:氧|空气)电池"
    ),
}
_ROLE_PATTERNS = {
    "cathode": r"\bcathodes?\b|\bpositive electrodes?\b|正极",
    "anode": r"\banodes?\b|\bnegative electrodes?\b|负极",
}


def scientific_topic(topic: str) -> str:
    """Strip only known handoff wrappers and negative history reuse commands.

    Do not strip scientific negation, numerical constraints, or evidence caveats.
    """
    match = re.search(
        r"原始科研问题\s*[：:]\s*(.*?)\s*候选化学式\s*[：:]\s*"
        r"(.*?)[。\n]\s*查询快照\s*[：:]",
        topic,
        re.DOTALL,
    )
    if match is not None:
        topic = " ".join((match.group(1).strip(), match.group(2).strip()))
    return re.sub(
        r"(?:不得|不要|不能|不)(?:再)?(?:复用|沿用|使用)\s*"
        r"(?:前面|之前|此前|上次|先前)的?[^，,；;。\n]*"
        r"(?:报告|检索结果|分析结果|流程)",
        "",
        topic,
    )


def battery_mentions(text: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    folded = text.casefold().replace("–", "-").replace("‐", "-")
    ions = tuple(k for k, p in _ION_PATTERNS.items() if re.search(p, folded))
    roles = tuple(k for k, p in _ROLE_PATTERNS.items() if re.search(p, folded))
    return ions, roles


def battery_query_terms(topic: str) -> tuple[str, ...]:
    ions, roles = battery_mentions(scientific_topic(topic))
    # A role alone can refer to non-battery electrochemistry; don't impose ions.
    if not ions and not re.search(r"\bbatter(?:y|ies)\b|电池", topic.casefold()):
        return ()
    return (*ions, *roles)


def battery_evidence_scope(
    topic: str, title: str, abstract: str
) -> tuple[bool, tuple[str, ...]]:
    """Return explicit mismatch and uncertainty; never infer measured performance.

    A wrong-only title cannot be rescued by incidental abstract comparisons.
    Mixed roles/ions require full text instead of automatic core promotion.
    """
    terms = battery_query_terms(topic)
    wanted_ions = tuple(x for x in terms if x in _ION_PATTERNS)
    wanted_roles = tuple(x for x in terms if x in _ROLE_PATTERNS)
    title_ions, title_roles = battery_mentions(title)
    ions, roles = battery_mentions(title + " " + abstract)
    missing: list[str] = []
    for wanted, title_found, found, label in (
        (wanted_ions, title_ions, ions, "电池类型"),
        (wanted_roles, title_roles, roles, "电极角色"),
    ):
        if not wanted:
            continue
        if title_found and not set(wanted).intersection(title_found):
            return True, (label + "不符",)
        if found and not set(wanted).intersection(found):
            return True, (label + "不符",)
        if not found or set(found) != set(wanted):
            missing.append(label + "待全文核验")
    return False, tuple(missing)
