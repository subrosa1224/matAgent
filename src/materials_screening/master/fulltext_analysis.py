"""Current-snapshot scope -> actual DataAnalysisAgent -> partial source report."""

import hashlib
import json
import re
from contextlib import nullcontext
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import ValidationError

from materials_screening.agent.models import AgentResult
from materials_screening.llm.errors import LLMStructuredOutputError
from materials_screening.sub_agents.data_analysis.descriptive_report import (
    render_descriptive_report,
)
from materials_screening.sub_agents.literature.matrix import unit_present_in_evidence

from .fulltext_analysis_contracts import AnalysisScopePlan, FulltextAnalysisRecord
from .fulltext_condition_locator import (
    LOCATOR_POLICY,
    validate_located_plan,
    validate_locations,
)
from .fulltext_conditions import _PROMPT as _CONDITION_PROMPT
from .fulltext_conditions import ConditionPlan, validate_condition_plan
from .fulltext_preview import _Pause
from .fulltext_scope_stages import (
    PRIMARY_STAGE_PROMPT,
    SUPPLEMENT_STAGE_PROMPT,
    SupplementaryScopePlan,
    merge_scopes,
    primary_candidates,
    supplementary_candidates,
)
from .fulltext_snapshots import source_identities
from .fulltext_targeted_evidence import TARGETED_POLICY, validate_targeted_evidence
from .fulltext_tasks import FulltextTask

_SCOPE_PROMPT = """Select the requested experimental metrics and samples from ONLY
the supplied source-checked candidate rows. The original request and chronological
followups are requirements, not paper facts. Keep Materials Project computed
properties outside this literature scope. Include explicitly requested controls;
exclude other material series. Allow useful source-backed supplementary metrics.
For each
required metric copy task_quote verbatim from the user requirements; use only
measurement_ids present in the candidates. An empty measurement_ids means missing,
not that the paper contains no measurements. Do not invent values, conditions or
units. State requested operations (describe, compare, trend) and the required
condition keys/independent variable; never infer missing experimental conditions.
The scope is a model proposal, not expert approval. Return compact JSON."""

_SCOPE_PROMPT += """
Use experimental_requirements to choose literature metrics, not the database-only
clauses. Match ordinary bilingual metric names (gain/增益, responsivity/响应度,
degradation/降解率). Do not leave a clearly requested matching candidate unaccounted
for: select its measurement_id or put an explicit reason in excluded_measurements.
experimental_requirements is a list of original contiguous requirement clauses.
For each task_quote copy ONE relevant clause or a continuous substring of it;
never join multiple clauses, copy the whole list, or paraphrase the task quote.
Never force inclusion if its sample or evidence is inappropriate. Missing conditions
are unknown, not evidence of identical conditions or permission for comparisons.
requested_operations must be a JSON array containing only the exact English tokens
"describe", "compare", "trend" (one to three items). Do not translate these tokens
or use synonyms, explanations, objects or metric names as operations.
Examples: "requested_operations": ["describe"] for descriptive statistics;
"requested_operations": ["describe", "compare"] for a requested comparison;
"requested_operations": ["describe", "trend"] for a requested trend.
Choosing an operation does not override evidence or condition checks.
For role="requested" (the default), the metric must be explicitly named in its
task_quote (ordinary bilingual aliases are allowed). For useful ancillary metrics,
set role="supplementary", give a nonempty supplementary_reason explaining relevance
to the quoted research task, and leave requested_metric null. Use actual candidate
IDs with matching original metric names. Supplementary entries require source rows;
they cannot replace requested outcomes or count as requested coverage. Do not label
a clearly requested matching outcome as supplementary. Do not use supplementary
entries to authorize cross-condition ranking or trend fitting.
Each selected row must have the same metric as that scope entry; do not relabel
bandgap, wavelengths or response times as responsivity. Keep illumination and
bias as conditions, not substitute outcome metrics. Irrelevant rows remain in
the extraction snapshot; omit them from the analysis scope.
For a verbose source metric, preserve its name and set requested_metric to the
explicit user-requested metric (e.g. 降解率). This is a proposed correspondence,
not permission to merge different physical quantities. Degradation efficiency
reported as percent may correspond to 降解率; TOC/mineralization and kinetic rate
constants do not. Responsivity peak wavelength in nm is not responsivity in A/W.
Select source IDs; never change the source values, units or condition fields.
Example: name="Photocatalytic degradation efficiency of TC under visible light
for 3 h", requested_metric="降解率", task_quote copied from the user's request.
Do not copy the long source name into requested_metric when that name is absent
from the user request. Apply proposed_requested_metrics from repair feedback by
choosing a supported request phrase, not by deleting a requested valid candidate.
"""

_SCOPE_REPAIR_PROMPT = """
This is a correction request. Return a complete replacement AnalysisScopePlan,
not a patch, critique, confirmation, or copy of the request envelope.
Do not return the input or feedback. requirements, candidate_metadata,
previous_proposal, validation_issue, validation_fields, metric_scope_feedback,
repair_instruction and output_contract are INPUT ONLY and must not be output keys.
Apply the feedback while obeying the original metric and evidence constraints.
Return ONE JSON object containing metrics and requested_operations.
Each metrics entry contains name, task_quote and measurement_ids; use supplied
IDs and exact user quotes, never placeholders. Optional top-level fields are
required_conditions, independent_variable, limitations, excluded_measurements.
requested_operations contains only describe, compare or trend.
No additional top-level fields, Markdown, prose, wrappers or nested plan object.
An individual metric may include requested_metric for a checked correspondence
to an explicitly requested metric; retain the original name and source IDs.
For ancillary evidence use role="supplementary", supplementary_reason and no
requested_metric. Requested metrics retain role="requested".
"""


def _scope_validation_fields(error):
    """Expose bounded schema locations, never provider values/messages/context."""
    cause = error.__cause__
    if not isinstance(cause, ValidationError):
        return []
    fields = set(AnalysisScopePlan.model_fields) | {
        "name",
        "requested_metric",
        "role",
        "supplementary_reason",
        "task_quote",
        "measurement_ids",
    }
    codes = {
        "literal_error",
        "too_short",
        "too_long",
        "missing",
        "extra_forbidden",
        "list_type",
        "tuple_type",
        "string_type",
        "dict_type",
    }
    feedback = []
    for row in cause.errors(
        include_input=False, include_context=False, include_url=False
    )[:8]:
        location = [
            part
            if (type(part) is int and 0 <= part <= 1000)
            or (isinstance(part, str) and part in fields)
            else "other"
            for part in row["loc"][:8]
        ]
        item = {
            "location": location,
            "error_code": row["type"] if row["type"] in codes else "invalid",
        }
        if location and location[0] == "requested_operations":
            item["allowed_values"] = ["describe", "compare", "trend"]
        feedback.append(item)
    return feedback


def _computed_only(text):
    return bool(
        re.search(r"Materials\s*Project|\bMP\b|数据库|计算", text, re.I)
    ) and not re.search(r"论文|文献|全文|实验|器件|提取", text)


def _experimental_requirements(requirements):
    return [
        part.strip()
        for part in re.findall(r"[^；;。\n]+[；;。]?", requirements)
        if part.strip() and not _computed_only(part)
    ]


def _explicit_report_conditions(requirements):
    """Bounded literal request recognition for display only, not evidence binding."""
    clauses = [
        part
        for part in _experimental_requirements(requirements)
        if re.search(r"提取|核对|分析|条件|extract|check|condition", part, re.I)
        and not re.search(r"不要|不用|暂不|无需|不需要|do not|don't", part, re.I)
    ]
    text = "\n".join(clauses)
    definitions = (
        (
            "光照",
            r"光照|光源|\billumination\b|\blighting\b|\blight source\b",
            ("illumination", "light_source", "illumination_conditions", "光照", "光源"),
        ),
        (
            "偏压",
            r"偏压|偏置电压|\bbias(?: voltage)?\b",
            ("bias_voltage", "bias", "偏压", "偏置电压"),
        ),
    )
    return [
        (label, keys)
        for label, pattern, keys in definitions
        if re.search(pattern, text, re.I)
    ]


def _expected_candidate_ids(candidates, requirements):
    experiment = "\n".join(_experimental_requirements(requirements)).casefold()
    aliases = (
        ("gain", "photoconductive gain", "增益", "光电导增益"),
        ("responsivity", "responsitivity", "响应度"),
        ("degradation", "degradation rate", "降解率"),
    )
    expected = set()
    for names in aliases:
        if any(name in experiment for name in names):
            expected.update(
                mid
                for mid, row in candidates.items()
                if str(row.get("metric", "")).strip().casefold() in names
                or (
                    "degradation" in names
                    and _mapped_metric_matches(
                        str(row.get("metric", "")), row.get("unit"), "降解率"
                    )
                )
            )
    return expected


_METRIC_ALIASES = (
    ("gain", "photoconductive gain", "增益", "光电导增益"),
    ("responsivity", "responsitivity", "响应度"),
    ("bandgap", "band gap", "带隙", "禁带宽度"),
    ("density", "密度"),
    ("degradation", "degradation rate", "降解率"),
    ("reported strength", "strength", "强度"),
)


def _metric_names(name):
    normalized = " ".join(name.strip().casefold().split())
    return next(
        (names for names in _METRIC_ALIASES if normalized in names), (normalized,)
    )


def _metric_named_in_quote(name, quote):
    for alias in _metric_names(name):
        if alias and re.search(
            r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])",
            quote.casefold(),
        ):
            return True
    return False


def _mapped_metric_matches(name, unit, requested):
    """Bounded name/unit check; no arbitrary model synonym claims or conversions."""
    normalized = " ".join(name.strip().casefold().split())
    target = set(_metric_names(requested))
    source = set(_metric_names(name))
    if "degradation" in target:
        head = bool(source.intersection(target)) or bool(
            re.fullmatch(
                r"(?:photocatalytic\s+)?degradation\s+(?:efficiency|percentage|percent|rate)"
                r"(?:\s+(?:of|under|after|for)\b.*)?",
                normalized,
            )
        )
        different_quantity = re.search(
            r"\btoc\b|mineralization|rate constant|kinetic", normalized
        )
        return (
            head
            and not different_quantity
            and str(unit).strip().casefold() in {"%", "percent"}
        )
    if not source.intersection(target):
        return False
    if "responsivity" in target:
        normalized_unit = re.sub(r"\s+", "", str(unit)).replace("−", "-")
        return normalized_unit in {"A/W", "mA/W", "AW-1", "mAW-1"}
    if "gain" in target:
        return unit in (None, "", "1", "dimensionless")
    return True


def _scope_row_matches(metric, row):
    if metric.requested_metric is None:
        return bool(
            set(_metric_names(metric.name)).intersection(_metric_names(row["metric"]))
        )
    return _mapped_metric_matches(
        metric.name, row.get("unit"), metric.requested_metric
    ) and _mapped_metric_matches(
        row["metric"], row.get("unit"), metric.requested_metric
    )


def _scope_metric_feedback(scope, candidates, requirements):
    """Explain every metric failure without silently rewriting the proposal."""
    if scope is None:
        return {}
    issues, supported = [], []
    for metric in scope.metrics:
        if metric.task_quote not in requirements:
            # Do not echo fabricated provider quotes as user requirements.
            issues.append(
                {
                    "metric_name": metric.name,
                    "reason": "quote_not_in_task",
                    "action": "use_verbatim_task_quote",
                }
            )
        elif metric.role == "supplementary" and not _computed_only(metric.task_quote):
            try:
                _validate_supplement(metric, candidates, requirements)
            except ValueError:
                issues.append(
                    {
                        "metric_name": metric.name,
                        "reason": "invalid_supplementary_evidence",
                        "action": "provide_source_rows_relevance_no_request_mapping",
                    }
                )
            else:
                supported.append(metric.name)
        elif _computed_only(metric.task_quote) or not _metric_named_in_quote(
            metric.requested_metric or metric.name, metric.task_quote
        ):
            item = {
                "metric_name": metric.name,
                "task_quote": metric.task_quote,
                "reason": "quote_does_not_request_this_literature_metric",
                "action": "remove_scope_entry",
            }
            rows = [
                candidates[mid] for mid in metric.measurement_ids if mid in candidates
            ]
            options = [
                alias
                for names in _METRIC_ALIASES
                for alias in names
                if re.search(
                    r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])",
                    metric.task_quote.casefold(),
                )
                and rows
                and len(rows) == len(metric.measurement_ids)
                and all(
                    _mapped_metric_matches(metric.name, row.get("unit"), alias)
                    and _mapped_metric_matches(row["metric"], row.get("unit"), alias)
                    for row in rows
                )
            ]
            if options and not _computed_only(metric.task_quote):
                item.update(
                    action="set_requested_metric",
                    proposed_requested_metrics=options,
                    reason="source_name_has_checked_request_correspondence",
                )
            elif (
                not _computed_only(metric.task_quote)
                and rows
                and all(_scope_row_matches(metric, row) for row in rows)
                and metric.requested_metric is None
            ):
                item.update(
                    action="mark_supplementary_if_relevant",
                    reason="source_backed_metric_not_explicitly_requested",
                    required_fields=["role=supplementary", "supplementary_reason"],
                )
            issues.append(item)
        else:
            mismatches = [
                mid
                for mid in metric.measurement_ids
                if mid in candidates and not _scope_row_matches(metric, candidates[mid])
            ]
            if mismatches:
                issues.append(
                    {
                        "metric_name": metric.name,
                        "task_quote": metric.task_quote,
                        "reason": "candidate_metric_mismatch",
                        "measurement_ids": mismatches,
                        "action": "select_matching_rows_only",
                    }
                )
            else:
                supported.append(metric.name)
    return {"issues": issues, "supported_proposal_metrics": supported}


def _validate_supplement(metric, candidates, requirements):
    if _metric_named_in_quote(metric.name, metric.task_quote):
        raise ValueError("Explicitly requested metric cannot be supplementary")
    if (
        metric.requested_metric is not None
        or not (metric.supplementary_reason or "").strip()
        or not metric.measurement_ids
    ):
        raise ValueError(
            "Supplement requires source rows, relevance and no request mapping"
        )
    if set(metric.measurement_ids).intersection(
        _expected_candidate_ids(candidates, requirements)
    ):
        raise ValueError("Explicitly requested candidate omitted from primary scope")
    if any(
        mid not in candidates or not _scope_row_matches(metric, candidates[mid])
        for mid in metric.measurement_ids
    ):
        raise ValueError("Supplementary candidate metric does not match scope metric")


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _cell(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def _measurement_value_cell(row):
    """Preserve source typography and append only a missing declared unit."""
    value, unit = row["value_text"], row["unit"]
    if unit and not unit_present_in_evidence(unit, value):
        value = f"{value} {unit}"
    return _cell(value)


def _metadata(value):
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, dict):
        raise ValueError("Invalid candidate condition or variable metadata")
    return parsed


def _supplementary_attributes(plan, measurement_id, annotation_format):
    """Bounded row references; full immutable sources stay in the analysis plan."""
    rows = []
    for binding in plan.bindings:
        if measurement_id not in binding.measurement_ids:
            continue
        item = binding.model_dump(mode="json")
        if annotation_format == "refs-v2":
            item = {
                **binding.model_dump(
                    include={"key", "kind", "value_text", "numeric_value", "unit"}
                ),
                "binding_sha256": _digest(item),
                "value_source_sha256": _digest(
                    binding.value_source.model_dump(mode="json")
                ),
                "applicability_source_sha256": _digest(
                    binding.applicability_source.model_dump(mode="json")
                ),
            }
        elif annotation_format != "inline-v1":
            raise ValueError("Unknown condition annotation format")
        rows.append(item)
    return json.dumps(rows, ensure_ascii=False, sort_keys=True)


def _sample_scope(candidates, requirements):
    """Conservative literal-name gate, not a chemical alias or intent resolver.

    Only strip an explicit percent loading prefix. A constituent mentioned inside
    a slash composite is not an independently requested pure-material control.
    General questions without any named candidate family remain model proposals.
    Never silently broaden a literal named family to other paper material series.
    """
    translate = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    requirements = requirements.translate(translate)
    families = {
        mid: re.sub(
            r"^\s*\d+(?:\.\d+)?\s*(?:wt\s*)?%\s+",
            "",
            row["material"].translate(translate),
        ).strip()
        for mid, row in candidates.items()
    }
    named = {
        family
        for family in families.values()
        if family
        and re.search(
            r"(?<![A-Za-z0-9/\-])" + re.escape(family) + r"(?![A-Za-z0-9/\-])",
            requirements,
        )
    }
    if not named:
        return candidates, {}
    exclusions = {
        mid: "样品材料系列未在任务中明确命名；字面范围门槛不推断别名或额外对照。"
        for mid, family in families.items()
        if family not in named
    }
    return {
        mid: row for mid, row in candidates.items() if mid not in exclusions
    }, exclusions


class FulltextAnalysisProcessor:
    def __init__(
        self,
        *,
        snapshots,
        analyses,
        dataset_factory,
        agent_factory,
        query_factory,
        model_profile,
        max_output_bytes=65536,
        condition_planner=None,
        condition_locator=None,
        condition_reader=None,
        targeted_planner=None,
        figure_review_service=None,
    ):
        self.snapshots, self.analyses = snapshots, analyses
        self.dataset_factory, self.agent_factory = dataset_factory, agent_factory
        self.query_factory, self.model_profile = query_factory, model_profile
        self.max_output_bytes = max_output_bytes
        # Deliberately opt-in until real provider condition extraction qualifies.
        self.condition_planner = condition_planner
        if (condition_locator is None) != (condition_reader is None):
            raise ValueError("Both condition stages must be configured")
        if condition_planner is not None and condition_locator is not None:
            raise ValueError("Do not mix legacy and two-stage condition policies")
        self.condition_locator, self.condition_reader = (
            condition_locator,
            condition_reader,
        )
        if targeted_planner is not None and condition_locator is None:
            raise ValueError("Targeted evidence requires two-stage shared conditions")
        self.targeted_planner = targeted_planner
        self.figure_review_service = figure_review_service

    def run(
        self,
        task,
        *,
        calls,
        save_task,
        chunks_by_document,
        refresh_sources,
        user_turn_id,
    ):
        record, tools = None, 0
        evidence_ids = set()

        def save(**updates):
            nonlocal task
            task = FulltextTask.model_validate(
                {
                    **task.model_dump(mode="json"),
                    **updates,
                    "updated_at": datetime.now(UTC),
                }
            )
            save_task(task)

        def checkpoint(updated):
            nonlocal record
            record = FulltextAnalysisRecord.model_validate(updated.model_dump())
            ref = self.analyses.save(record)
            save(fulltext_analysis_ref=ref)

        def revision(**updates):
            checkpoint(
                record.model_copy(
                    update={
                        "record_id": "fulltext-analysis-" + uuid4().hex,
                        "parent_record_id": record.record_id,
                        "created_at": datetime.now(UTC),
                        **updates,
                    }
                )
            )

        def finish(text, code=None, cancelled=False):
            warning = ()
            if len(text.encode()) > self.max_output_bytes:
                warning = ("显示截断；完整分区报告保存在本任务的不可变分析记录。",)
                prefix = warning[0] + "\n"
                text = prefix + text.encode()[
                    : max(0, self.max_output_bytes - len(prefix.encode()))
                ].decode("utf-8", "ignore")
                text = text.encode()[: self.max_output_bytes].decode("utf-8", "ignore")
            return AgentResult(
                conversation_id=task.conversation_id,
                user_turn_id=user_turn_id,
                status="cancelled" if cancelled else "error" if code else "completed",
                final_status=None if cancelled else "error" if code else "completed",
                response_text=text,
                model_call_count=calls.calls,
                tool_call_count=tools,
                selected_tools=("literature", "data_analysis")
                if tools
                else ("literature",),
                warnings=warning,
                evidence_ids=tuple(sorted(evidence_ids)),
                error={"code": code, "message": text, "retryable": True}
                if code
                else None,
            )

        try:
            if task.figure_review_policy != "disabled":
                if (
                    self.figure_review_service is None
                    or task.figure_evidence_ref is None
                ):
                    raise ValueError("Figure review must finish before analysis")
                self.figure_review_service.report_batch(task, task.figure_evidence_ref)
            save(stage="analyzing", resume_stage=None)
            selected = {
                doc: task.extraction_snapshots[doc]
                for doc, decision in task.preview_decisions.items()
                if decision.action == "extract"
            }
            policy = _digest(
                {
                    "scope_policy": "split-primary-supplementary-v2",
                    "version": "master-analysis-v5-targeted-evidence"
                    if self.targeted_planner is not None
                    else "master-analysis-v4-two-stage-conditions"
                    if self.condition_locator is not None
                    else "master-analysis-v3-condition-checkpoint"
                    if self.condition_planner is not None
                    else "master-analysis-v2-literal-sample-gate",
                    **(
                        {
                            "figure_evidence_ref": task.figure_evidence_ref.model_dump(
                                mode="json"
                            )
                        }
                        if task.figure_evidence_ref is not None
                        else {}
                    ),
                    **(
                        {"targeted_policy": TARGETED_POLICY}
                        if self.targeted_planner is not None
                        else {}
                    ),
                    **(
                        {"locator_policy": LOCATOR_POLICY}
                        if self.condition_locator is not None
                        else {}
                    ),
                    **(
                        {"condition_prompt_sha256": _digest(_CONDITION_PROMPT)}
                        if self.condition_planner is not None
                        else {}
                    ),
                    "question": task.original_question,
                    "instructions": task.user_instructions,
                    "database_queries": task.database_query_ids,
                    "database_datasets": task.database_dataset_ids,
                    "prior_analyses": task.analysis_ids,
                    "snapshots": {
                        key: ref.model_dump() for key, ref in selected.items()
                    },
                    "profile": self.model_profile,
                }
            )
            old = (
                self.analyses.load(task.fulltext_analysis_ref)
                if task.fulltext_analysis_ref
                else None
            )
            if old and (
                old.task_id != task.task_id
                or old.conversation_id != task.conversation_id
            ):
                raise ValueError("Analysis belongs to a different task")
            current = (
                old is not None
                and old.policy_sha256 == policy
                and old.selected_snapshots == selected
            )
            if current:
                record = old
            else:
                checkpoint(
                    FulltextAnalysisRecord(
                        task_id=task.task_id,
                        conversation_id=task.conversation_id,
                        policy_sha256=policy,
                        selected_snapshots=selected,
                        figure_evidence_ref=task.figure_evidence_ref,
                        parent_record_id=old.record_id if old else None,
                        annotation_format="refs-v2"
                        if self.condition_locator is not None
                        else "inline-v1",
                    )
                )
            # Safe representation-only revision BEFORE any dataset/statistics.
            # Preserve validated scope/locations/values and old immutable record.
            if (
                self.condition_locator is not None
                and record.annotation_format == "inline-v1"
                and record.status == "collecting"
                and not record.dataset_ids
                and not record.analysis_ids
            ):
                revision(annotation_format="refs-v2")
            query_rows = {}
            if task.database_query_ids:
                if self.query_factory is None:
                    raise ValueError("Task database snapshots unavailable")
                queries = self.query_factory()
                query_rows = {
                    query_id: queries.load_records(query_id)
                    for query_id in task.database_query_ids
                }
            query_digests = {
                query_id: _digest([row.model_dump(mode="json") for row in rows])
                for query_id, rows in query_rows.items()
            }
            if not record.query_digests and record.status == "collecting":
                if query_digests:
                    revision(query_digests=query_digests)
            elif record.query_digests != query_digests:
                raise ValueError("Task database query snapshot changed")
            datasets = self.dataset_factory()
            candidates, by_doc = {}, {}
            for doc, ref in selected.items():
                calls.check()
                snapshot = self.snapshots.load(ref)
                if snapshot.task_id != task.task_id or snapshot.status != "complete":
                    raise ValueError("Invalid task extraction snapshot")
                if (
                    source_identities(chunks_by_document[doc])
                    != snapshot.source_identities
                ):
                    raise ValueError("Analysis source changed")
                handoff = task.measurement_handoffs[doc]
                rows = (
                    json.loads(
                        datasets.load_dataframe(handoff.dataset_id).to_json(
                            orient="records", force_ascii=False
                        )
                    )
                    if handoff.dataset_id
                    else []
                )
                if len(rows) != handoff.record_count:
                    raise ValueError("Handoff record count changed")
                for row in rows:
                    if (
                        row["task_id"] != task.task_id
                        or row["snapshot_id"] != ref.snapshot_id
                        or row["document_id"] != doc
                        or row["review_status"] != "pending"
                        or row["analysis_mode"] != "trial"
                    ):
                        raise ValueError("Candidate identity or review policy mismatch")
                    if row["measurement_id"] in candidates:
                        raise ValueError("Ambiguous candidate identity")
                    candidates[row["measurement_id"]] = row
                by_doc[doc] = rows
            requirements = (
                task.original_question + "\n" + "\n".join(task.user_instructions)
            )
            candidates, exclusions = _sample_scope(candidates, requirements)
            if record.scope_exclusions != exclusions:
                if record.status == "complete":
                    raise ValueError("Saved report sample scope differs")
                revision(scope_exclusions=exclusions)
            primary_pool = primary_candidates(candidates, requirements)
            if record.scope is None and record.primary_scope is None:
                projection = [
                    {
                        key: row[key]
                        for key in (
                            "measurement_id",
                            "document_id",
                            "group_label",
                            "group_role",
                            "material",
                            "metric",
                            "unit",
                            "variables",
                            "conditions",
                        )
                    }
                    for row in primary_pool.values()
                ]
                text = json.dumps(
                    {
                        "requirements": requirements,
                        "experimental_requirements": _experimental_requirements(
                            requirements
                        ),
                        "candidate_metadata": projection,
                    },
                    ensure_ascii=False,
                )
                if len(text.encode()) > 65536:
                    raise ValueError("Scope input exceeds bounded window")
                for attempt in range(2):
                    scope = None
                    try:
                        scope = calls.generate_structured(
                            system_prompt=_SCOPE_PROMPT
                            + (_SCOPE_REPAIR_PROMPT if attempt else "")
                            + PRIMARY_STAGE_PROMPT,
                            user_text=text,
                            output_model=AnalysisScopePlan,
                            schema_name="fulltext_analysis_scope_v1",
                            max_output_tokens=4096,
                        ).parsed
                        if any(metric.role != "requested" for metric in scope.metrics):
                            raise ValueError(
                                "Primary stage cannot choose supplementary metrics"
                            )
                        self._validate_scope(scope, primary_pool, requirements)
                        break
                    except (ValueError, LLMStructuredOutputError) as scope_error:
                        if attempt == 1:
                            raise
                        text = json.dumps(
                            {
                                "requirements": requirements,
                                "experimental_requirements": _experimental_requirements(
                                    requirements
                                ),
                                "candidate_metadata": projection,
                                "previous_proposal": scope.model_dump(mode="json")
                                if scope is not None
                                else None,
                                "validation_issue": "Invalid structured scope output; "
                                "return only a JSON object matching AnalysisScopePlan."
                                if isinstance(scope_error, LLMStructuredOutputError)
                                else str(scope_error),
                                "validation_fields": _scope_validation_fields(
                                    scope_error
                                ),
                                "metric_scope_feedback": _scope_metric_feedback(
                                    scope, primary_pool, requirements
                                ),
                                "repair_instruction": (
                                    "Address EVERY metric_scope_feedback issue. "
                                    "Remove entries marked remove_scope_entry, rather "
                                    "than resubmitting them or attaching "
                                    "a general goal. "
                                    "Retain supported requested metrics; missing "
                                    "illumination/bias remain unknown conditions. "
                                    "Remove database-only and unrequested metrics. "
                                    "Each metric must be explicitly named in its quote "
                                    "and match its selected candidate rows. "
                                    "Account for every "
                                    "explicitly requested matching candidate by "
                                    "selection or an explicit exclusion reason. "
                                    "Use only supplied "
                                    "IDs and verbatim requirement quotes."
                                ),
                                "output_contract": {
                                    "return": "complete replacement AnalysisScopePlan",
                                    "required_fields": [
                                        "metrics",
                                        "requested_operations",
                                    ],
                                    "allowed_fields": list(
                                        AnalysisScopePlan.model_fields
                                    ),
                                    "allowed_operations": [
                                        "describe",
                                        "compare",
                                        "trend",
                                    ],
                                    "input_feedback_is_not_output": True,
                                },
                            },
                            ensure_ascii=False,
                        )
                        if len(text.encode()) > 65536:
                            raise ValueError(
                                "Scope repair input exceeds bounded window"
                            ) from None
                revision(primary_scope=scope)
            if record.scope is None:
                primary = record.primary_scope
                if any(metric.role != "requested" for metric in primary.metrics):
                    raise ValueError(
                        "Primary checkpoint contains supplementary metrics"
                    )
                self._validate_scope(primary, primary_pool, requirements)
                remaining = supplementary_candidates(candidates, primary_pool, primary)
                supplement = SupplementaryScopePlan()
                if remaining:
                    text = json.dumps(
                        {
                            "requirements": requirements,
                            "primary_scope": primary.model_dump(mode="json"),
                            "candidate_metadata": [
                                {
                                    key: row[key]
                                    for key in (
                                        "measurement_id",
                                        "document_id",
                                        "group_label",
                                        "material",
                                        "metric",
                                        "unit",
                                        "variables",
                                        "conditions",
                                    )
                                }
                                for row in remaining.values()
                            ],
                        },
                        ensure_ascii=False,
                    )
                    if len(text.encode()) > 65536:
                        raise ValueError(
                            "Supplementary scope input exceeds bounded window"
                        )
                    original_text = text
                    for attempt in range(2):
                        try:
                            supplement = calls.generate_structured(
                                system_prompt=SUPPLEMENT_STAGE_PROMPT,
                                user_text=text,
                                output_model=SupplementaryScopePlan,
                                schema_name="fulltext_supplementary_scope_v1",
                                max_output_tokens=4096,
                            ).parsed
                            combined = merge_scopes(primary, supplement)
                            self._validate_scope(combined, candidates, requirements)
                            if any(
                                mid not in remaining
                                for metric in supplement.metrics
                                for mid in metric.measurement_ids
                            ):
                                raise ValueError(
                                    "Supplementary selection outside remaining pool"
                                )
                            break
                        except (ValueError, LLMStructuredOutputError) as error:
                            if attempt:
                                raise
                            text = json.dumps(
                                {
                                    **json.loads(original_text),
                                    "correction": (
                                        "Return only a SupplementaryScopePlan. "
                                        "Use source names, supplementary role, "
                                        "relevance, verbatim quotes and remaining "
                                        "IDs. No request mapping."
                                    ),
                                    "validation_issue": "Invalid supplementary scope",
                                    "validation_fields": _scope_validation_fields(
                                        error
                                    ),
                                },
                                ensure_ascii=False,
                            )
                revision(scope=merge_scopes(primary, supplement))
            self._validate_scope(record.scope, candidates, requirements)
            labels = {
                mid: metric.name
                for metric in record.scope.metrics
                for mid in metric.measurement_ids
            }
            if not set(record.condition_plans).issubset(selected):
                raise ValueError("Condition checkpoint belongs to another document")
            if not set(record.condition_locations).issubset(selected):
                raise ValueError("Location checkpoint belongs to another document")
            if not set(record.targeted_plans).issubset(selected):
                raise ValueError("Targeted checkpoint belongs to another document")
            for doc, rows in by_doc.items():
                scoped = [
                    dict(row, scope_metric=labels[row["measurement_id"]])
                    for row in rows
                    if row["measurement_id"] in labels
                ]
                if not scoped:
                    # Never register an empty table or execute empty statistics.
                    continue
                if (
                    self.condition_planner is not None
                    or self.condition_locator is not None
                ):
                    targets = {row["measurement_id"]: row for row in scoped}
                    if doc not in record.condition_plans:
                        if record.status == "complete":
                            raise ValueError("Completed condition checkpoint missing")
                        calls.check()
                        if self.condition_locator is not None:
                            if doc not in record.condition_locations:
                                locations = self.condition_locator(
                                    calls=calls,
                                    candidates=targets,
                                    chunks=chunks_by_document[doc],
                                )
                                if locations is not None:
                                    locations = validate_locations(
                                        locations, targets, chunks_by_document[doc]
                                    )
                                    revision(
                                        condition_locations={
                                            **record.condition_locations,
                                            doc: locations,
                                        }
                                    )
                            locations = record.condition_locations.get(doc)
                            plan = (
                                self.condition_reader(
                                    calls=calls,
                                    locations=locations,
                                    candidates=targets,
                                    chunks=chunks_by_document[doc],
                                )
                                if locations is not None
                                else ConditionPlan(
                                    unresolved=(
                                        "没有明确的局部共同测试声明；不补推共享条件。",
                                    )
                                )
                            )
                        else:
                            plan = self.condition_planner(
                                calls=calls,
                                requirements=requirements,
                                candidates=targets,
                                chunks=chunks_by_document[doc],
                            )
                        plan = validate_condition_plan(
                            plan, targets, chunks_by_document[doc]
                        )
                        revision(condition_plans={**record.condition_plans, doc: plan})
                    plan = validate_condition_plan(
                        record.condition_plans[doc], targets, chunks_by_document[doc]
                    )
                    if self.condition_locator is not None:
                        locations = record.condition_locations.get(doc)
                        if locations is not None:
                            validate_located_plan(
                                plan, locations, targets, chunks_by_document[doc]
                            )
                        elif plan.bindings:
                            raise ValueError(
                                "Condition bindings have no saved locations"
                            )
                    # Separate annotations, never overwrite original conditions,
                    # preparation variables, measured values or partition keys.
                    scoped = [
                        dict(
                            row,
                            supplementary_attributes=_supplementary_attributes(
                                plan, row["measurement_id"], record.annotation_format
                            ),
                            supplementary_review_status="pending",
                            **(
                                {
                                    "supplementary_attribute_format": "refs-v2",
                                    "supplementary_condition_plan_sha256": _digest(
                                        plan.model_dump(mode="json")
                                    ),
                                }
                                if record.annotation_format == "refs-v2"
                                else {}
                            ),
                        )
                        for row in scoped
                    ]
                if self.targeted_planner is not None:
                    targets = {row["measurement_id"]: row for row in scoped}
                    if doc not in record.targeted_plans:
                        if record.status == "complete":
                            raise ValueError("Completed targeted checkpoint missing")
                        calls.check()
                        targeted = self.targeted_planner(
                            calls=calls,
                            candidates=targets,
                            chunks=chunks_by_document[doc],
                        )
                        targeted = validate_targeted_evidence(
                            targeted, targets, chunks_by_document[doc]
                        )
                        revision(
                            targeted_plans={**record.targeted_plans, doc: targeted}
                        )
                    targeted = validate_targeted_evidence(
                        record.targeted_plans[doc], targets, chunks_by_document[doc]
                    )
                    scoped = [
                        dict(
                            row,
                            targeted_attributes=_supplementary_attributes(
                                targeted.attributes, row["measurement_id"], "refs-v2"
                            ),
                            targeted_evidence_sha256=_digest(
                                targeted.model_dump(mode="json")
                            ),
                            targeted_review_status="pending",
                        )
                        for row in scoped
                    ]
                if doc not in record.dataset_ids:
                    dataset = datasets.register_records(
                        scoped,
                        source_artifact_id="artifact-fulltext-scope-" + policy[:24],
                        display_name=task.task_id + "-scoped-trial.json",
                    )
                    revision(
                        dataset_ids={**record.dataset_ids, doc: dataset.dataset_id}
                    )
                dataset_id = record.dataset_ids[doc]
                frame = datasets.load_dataframe(dataset_id)
                if (
                    json.loads(frame.to_json(orient="records", force_ascii=False))
                    != scoped
                ):
                    raise ValueError("Scoped dataset differs from current snapshot")
                if doc not in record.analysis_ids:
                    calls.check()
                    if calls.budget - calls.calls < 3:
                        raise _Pause()
                    # Reserve the actual agent's three bounded routing steps before
                    # invoking it. No prose-model fallback or inferential tests.
                    calls.calls += 3
                    agent = self.agent_factory()
                    context = (
                        agent if hasattr(agent, "__enter__") else nullcontext(agent)
                    )
                    with context as runner:
                        outcome = runner.ask(
                            message=f"描述统计 {dataset_id} columns=numeric_value "
                            "group_by=measurement_context"
                        )
                    tools += outcome.tool_call_count
                    if (
                        outcome.status != "completed"
                        or outcome.final_status != "completed"
                        or outcome.error
                    ):
                        raise ValueError("DataAnalysisAgent failed")
                    if outcome.model_call_count != 3 or outcome.tool_call_count != 2:
                        raise ValueError(
                            "DataAnalysisAgent exceeded explicit operation contract"
                        )
                    ids = tuple(
                        dict.fromkeys(
                            re.findall(r"analysis-[a-f0-9]{32}", outcome.response_text)
                        )
                    )
                    if len(ids) != 1:
                        raise ValueError("Missing explicit analysis result reference")
                    analysis = datasets.get_analysis(ids[0])
                    self._validate_analysis(analysis, frame, dataset_id)
                    if analysis.evidence_id not in outcome.evidence_ids:
                        raise ValueError("Analysis evidence reference mismatch")
                    revision(
                        analysis_ids={**record.analysis_ids, doc: analysis.analysis_id},
                        analysis_digests={
                            **record.analysis_digests,
                            doc: _digest(analysis.model_dump(mode="json")),
                        },
                    )
                analysis = datasets.get_analysis(record.analysis_ids[doc])
                self._validate_analysis(analysis, frame, dataset_id)
                if record.analysis_digests[doc] != _digest(
                    analysis.model_dump(mode="json")
                ):
                    raise ValueError("Saved analysis changed")
                evidence_ids.add(analysis.evidence_id)
                calls.check()
            for doc, ref in selected.items():
                if (
                    source_identities(refresh_sources(doc))
                    != self.snapshots.load(ref).source_identities
                ):
                    raise ValueError("Source changed during statistics")
            if record.status != "complete":
                save(stage="reporting", resume_stage=None)
                markdown = self._report(task, record, datasets, by_doc, query_rows)
                revision(status="complete", report_markdown=markdown)
            save(stage="finished", resume_stage=None)
            return finish(record.report_markdown)
        except _Pause as exc:
            save(stage="ready_to_resume", resume_stage="analyzing")
            return finish(
                "已保存任务范围、成功统计与恢复点；未启动下一分析。"
                if exc.cancelled
                else "本轮预算已到；已保存成功阶段，继续时复核并接续未完成分析。",
                None if exc.cancelled else "FULLTEXT_ANALYSIS_BUDGET",
                cancelled=exc.cancelled,
            )
        except Exception:
            save(stage="failed", resume_stage="analyzing")
            return finish(
                "任务范围、数据分析或报告复核失败；提取快照和成功统计保留，未冒充完整结果。可有界重试。",
                "FULLTEXT_ANALYSIS_FAILED",
            )

    @staticmethod
    def _validate_scope(scope, candidates, requirements):
        ids = []
        for metric in scope.metrics:
            if metric.task_quote not in requirements:
                raise ValueError("Requirement quote is not from this task")
            if _computed_only(metric.task_quote):
                raise ValueError("Database computed metric is outside literature scope")
            if metric.role == "supplementary":
                _validate_supplement(metric, candidates, requirements)
            elif not _metric_named_in_quote(
                metric.requested_metric or metric.name, metric.task_quote
            ):
                raise ValueError(
                    "Analysis metric is not explicitly requested in its quote"
                )
            for mid in metric.measurement_ids:
                if mid in candidates and not _scope_row_matches(
                    metric, candidates[mid]
                ):
                    raise ValueError(
                        "Selected candidate metric does not match scope metric"
                    )
            ids.extend(metric.measurement_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(candidates):
            raise ValueError("Duplicate or unknown scoped measurement")
        excluded = scope.excluded_measurements
        if (
            not set(excluded).issubset(candidates)
            or set(excluded).intersection(ids)
            or any(
                not reason.strip() or len(reason) > 1000 for reason in excluded.values()
            )
        ):
            raise ValueError("Invalid explicit measurement exclusions")
        if _expected_candidate_ids(candidates, requirements) - set(ids) - set(excluded):
            raise ValueError("Explicitly requested candidate omitted without reason")

    @staticmethod
    def _validate_analysis(analysis, frame, dataset_id):
        if (
            analysis.dataset_id != dataset_id
            or analysis.analysis_type != "descriptive"
            or analysis.method != "describe"
            or analysis.parameters.get("columns") != ["numeric_value"]
            or analysis.parameters.get("group_by") != "measurement_context"
        ):
            raise ValueError("Wrong analysis operation or dataset")
        stats = analysis.summary.get("statistics", [])
        actual = frame.groupby("measurement_context", dropna=False)
        if len(stats) != len(actual):
            raise ValueError("Wrong statistics partition coverage")
        seen = set()
        for row in stats:
            key = row["group"]
            if key in seen or key not in actual.groups:
                raise ValueError("Unknown or duplicate statistics group")
            seen.add(key)
            values = actual.get_group(key)["numeric_value"]
            if (
                row["count"] != len(values)
                or row["mean"] != float(values.mean())
                or row["min"] != float(values.min())
                or row["max"] != float(values.max())
            ):
                raise ValueError("Statistics do not match explicit input rows")

    def _report(self, task, record, datasets, by_doc, query_rows):
        lines = [
            "# 主控科研任务分区报告",
            "",
            "执行：当前有界试运行已结束；科研覆盖：partial。",
            "审核：pending（待审核，仅供试运行），不是材料专业验证。",
            "",
            "## 原问题",
            "",
            task.original_question,
            "",
            "## 数据库计算属性（不等于论文实验性能）",
            "",
        ]
        if query_rows:
            for query_id, rows in query_rows.items():
                lines.extend(
                    [
                        f"来源：{query_id}；共 {len(rows)} 个结构，仅展示前 20 个；"
                        "未重新查询数据库。",
                        "",
                        "| ID | 化学式 | 计算带隙 eV | 凸包能 eV/atom | 密度 g/cm³ |",
                        "|---|---|---:|---:|---:|",
                    ]
                )
                for row in rows[:20]:
                    lines.append(
                        f"| {_cell(row.material_id)} | {_cell(row.formula_pretty)} | "
                        f"{row.band_gap_ev} | {row.energy_above_hull_ev_atom} | "
                        f"{row.density_g_cm3} |"
                    )
        else:
            lines.append("本次没有可展示的任务数据库快照；不虚构计算属性。")
        lines.extend(
            [
                "",
                "原数据库数据集引用：" + ", ".join(task.database_dataset_ids),
                "既有计算分析引用：" + ", ".join(task.analysis_ids),
                "",
                "## 文献选择与新提取",
                "",
            ]
        )
        for doc, decision in task.preview_decisions.items():
            preview = next(row for row in task.previews if row.document_id == doc)
            lines.append(f"{preview.title}：{decision.action}；{decision.reason}")
            if doc in record.selected_snapshots:
                snapshot = self.snapshots.load(record.selected_snapshots[doc])
                handoff = task.measurement_handoffs[doc]
                lines.append(
                    f"新提取快照：{snapshot.snapshot_id}；"
                    f"候选标量 {handoff.record_count} 条；"
                    f"隔离 {len(handoff.isolated_measurement_ids)} 条。"
                )
        lines.extend(
            [
                "",
                "## 本轮实验指标范围（模型提出，尚未专业审核）",
                "",
                "主要指标用于回答用户要求；补充指标仅提供辅助证据，不代表主要任务已完成。"
                "不同原始指标、单位、样品及条件仍隔离统计，补充用途不等于专业认可。",
            ]
        )
        for metric in record.scope.metrics:
            label = (
                "用户要求的主要指标"
                if metric.role == "requested"
                else "补充指标（不替代主要任务）"
            )
            lines.append(
                f"- [{label}] {metric.name}："
                f"纳入 {len(metric.measurement_ids)} 条候选；"
                f"任务依据：{metric.task_quote}"
            )
            if metric.role == "supplementary":
                lines.append(
                    "  补充用途（模型提出，待审核）："
                    f"{_cell(metric.supplementary_reason)}"
                )
            if metric.requested_metric is not None:
                lines.append(
                    f"  对应请求指标：{_cell(metric.requested_metric)}；"
                    "模型提出且通过字面名称/单位门槛，仍需专业审核；原指标不改名。"
                )
            if not metric.measurement_ids:
                lines.append("  缺项：在本轮候选中未匹配；不能据此断言论文没有该指标。")
        if record.scope.excluded_measurements:
            lines.append("模型明确排除的候选（理由尚未专业审核，原记录保留）：")
            lines.extend(
                f"- {_cell(mid)}：{_cell(reason)}"
                for mid, reason in record.scope.excluded_measurements.items()
            )
        if record.scope_exclusions:
            lines.append(
                f"材料系列字面范围门槛：另有 {len(record.scope_exclusions)} "
                "条候选未纳入统计；"
                "原候选保留，不推断材料别名，也不把复合物成分默认当作额外纯相对照。"
            )
        ids = {mid for metric in record.scope.metrics for mid in metric.measurement_ids}
        roles = {
            mid: metric.role
            for metric in record.scope.metrics
            for mid in metric.measurement_ids
        }
        scoped_rows = [
            row
            for rows in by_doc.values()
            for row in rows
            if row["measurement_id"] in ids
        ]
        lines.extend(["", "### 请求条件的记录覆盖（仅检查字段，不代表条件可比）", ""])
        explicit_conditions = _explicit_report_conditions(
            task.original_question + "\n" + "\n".join(task.user_instructions)
        )
        for label, keys in explicit_conditions:
            missing = sum(
                all(_metadata(row["conditions"]).get(key) in (None, "") for key in keys)
                for row in scoped_rows
            )
            lines.append(
                f"- 任务明确要求：{label}；未绑定／未知 "
                f"{missing}/{len(scoped_rows)} 条。"
                "仅检查原记录中明确命名的条件字段，不代表完整实验协议或条件可比；"
                "不从相邻段落补值，也不能据此断言论文没有报告该条件。"
            )
        if not explicit_conditions and not record.scope.required_conditions:
            lines.append("本轮未识别明确的条件字段清单；不代表条件齐全或已可比。")
        for key in record.scope.required_conditions:
            missing = sum(
                _metadata(row["conditions"]).get(key) in (None, "")
                for row in scoped_rows
            )
            lines.append(
                f"- 条件 `{_cell(key)}`：缺失 {missing}/{len(scoped_rows)} 条；"
                "字段名由模型提出，未从相邻段落补推或认定同条件。"
            )
        if record.scope.independent_variable:
            key = record.scope.independent_variable
            missing = sum(
                _metadata(row["variables"]).get(key) in (None, "")
                for row in scoped_rows
            )
            lines.append(
                f"- 自变量 `{_cell(key)}`：未绑定 {missing}/{len(scoped_rows)} 条；"
                "本轮不据此拟合趋势。"
            )
        if record.condition_plans:
            lines.extend(
                [
                    "",
                    "## 补充条件证据（pending，不代表条件可比）",
                    "",
                    "以下仅核验字面来源、页码和样品身份；未修改原提取条件，"
                    "也未批准专业适用性、完整测试协议或可比集合。",
                    "",
                    "| 文献 | 类型/字段 | 原文字面值 | "
                    "绑定条目数 | 来源页 | 原文证据 |",
                    "|---|---|---|---:|---|---|",
                ]
            )
            for doc, plan in record.condition_plans.items():
                for binding in plan.bindings:
                    span = binding.value_source
                    lines.append(
                        f"| {_cell(doc)} | {binding.kind}/{_cell(binding.key)} | "
                        f"{_cell(binding.value_text)} | "
                        f"{len(binding.measurement_ids)} | "
                        f"{span.page_from}–{span.page_to} | {_cell(span.quote)} |"
                    )
                if not plan.bindings:
                    lines.append(
                        f"{_cell(doc)}：没有已绑定的补充属性，不能视为同条件。"
                    )
                lines.extend(f"- {_cell(note)}" for note in plan.unresolved)
                if plan.isolated_attributes:
                    lines.append("隔离的补充属性（不是实验事实或统计输入）：")
                    lines.extend(
                        f"- {_cell(row.key)}：{_cell(row.reason)}；"
                        "模型提案未纳入条件绑定。"
                        for row in plan.isolated_attributes
                    )
        if record.targeted_plans:
            lines.extend(
                [
                    "",
                    "## 逐样品时间与报告比例（pending，尚未建立可比集合）",
                    "",
                    "时间仅绑定明确点名的样品，不向相邻复合样品传播。比例同时保存制备定义与材料语境，"
                    "不解释为确定分母的质量分数；原测量条件与统计分区不改。",
                    "",
                    "| 样品条目 | 字段 | 字面值 | 页码 | 原文证据 |",
                    "|---|---|---|---|---|",
                ]
            )
            for plan in record.targeted_plans.values():
                for binding in plan.attributes.bindings:
                    mid = binding.measurement_ids[0]
                    span = binding.value_source
                    lines.append(
                        f"| {_cell(mid)} | {_cell(binding.key)} | "
                        f"{_cell(binding.value_text)} | "
                        f"{span.page_from}–{span.page_to} | {_cell(span.quote)} |"
                    )
                    if mid in plan.preparation_sources:
                        for source in plan.preparation_sources[mid]:
                            lines.append(
                                f"- 制备来源 {source.page_from}–{source.page_to} 页："
                                f"{_cell(source.quote)}"
                            )
                lines.extend(f"- {_cell(note)}" for note in plan.attributes.unresolved)
                if plan.attributes.isolated_attributes:
                    lines.append("逐样品提案隔离：不是实验事实，未填入条件或比例。")
                    lines.extend(
                        f"- {_cell(item.key)}：{_cell(item.reason)}"
                        for item in plan.attributes.isolated_attributes
                    )
        if record.figure_evidence_ref is not None:
            from .figure_evidence_gate import render_figure_appendix

            batch = self.figure_review_service.report_batch(
                task, record.figure_evidence_ref
            )
            lines.append(render_figure_appendix(batch))
        lines.extend(["", "## 来源、样品与条件隔离的描述统计", ""])
        if not record.analysis_ids:
            lines.append(
                "没有本任务可安全统计的选定标量；跳过统计，没有用空表制造结果。"
            )
        for doc, analysis_id in record.analysis_ids.items():
            analysis = datasets.get_analysis(analysis_id)
            lines.extend(
                [
                    render_descriptive_report(
                        analysis, datasets.get(record.dataset_ids[doc])
                    ).answer,
                    "记录数是条目数，不是独立实验重复数；单点均值只是原值，不支持显著性或可靠趋势。",
                    "",
                    "| 分析用途 | 样品 | 指标 | 原始值 | 条件 | 来源页 |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for row in by_doc[doc]:
                if row["measurement_id"] in ids:
                    purpose = {"requested": "主要", "supplementary": "补充"}[
                        roles[row["measurement_id"]]
                    ]
                    lines.append(
                        f"| {purpose} | "
                        f"{_cell(row['group_label'])} | {_cell(row['metric'])} | "
                        f"{_measurement_value_cell(row)} | "
                        f"{_cell(row['conditions'])} | {row['page_from']} |"
                    )
        lines.extend(
            [
                "",
                "## 综合判断与未解决事项",
                "",
                "数据库纯相计算属性与复合样品实验值分别列示，"
                "未把实验性能归给 MP 结构；未跨论文、单位、光源或污染物排名。",
                "条件缺项与指标范围仍需核对；未知条件不等于相同条件。本文只完成有来源的隔离描述，不得作为最佳材料推荐。",
            ]
        )
        if any(op in record.scope.requested_operations for op in ("compare", "trend")):
            lines.append(
                "请求的比较/趋势未执行：当前只接入隔离描述，尚未独立建立可比集合与条件证据。不能把降级报告称为已完成所请求的趋势分析。"
            )
        lines.extend(f"- {note}" for note in record.scope.limitations)
        lines.extend(
            [
                "",
                "作者主张与通用提取器的 pending 比较记录"
                "没有当作新的独立实验或统计结论。",
                "本轮有界提取不是完整全文覆盖验收；后续可补充条件证据或启动新的任务分析尝试。",
            ]
        )
        return "\n".join(lines)
