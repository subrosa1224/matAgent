# ruff: noqa: E501
"""Intern agent model using the OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)
from pydantic import SecretStr, ValidationError

from materials_screening.agent.errors import AgentModelError
from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFinalDraft,
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)
from materials_screening.agent.settings import AgentSettings
from materials_screening.llm.intern_transport import (
    connection_failure_detail,
    create_completion,
    create_intern_client,
)

_PROVIDER = "intern"
_GENERIC_PROVIDER_ERRORS = frozenset(
    {
        "an error occurred while processing your prompt. please check your input and try again.",
        "there was an error processing your prompt. please try again or provide more details about what you're trying to accomplish.",
    }
)


class InternAgentModel:
    """Adapter for InternLM's ``/chat/completions`` function-calling API."""

    def __init__(
        self,
        settings: AgentSettings,
        *,
        client: OpenAI | None = None,
        api_key: SecretStr | None = None,
    ) -> None:
        key = api_key or self._api_key_from_env()
        self._settings = settings
        self._client = client or create_intern_client(
            api_key=key.get_secret_value(),
            base_url=settings.agent_base_url,
            timeout_seconds=settings.agent_model_timeout_seconds,
            use_system_proxy=settings.intern_use_system_proxy,
        )

    @staticmethod
    def _api_key_from_env() -> SecretStr:
        raw = os.getenv("INTERN_API_KEY", "").strip()
        if not raw:
            raise AgentModelError("INTERN_API_KEY is not set")
        return SecretStr(raw)

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        started = time.perf_counter()
        if _forced_literature_search_tool(request) == "screen_candidate_literature":
            # Some compatible model services ignore tool_choice. This exact
            # structured handoff must not fall back to ordinary 6-query search.
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentFunctionCallItem(
                        call_id="candidate-pool-screen",
                        name="screen_candidate_literature",
                        arguments=self._normalize_tool_arguments(
                            request, "screen_candidate_literature", "{}"
                        ),
                    ),
                ),
                request_id="deterministic-candidate-pool-call",
                provider=_PROVIDER,
                model=self._settings.agent_model,
            )
        deterministic_draft = self._literature_result_draft(request)
        if deterministic_draft is not None:
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentMessageItem(
                        role="assistant",
                        content=deterministic_draft.model_dump_json(),
                    ),
                ),
                request_id="deterministic-literature-result",
                provider=_PROVIDER,
                model=self._settings.agent_model,
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
        response = self._call_api(request)
        choice = response.choices[0] if response.choices else None
        if choice is None:
            raise AgentModelError("Intern returned no choices")
        usage = getattr(response, "usage", None)
        finish = str(getattr(choice, "finish_reason", "") or "")
        status = (
            AgentModelStatus.INCOMPLETE
            if finish == "length"
            else AgentModelStatus.COMPLETED
        )
        message = choice.message
        items: list[AgentMessageItem | AgentFunctionCallItem] = []
        for call in getattr(message, "tool_calls", None) or []:
            name = str(call.function.name)
            items.append(
                AgentFunctionCallItem(
                    call_id=str(call.id),
                    name=name,
                    arguments=self._normalize_tool_arguments(
                        request, name, str(call.function.arguments)
                    ),
                )
            )
        if not items:
            content = str(getattr(message, "content", "") or "").strip()
            if not content:
                user_text = self._last_user_text(request)
                seed_answer = (
                    "查询工具执行完成。"
                    if re.search(r"[\u3400-\u9fff]", user_text)
                    else "The query tool completed."
                )
                seed = AgentFinalDraft(status="completed", answer=seed_answer)
                draft = self._ground_final_draft(request, seed)
                if not draft.evidence_ids or draft.answer == seed_answer:
                    raise AgentModelError("Intern returned an empty final message")
                # A complete evidence table does not depend on the truncated or
                # empty model message, so publish it as a completed response.
                status = AgentModelStatus.COMPLETED
            else:
                if _looks_like_schema_description(content):
                    raise AgentModelError(
                        "Intern described the internal response contract"
                    )
                if content.casefold() in _GENERIC_PROVIDER_ERRORS:
                    raise AgentModelError(
                        "Intern rejected the prompt with a generic service error"
                    )
                payload = _extract_json_object(content)
                try:
                    draft = AgentFinalDraft.model_validate(payload)
                except (ValidationError, TypeError):
                    # Intern sometimes returns a JSON object that is syntactically
                    # valid but omits required envelope fields on short follow-ups
                    # such as "yes". Treat that the same as a plain-text answer and
                    # give the strict serializer one opportunity to normalize it.
                    try:
                        repaired = self._repair_final_payload(request, content)
                        draft = AgentFinalDraft.model_validate(repaired)
                    except (AgentModelError, ValidationError):
                        # Protocol fields are deterministic metadata. If Intern's
                        # serializer also fails, preserve its actual answer and
                        # construct the safe envelope locally instead of failing a
                        # successful database turn because of malformed wrapping.
                        draft = self._local_final_draft(content, payload)
            if draft.answer.strip().casefold() in _GENERIC_PROVIDER_ERRORS:
                raise AgentModelError(
                    "Intern rejected the prompt with a generic service error"
                )
            draft = self._ground_final_draft(request, draft)
            draft = self._align_final_language(request, draft)
            items.append(
                AgentMessageItem(role="assistant", content=draft.model_dump_json())
            )
        return MaterialAgentResponse(
            status=status,
            output_items=tuple(items),
            request_id=str(getattr(response, "id", "") or "unknown"),
            provider=_PROVIDER,
            model=str(getattr(response, "model", "") or self._settings.agent_model),
            latency_ms=round((time.perf_counter() - started) * 1000),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            error="Intern response exceeded max_tokens"
            if status is AgentModelStatus.INCOMPLETE
            else None,
        )

    def _call_api(self, request: MaterialAgentRequest) -> Any:
        kwargs: dict[str, Any] = {
            "model": self._settings.agent_model,
            "messages": self._messages(request),
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
            "stream": False,
            "extra_body": {"thinking_mode": self._settings.agent_thinking_mode},
        }
        if request.allow_tool_calls:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": d.name,
                        "description": d.description,
                        "parameters": d.parameters,
                    },
                }
                for d in request.tool_definitions
            ]
            forced_tool = _forced_literature_search_tool(request)
            if forced_tool is not None:
                # Topic discovery must produce provider evidence. Optional tool
                # selection occasionally lets the model invent uncited papers.
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": forced_tool},
                }
        return self._create_completion(kwargs)

    def _repair_final_payload(
        self, request: MaterialAgentRequest, content: str
    ) -> dict[str, Any]:
        """Convert a non-JSON final answer into the required safe envelope."""
        schema = json.dumps(request.final_draft_schema, ensure_ascii=False)
        response = self._create_completion(
            {
                "model": self._settings.agent_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON serializer. Convert the supplied "
                            "assistant answer into exactly one JSON object matching this "
                            f"schema: {schema}. Use status='completed' unless the answer "
                            "asks the user for information. Preserve the answer text "
                            "and its original language; do not translate it. "
                            "Use empty arrays for referenced_material_ids, evidence_ids, "
                            "and warnings when they are not explicitly present. Use null "
                            "for optional fields when absent. Output JSON only."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
                "max_tokens": request.max_output_tokens,
                "stream": False,
                "extra_body": {"thinking_mode": False},
            }
        )
        choice = response.choices[0] if response.choices else None
        repaired = str(
            getattr(getattr(choice, "message", None), "content", "") or ""
        ).strip()
        payload = _extract_json_object(repaired)
        if payload is None:
            raise AgentModelError(
                "Intern final output remained invalid JSON after repair"
            )
        return payload

    @staticmethod
    def _local_final_draft(
        content: str, payload: dict[str, Any] | None
    ) -> AgentFinalDraft:
        """Build the protocol envelope locally from a model answer payload."""
        source = payload if isinstance(payload, dict) else {}
        answer = next(
            (
                value.strip()
                for key in ("answer", "response", "final_answer", "content", "message")
                if isinstance((value := source.get(key)), str) and value.strip()
            ),
            "",
        )
        if not answer and payload is None:
            answer = content.strip()
        if not answer:
            raise AgentModelError(
                "Intern final output does not contain a usable answer"
            )

        status = source.get("status")
        if status not in {"completed", "needs_user_input", "error"}:
            status = "completed"
        follow_up = source.get("follow_up_question")
        if not isinstance(follow_up, str) or not follow_up.strip():
            follow_up = None
        if status == "needs_user_input" and follow_up is None:
            status = "completed"

        def string_list(key: str) -> list[str]:
            value = source.get(key)
            if not isinstance(value, list):
                return []
            return [item for item in value if isinstance(item, str) and item]

        active_thread = source.get("active_workflow_thread_id")
        if not isinstance(active_thread, str):
            active_thread = None
        return AgentFinalDraft(
            status=status,
            answer=answer,
            active_workflow_thread_id=active_thread,
            referenced_material_ids=string_list("referenced_material_ids"),
            evidence_ids=string_list("evidence_ids"),
            warnings=string_list("warnings"),
            follow_up_question=follow_up,
        )

    def _create_completion(self, kwargs: dict[str, Any]) -> Any:
        try:
            return create_completion(
                self._client,
                kwargs,
                max_attempts=self._settings.agent_model_max_attempts,
                timeout_seconds=self._settings.agent_model_timeout_seconds,
            )
        except APITimeoutError as exc:
            raise AgentModelError(
                f"Intern request timed out ({connection_failure_detail(exc)})"
            ) from exc
        except APIConnectionError as exc:
            raise AgentModelError(
                f"Intern connection failed ({connection_failure_detail(exc)})"
            ) from exc
        except AuthenticationError as exc:
            raise AgentModelError("Intern authentication failed") from exc
        except PermissionDeniedError as exc:
            raise AgentModelError("Intern permission denied") from exc
        except NotFoundError as exc:
            raise AgentModelError("Intern model or endpoint not found") from exc
        except BadRequestError as exc:
            raise AgentModelError("Intern rejected the request payload") from exc
        except RateLimitError as exc:
            raise AgentModelError("Intern rate limit exceeded") from exc
        except InternalServerError as exc:
            raise AgentModelError("Intern service error") from exc
        except APIStatusError as exc:
            raise AgentModelError(
                f"Intern API status error: {exc.status_code}"
            ) from exc
        except APIError as exc:
            raise AgentModelError("Intern API error") from exc

    def _align_final_language(
        self, request: MaterialAgentRequest, draft: AgentFinalDraft
    ) -> AgentFinalDraft:
        """Repair an obvious Chinese/English mismatch without changing evidence."""
        user_text = self._last_user_text(request)
        if not user_text:
            return draft
        user_has_cjk = bool(re.search(r"[\u3400-\u9fff]", user_text))
        answer_has_cjk = bool(re.search(r"[\u3400-\u9fff]", draft.answer))
        if user_has_cjk == answer_has_cjk:
            return draft
        target_language = "Simplified Chinese (简体中文)" if user_has_cjk else "English"
        schema = json.dumps(request.final_draft_schema, ensure_ascii=False)
        for attempt in range(2):
            retry_note = (
                " The previous translation did not use the required language; "
                "this attempt must comply."
                if attempt
                else ""
            )
            response = self._create_completion(
                {
                    "model": self._settings.agent_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a faithful translator, not a question-answering "
                                "assistant. Translate only answer, warnings, and "
                                "follow_up_question into "
                                f"{target_language}. Do not add, remove, infer, summarize, "
                                "or alter any factual claim. Do not change status, IDs, "
                                "evidence, formulas, field names, numbers, or units. "
                                f"Return one JSON object matching this schema: {schema}."
                                f"{retry_note}"
                            ),
                        },
                        {"role": "user", "content": draft.model_dump_json()},
                    ],
                    "temperature": 0.0,
                    "max_tokens": request.max_output_tokens,
                    "stream": False,
                    "extra_body": {"thinking_mode": False},
                }
            )
            choice = response.choices[0] if response.choices else None
            content = str(
                getattr(getattr(choice, "message", None), "content", "") or ""
            ).strip()
            payload = _extract_json_object(content)
            if payload is None:
                continue
            try:
                edited = AgentFinalDraft.model_validate(payload)
            except ValidationError:
                continue
            edited_has_cjk = bool(re.search(r"[\u3400-\u9fff]", edited.answer))
            if user_has_cjk != edited_has_cjk:
                continue
            # Structural and evidence-bearing fields always come from the
            # original; the language editor may change text fields only.
            return draft.model_copy(
                update={
                    "answer": edited.answer,
                    "warnings": edited.warnings,
                    "follow_up_question": edited.follow_up_question,
                }
            )
        compact = {
            "answer": draft.answer,
            "warnings": draft.warnings,
            "follow_up_question": draft.follow_up_question,
        }
        instruction = (
            "你是忠实翻译器。只把输入 JSON 的文本值翻译成简体中文，绝对不能"
            "回答问题、增删事实或修改数字、ID、化学式、字段名和单位。只输出"
            "包含 answer、warnings、follow_up_question 的 JSON。"
            if user_has_cjk
            else (
                "You are a faithful translator. Translate only the JSON text "
                "values into English without adding or removing facts. Return "
                "only answer, warnings, and follow_up_question as JSON."
            )
        )
        response = self._create_completion(
            {
                "model": self._settings.agent_model,
                "messages": [
                    {"role": "system", "content": instruction},
                    {
                        "role": "user",
                        "content": json.dumps(compact, ensure_ascii=False),
                    },
                ],
                "temperature": 0.0,
                "max_tokens": request.max_output_tokens,
                "stream": False,
                "extra_body": {"thinking_mode": False},
            }
        )
        choice = response.choices[0] if response.choices else None
        content = str(
            getattr(getattr(choice, "message", None), "content", "") or ""
        ).strip()
        payload = _extract_json_object(content)
        translated_answer = payload.get("answer") if payload else None
        if not isinstance(translated_answer, str) or not translated_answer.strip():
            raise AgentModelError(
                f"Intern did not produce the required {target_language} answer"
            )
        translated_has_cjk = bool(re.search(r"[\u3400-\u9fff]", translated_answer))
        if user_has_cjk != translated_has_cjk:
            raise AgentModelError(
                f"Intern did not produce the required {target_language} answer"
            )
        warnings = payload.get("warnings", draft.warnings)
        follow_up = payload.get("follow_up_question", draft.follow_up_question)
        return draft.model_copy(
            update={
                "answer": translated_answer,
                "warnings": warnings if isinstance(warnings, list) else draft.warnings,
                "follow_up_question": (
                    follow_up
                    if isinstance(follow_up, str) or follow_up is None
                    else draft.follow_up_question
                ),
            }
        )

    @staticmethod
    def _ground_final_draft(
        request: MaterialAgentRequest, draft: AgentFinalDraft
    ) -> AgentFinalDraft:
        """Fill omitted grounding metadata only from successful tool evidence."""
        evidence_ids: list[str] = []
        evidenced_material_ids: set[str] = set()
        material_rows: list[dict[str, Any]] = []
        material_fields: list[str] = []
        material_output: dict[str, Any] = {}
        literature_output: dict[str, Any] = {}
        last_successful_tool_name = ""
        for item in request.input_items:
            if not isinstance(item, AgentFunctionOutputItem):
                continue
            try:
                envelope = json.loads(item.output)
            except json.JSONDecodeError:
                continue
            if not isinstance(envelope, dict) or envelope.get("status") != "ok":
                continue
            tool_name = envelope.get("tool_name")
            if isinstance(tool_name, str):
                last_successful_tool_name = tool_name
            evidence_id = envelope.get("evidence_id")
            output = envelope.get("output")
            if not evidence_id and isinstance(output, dict):
                evidence_id = output.get("evidence_id")
            if isinstance(evidence_id, str) and evidence_id:
                evidence_ids.append(evidence_id)
            if isinstance(output, dict):
                if tool_name in {
                    "literature_search",
                    "openalex_search",
                    "s2_search",
                    "screen_candidate_literature",
                }:
                    literature_output = output
                rows = output.get("materials")
                fields = output.get("fields")
                if isinstance(rows, list) and all(
                    isinstance(row, dict) for row in rows
                ):
                    material_rows = rows
                    material_output = output
                    material_fields = (
                        [field for field in fields if isinstance(field, str)]
                        if isinstance(fields, list)
                        else []
                    )
            evidenced_material_ids.update(
                re.findall(r"\bmp-[A-Za-z0-9]+\b", item.output)
            )
        if not evidence_ids:
            return draft
        answer_material_ids = set(re.findall(r"\bmp-[A-Za-z0-9]+\b", draft.answer))
        updates: dict[str, Any] = {}
        if not draft.evidence_ids:
            updates["evidence_ids"] = list(dict.fromkeys(evidence_ids))
        if not draft.referenced_material_ids and answer_material_ids:
            updates["referenced_material_ids"] = sorted(
                answer_material_ids & evidenced_material_ids
            )
        user_text = InternAgentModel._last_user_text(request)
        explicitly_requests_rows = any(
            marker in user_text.casefold()
            for marker in (
                "返回",
                "列出",
                "展示",
                "显示",
                "明细",
                "表格",
                "list",
                "show",
                "display",
                "table",
            )
        )
        row_producing_tools = {
            "search_materials",
            "get_material_details",
            "get_query_result",
            "compare_materials",
        }
        if (
            material_rows
            and explicitly_requests_rows
            and last_successful_tool_name in row_producing_tools
        ):
            table, table_ids = InternAgentModel._material_table(
                material_rows, material_fields
            )
            if table:
                total = material_output.get("matched_count")
                if not isinstance(total, int):
                    total = material_output.get("total")
                if not isinstance(total, int):
                    total = len(material_rows)
                shown = min(len(material_rows), 20)
                if re.search(r"[\u3400-\u9fff]", user_text):
                    summary = f"查询完成：共匹配 {total} 条，当前展示 {shown} 条。"
                else:
                    summary = (
                        f"Query completed: {total} matches; showing {shown} records."
                    )
                updates["answer"] = f"{summary}\n\n{table}"
                updates["referenced_material_ids"] = table_ids
                tool_warnings = material_output.get("warnings")
                updates["warnings"] = (
                    [
                        warning
                        for warning in tool_warnings
                        if isinstance(warning, str) and warning.strip()
                    ]
                    if isinstance(tool_warnings, list)
                    else []
                )
                updates["follow_up_question"] = None
        if isinstance(literature_output.get("finalists"), list):
            updates["answer"] = InternAgentModel._candidate_screen_answer(
                literature_output
            )
            tool_warnings = literature_output.get("warnings")
            updates["warnings"] = (
                [str(warning) for warning in tool_warnings]
                if isinstance(tool_warnings, list)
                else []
            )
            updates["follow_up_question"] = None
        papers = literature_output.get("papers")
        if isinstance(papers, list):
            shown_papers = [paper for paper in papers[:15] if isinstance(paper, dict)]
            query_id = literature_output.get("query_id")
            detail_offset = _literature_detail_offset(user_text)
            all_remaining = _requests_all_remaining_literature_details(user_text)
            detail_limit = len(shown_papers) if all_remaining else detail_offset + 5
            lines = [
                (
                    f"继续展示候选论文第 {detail_offset + 1}–"
                    f"{min(detail_limit, len(shown_papers))} 篇的详细信息。"
                    if detail_offset
                    else f"检索完成：找到 {literature_output.get('returned_count', len(papers))} 篇候选文献。"
                )
            ]
            if isinstance(query_id, str) and query_id:
                lines.append(f"统一检索 ID：`{query_id}`")
            retrieval_mode = literature_output.get("retrieval_mode", "unknown")
            mode_labels = {
                "cache": "缓存复用，本次未联网检索",
                "fresh": "本次检索",
                "mixed": "本次检索与历史来源混合，部分元数据并非本次取得",
                "unknown": "未记录，不能据此证明本次联网检索",
            }
            lines.append(
                f"检索方式：{mode_labels.get(retrieval_mode, mode_labels['unknown'])}。"
            )
            if literature_output.get("created_at"):
                lines.append(f"来源快照时间：{literature_output['created_at']}。")

            if not detail_offset:
                from materials_screening.sub_agents.literature.search_coverage import (
                    render_search_coverage,
                )

                coverage = render_search_coverage(literature_output)
                if coverage:
                    lines.extend(("", coverage))

            expanded = literature_output.get("expanded_query")
            statuses = literature_output.get("provider_statuses")
            has_provider_queries = isinstance(statuses, list) and any(
                isinstance(status, dict) and status.get("search_queries")
                for status in statuses
            )
            queries = (
                expanded.get("search_queries") if isinstance(expanded, dict) else None
            )
            if (
                not detail_offset
                and not has_provider_queries
                and isinstance(queries, list)
                and queries
            ):
                lines.extend(("", "**实际检索式**"))
                lines.extend(
                    f"- {query}" for query in queries if isinstance(query, str)
                )

            if not detail_offset and isinstance(statuses, list) and statuses:
                lines.extend(
                    (
                        "",
                        "**数据源状态（来源快照，非本次请求）**"
                        if retrieval_mode == "cache"
                        else "**数据源状态**",
                    )
                )
                for status in statuses:
                    if not isinstance(status, dict):
                        continue
                    provider = status.get("provider") or "unknown"
                    label = "正常" if status.get("status") == "ok" else "降级"
                    returned = status.get("records_returned", 0)
                    detail = f"- {provider}：{label}，返回 {returned} 条"
                    warning = status.get("warning")
                    if isinstance(warning, str) and warning.strip():
                        detail += f"；{warning.strip()}"
                    lines.append(detail)
                    submitted = status.get("submitted_materials")
                    unqueried = status.get("unqueried_materials")
                    failed = status.get("failed_materials")
                    if isinstance(submitted, list) and (
                        submitted or unqueried or failed
                    ):
                        lines.append(
                            f"  - 候选查询：已提交 {len(submitted)} 种；"
                            f"失败 {len(failed or [])} 种；"
                            f"未查询 {len(unqueried or [])} 种。"
                            "提交查询不表示已找到论文或完整召回。"
                        )
                        if unqueried:
                            lines.append("  - 未查询：" + "、".join(unqueried))
                        if failed:
                            lines.append("  - 查询失败：" + "、".join(failed))
                    at_limit = status.get("queries_at_record_limit")
                    if isinstance(at_limit, int) and at_limit > 0:
                        lines.append(
                            f"  - {at_limit} 个查询达到记录返回上限，未遍历全部文献。"
                        )
                    provider_queries = status.get("search_queries")
                    if isinstance(provider_queries, list) and provider_queries:
                        lines.append(
                            "  - 来源快照检索式："
                            if retrieval_mode == "cache"
                            else "  - 实际检索式："
                        )
                        lines.extend(
                            f"    - {query}"
                            for query in provider_queries
                            if isinstance(query, str)
                        )

            if not detail_offset:
                lines.extend(
                    (
                        "",
                        "**候选论文**",
                        "",
                        "| # | 相关性 | 年份 | 论文题目 | DOI |",
                        "|---:|---|---:|---|---|",
                    )
                )
            relevance_labels = {
                "core": "核心相关",
                "high": "高度相关",
                "extended": "扩展阅读",
            }
            access_labels = {
                "metadata_only": "仅元数据",
                "abstract_available": "有摘要",
                "open_access_reported": "来源报告开放获取",
            }
            for index, paper in enumerate(shown_papers, 1):
                if detail_offset:
                    break
                title = str(paper.get("title") or "—")[:240].replace("|", "\\|")
                doi = str(paper.get("doi") or "—")[:120].replace("|", "\\|")
                relevance = relevance_labels.get(
                    str(paper.get("relevance_level") or ""), "未分级"
                )
                lines.append(
                    f"| {index} | {relevance} | {paper.get('year') or '—'} | {title} | {doi} |"
                )

            detailed_papers = shown_papers[detail_offset:detail_limit]
            if detailed_papers:
                detail_end = detail_offset + len(detailed_papers)
                lines.extend(
                    (
                        "",
                        f"**重点论文详情（第{detail_offset + 1}–{detail_end}篇）**",
                    )
                )
            for index, paper in enumerate(detailed_papers, detail_offset + 1):
                title = str(paper.get("title") or "—")[:240]
                lines.extend(("", f"**{index}. {title}**"))
                authors = paper.get("authors")
                if isinstance(authors, list) and authors:
                    visible_authors = "、".join(str(author) for author in authors[:5])
                    if len(authors) > 5:
                        visible_authors += " 等"
                    lines.append(f"- 作者：{visible_authors}")
                if paper.get("venue"):
                    lines.append(f"- 期刊/会议：{paper['venue']}")
                citations = paper.get("cited_by_count")
                lines.append(
                    f"- 引用数：{citations if citations is not None else '未知'}"
                )
                access = access_labels.get(
                    str(paper.get("access_status") or ""), "状态未知"
                )
                lines.append(f"- 获取状态：{access}")
                if paper.get("landing_page_url"):
                    lines.append(f"- 文献入口：{paper['landing_page_url']}")
                lines.append(
                    f"- 入选原因：{paper.get('selection_reason') or '主题相关'}"
                )
                abstract = paper.get("abstract")
                if isinstance(abstract, str) and abstract.strip():
                    preview = " ".join(abstract.split())
                    abstract_limit = 120 if all_remaining else 300
                    if len(preview) > abstract_limit:
                        preview = f"{preview[: abstract_limit - 3]}..."
                    lines.append(f"- 摘要：{preview}")
                else:
                    lines.append("- 摘要：数据源未提供")
            if not detail_offset and len(shown_papers) > len(detailed_papers):
                lines.extend(
                    (
                        "",
                        f"表格列出 {len(shown_papers)} 篇；为保证页面稳定，"
                        f"详细信息展开前 {len(detailed_papers)} 篇。",
                    )
                )
            if len(papers) > len(shown_papers):
                lines.extend(("", f"当前展示前 {len(shown_papers)} 篇。"))
            updates["answer"] = "\n".join(lines)
            tool_warnings = literature_output.get("warnings")
            updates["warnings"] = (
                [str(warning) for warning in tool_warnings]
                if isinstance(tool_warnings, list)
                else []
            )
            updates["follow_up_question"] = None
        return draft.model_copy(update=updates) if updates else draft

    @staticmethod
    def _literature_result_draft(
        request: MaterialAgentRequest,
    ) -> AgentFinalDraft | None:
        """Render a current-turn search result without a second model pass."""

        latest_user_index = -1
        latest_user_text = ""
        for index, item in enumerate(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                latest_user_index = index
                latest_user_text = item.content
        if latest_user_index < 0:
            return None
        search_start = (
            0 if _literature_detail_offset(latest_user_text) else latest_user_index + 1
        )
        for item in reversed(request.input_items[search_start:]):
            if not isinstance(item, AgentFunctionOutputItem):
                continue
            try:
                envelope = json.loads(item.output)
            except json.JSONDecodeError:
                continue
            if not isinstance(envelope, dict) or envelope.get("status") != "ok":
                continue
            if envelope.get("tool_name") not in {
                "literature_search",
                "openalex_search",
                "s2_search",
                "screen_candidate_literature",
            }:
                continue
            seed = AgentFinalDraft(status="completed", answer="检索完成。")
            grounded = InternAgentModel._ground_final_draft(request, seed)
            return grounded if grounded.answer != seed.answer else None
        return None

    @staticmethod
    def _candidate_screen_answer(output: dict[str, Any]) -> str:
        finalists = output.get("finalists")
        rows = finalists if isinstance(finalists, list) else []
        lines = [
            "候选池题名/摘要预检："
            f"计划 {output.get('candidate_count', 0)} 种材料，"
            f"其中 {output.get('qualifying_candidate_count', 0)} 种具有 A/B 级线索。",
            f"预检快照：`{output.get('screening_id', '—')}`",
            (
                f"检索方式：{output.get('retrieval_mode', 'unknown')}；"
                f"本次联网 {output.get('queries_attempted', 0)} 次；"
                f"缓存命中 {output.get('cache_hits', 0)} 次。"
                if output.get("retrieval_mode", "unknown") != "unknown"
                else "检索方式未记录；历史尝试计数不代表本次联网。"
            ),
            "A/B 为元数据自动预分级，不是全文确认或材料可行性结论；化学式相同也不代表同一物相。",
            "",
            "**文献线索重排后的重点核验对象**",
            "",
            "| 排名 | 化学式 | 原属性排名 | 证据等级 | 匹配论文数 | 说明 |",
            "|---:|---|---:|---|---:|---|",
        ]
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                continue
            note = str(row.get("note") or "—").replace("|", "\\|")
            lines.append(
                f"| {index} | {row.get('formula', '—')} | "
                f"{row.get('original_rank', '—')} | "
                f"{row.get('evidence_grade', 'NONE')} | "
                f"{row.get('matching_paper_count', 0)} | {note} |"
            )
        source_rows = [
            row
            for row in rows
            if isinstance(row, dict) and row.get("source_created_at")
        ]
        if source_rows:
            lines.extend(("", "**元数据来源时间**"))
            lines.extend(
                f"- {row.get('formula', '—')}：{row['source_created_at']}；"
                f"来源查询：{row.get('source_query_id', '—')}。"
                for row in source_rows
            )
        if not rows:
            lines.append(
                "本轮没有满足 A/B 预分级的重点核验对象；不能用 C/NONE 凑名额。"
            )
        supplementary = output.get("supplementary_finalists")
        supplementary_rows = supplementary if isinstance(supplementary, list) else []
        if supplementary_rows:
            lines.extend(
                (
                    "",
                    "**独立探索池重点核验对象（不是原严格条件达标名单）**",
                    "",
                    "下调计算带隙下限并限二元氧化物，不并入原统计，实验光学带隙待全文核验。",
                    "",
                    "| 化学式 | 预分级 | 匹配论文数 |",
                    "|---|---|---:|",
                )
            )
            for row in supplementary_rows:
                if isinstance(row, dict):
                    lines.append(
                        f"| {row.get('formula', '—')} | {row.get('evidence_grade', 'NONE')} | {row.get('matching_paper_count', 0)} |"
                    )
        if output.get("retrieval_complete") is False:
            lines.append(
                "检索未完成：失败或未尝试的材料证据未知，不能判定其没有文献；可重试。"
            )
        downloads = output.get("download_candidates")
        papers = downloads if isinstance(downloads, list) else []
        lines.extend(("", "**建议下载并进行全文核验的论文**"))
        if not papers:
            lines.append("- 本轮没有 A/B 级论文可供下载；不能据此形成器件可行性结论。")
        for index, paper in enumerate(papers[:15], 1):
            if not isinstance(paper, dict):
                continue
            title = str(paper.get("title") or "题名未知")[:160]
            doi = str(paper.get("doi") or "无 DOI")
            grade = str(paper.get("application_evidence_grade") or "未分级")
            url = str(
                paper.get("landing_page_url")
                or (f"https://doi.org/{doi}" if doi != "无 DOI" else "无公开入口")
            )
            lines.append(f"{index}. [{grade}级] {title}；DOI: {doi}；入口: {url}")
        if len(papers) > 15:
            lines.append(
                f"此处展示前 15 篇，全部 {len(papers)} 篇及其材料关联保留在预检快照。"
            )
        if len(rows) < int(output.get("candidate_count", 0) or 0):
            lines.extend(
                (
                    "",
                    f"请求最多 {output.get('final_limit', 5)} 种，实际列出 {len(rows)} 种；完整候选与检索状态保留在预检快照。",
                )
            )
        if papers:
            lines.append(
                "请下载上述可获得的论文并上传 PDF，再核验实验物相、波长、响应度/探测率和测试条件。未获取全文前不输出已验证的器件推荐。"
            )
        return "\n".join(lines)

    @staticmethod
    def _material_table(
        rows: list[dict[str, Any]], fields: list[str]
    ) -> tuple[str, list[str]]:
        """Render evidenced search rows when the model omitted requested data."""
        visible_fields = [
            field for field in fields if any(field in row for row in rows)
        ]
        if not visible_fields:
            visible_fields = list(rows[0]) if rows else []
        if not visible_fields:
            return "", []
        labels = {
            "material_id": "材料 ID",
            "formula_pretty": "化学式",
            "band_gap_ev": "带隙 (eV)",
            "density_g_cm3": "密度 (g/cm³)",
            "formation_energy_ev_atom": "形成能 (eV/atom)",
            "energy_above_hull_ev_atom": "凸包上能量 (eV/atom)",
            "is_stable": "稳定性",
            "elements": "元素",
            "crystal_system": "晶系",
            "spacegroup_symbol": "空间群符号",
            "spacegroup_number": "空间群编号",
        }
        shown = rows[:20]
        header = "| " + " | ".join(labels.get(f, f) for f in visible_fields) + " |"
        divider = "| " + " | ".join("---" for _ in visible_fields) + " |"
        lines = [header, divider]
        material_ids: list[str] = []
        for row in shown:
            values: list[str] = []
            for field in visible_fields:
                value = row.get(field)
                if field == "material_id" and isinstance(value, str):
                    material_ids.append(value)
                if isinstance(value, float):
                    rendered = f"{value:.6g}"
                elif value is True:
                    rendered = "是"
                elif value is False:
                    rendered = "否"
                elif value is None:
                    rendered = "—"
                elif field == "elements" and isinstance(value, (list, tuple)):
                    rendered = "、".join(str(element) for element in value)
                else:
                    rendered = str(value)
                values.append(rendered.replace("|", "\\|"))
            lines.append("| " + " | ".join(values) + " |")
        if len(rows) > len(shown):
            remaining = len(rows) - len(shown)
            lines.append(
                f"\n仅展示前 {len(shown)} 条，共返回 {len(rows)} 条。"
                f"如需查看其余 {remaining} 条，请回复“查看剩余材料”或“显示下一页”。"
            )
        return "\n".join(lines), list(dict.fromkeys(material_ids))

    @staticmethod
    def _last_user_text(request: MaterialAgentRequest) -> str:
        for item in reversed(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                return item.content
        return ""

    @staticmethod
    def _normalize_tool_arguments(
        request: MaterialAgentRequest, tool_name: str, raw: str
    ) -> str:
        """Coerce scalar strings to arrays only where the tool schema requires it."""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if not isinstance(payload, dict):
            return raw
        definition = next(
            (item for item in request.tool_definitions if item.name == tool_name),
            None,
        )
        if definition is None:
            return raw
        properties = definition.parameters.get("properties")
        if not isinstance(properties, dict):
            return raw
        changed = False
        if tool_name == "literature_search" and "research_question" in properties:
            user_text = InternAgentModel._last_user_text(request)
            from materials_screening.sub_agents.literature.topic_scope import (
                candidate_handoff_projection,
            )

            handoff = candidate_handoff_projection(user_text)
            if handoff is not None:
                question, candidates, topic = handoff
                payload["research_question"] = question
                payload["topic"] = topic
                payload["material_keywords"] = list(candidates)
                changed = True
            original = re.search(
                r"原始科研问题\s*[：:]\s*(.*?)\s*候选化学式\s*[：:]",
                user_text,
                re.DOTALL,
            )
            if original is not None and 0 < len(original.group(1).strip()) <= 4000:
                payload["research_question"] = original.group(1).strip()
                changed = True
            elif (
                0 < len(user_text) <= 4000
                and _forced_literature_search_tool(request) == "literature_search"
            ):
                payload["research_question"] = user_text
                changed = True
        for key, value in tuple(payload.items()):
            property_schema = properties.get(key)
            if (
                not isinstance(property_schema, dict)
                or property_schema.get("type") != "array"
                or not isinstance(value, str)
            ):
                continue
            values = [part.strip() for part in value.split(",") if part.strip()]
            payload[key] = values
            changed = True
        if tool_name == "search_materials":
            normalized_user = InternAgentModel._last_user_text(request).casefold()
            exact_markers = (
                "仅含",
                "只含",
                "仅包含",
                "只包含",
                "only contain",
                "only contains",
                "containing only",
                "consisting only",
            )
            elements = payload.get("required_elements")
            if (
                any(marker in normalized_user for marker in exact_markers)
                and isinstance(elements, list)
                and len(elements) >= 2
                and all(isinstance(element, str) and element for element in elements)
            ):
                payload["chemsys"] = "-".join(dict.fromkeys(elements))
                payload["required_elements"] = []
                changed = True

            filters = payload.get("filters")
            if isinstance(filters, list):
                has_hull_threshold = any(
                    isinstance(item, dict)
                    and item.get("field") == "energy_above_hull_ev_atom"
                    and item.get("operator") in {"lt", "lte"}
                    for item in filters
                )
                strict_markers = (
                    "严格稳定",
                    "热力学严格稳定",
                    "位于凸包",
                    "凸包上材料",
                    "is_stable=true",
                    "is_stable = true",
                    "strictly stable",
                    "on the hull",
                )
                if has_hull_threshold and not any(
                    marker in normalized_user for marker in strict_markers
                ):
                    filtered = [
                        item
                        for item in filters
                        if not (
                            isinstance(item, dict)
                            and item.get("field") == "is_stable"
                            and item.get("operator") == "eq"
                            and item.get("value") is True
                        )
                    ]
                    if len(filtered) != len(filters):
                        payload["filters"] = filtered
                        changed = True
        if tool_name == "screen_candidate_literature":
            user_text = InternAgentModel._last_user_text(request)
            # Numbered pool entries are authoritative; do not parse NONE, MP,
            # or other protocol labels as chemical formulas.
            formulas = re.findall(r"(?:^|[；：])\s*\d+:([^；。\s]+)", user_text)
            if not formulas:
                formulas = re.findall(
                    r"\b(?:[A-Z][a-z]?\d*|\((?:[A-Z][a-z]?\d*)+\)\d*){2,}\b",
                    user_text,
                )
            if formulas:
                supplemental = re.findall(
                    r"(?:^|[；：])\s*S\d+:([^；。\s]+)", user_text
                )
                payload["materials"] = list(
                    dict.fromkeys((*formulas[:50], *supplemental[:50]))
                )[:100]
                payload["supplementary_materials"] = supplemental[:50]
                changed = True
            payload["application"] = "UV photodetector"
            limit_match = re.search(r"前\s*(\d{1,2})\s*名", user_text)
            payload["final_limit"] = (
                min(max(int(limit_match.group(1)), 1), 20) if limit_match else 5
            )
            payload["papers_per_candidate"] = 3
            changed = True
        if tool_name in {"search_materials", "get_material_details"}:
            normalized_user = InternAgentModel._last_user_text(request).casefold()
            requested_fields: list[str] = ["material_id", "formula_pretty"]
            field_markers = (
                ("elements", ("元素", "elements")),
                ("band_gap_ev", ("带隙", "band gap", "band_gap")),
                ("density_g_cm3", ("密度", "density")),
                (
                    "formation_energy_ev_atom",
                    ("形成能", "formation energy", "formation_energy"),
                ),
                (
                    "energy_above_hull_ev_atom",
                    ("凸包", "energy above hull", "energy_above_hull"),
                ),
                ("is_stable", ("稳定性", "is_stable", "stability")),
                ("crystal_system", ("晶系", "crystal system", "crystal_system")),
            )
            for field, markers in field_markers:
                if any(marker in normalized_user for marker in markers):
                    requested_fields.append(field)
            if any(
                marker in normalized_user
                for marker in ("空间群", "space group", "spacegroup")
            ):
                requested_fields.extend(("spacegroup_symbol", "spacegroup_number"))
            existing_fields = payload.get("fields")
            if not isinstance(existing_fields, list):
                existing_fields = []
            normalized_fields = list(
                dict.fromkeys((*existing_fields, *requested_fields))
            )
            if normalized_fields != existing_fields:
                payload["fields"] = normalized_fields
                changed = True
        if not changed:
            return raw
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _messages(request: MaterialAgentRequest) -> list[dict[str, Any]]:
        final_template = (
            '{"status":"completed","answer":"...",'
            '"active_workflow_thread_id":null,'
            '"referenced_material_ids":[],"evidence_ids":[],'
            '"warnings":[],"follow_up_question":null}'
        )
        protocol = (
            "You may call an available function when more evidence is required. "
            "Whenever you answer the user instead of calling a function, return "
            "only one valid JSON object matching this final-answer schema template: "
            f"{final_template}. status must be completed, needs_user_input, or error. "
            "Do not describe this template. Do not use Markdown fences or put text "
            "outside the JSON object. The values of answer, warnings, and "
            "follow_up_question must use the same language as the most recent "
            "user message. Keep JSON keys, material IDs, chemical formulas, "
            "field names, and units unchanged."
        )
        if not request.allow_tool_calls:
            protocol += (
                " Function calls are disabled for this turn. You must now "
                "return the final-answer JSON object."
            )
        user_text = InternAgentModel._last_user_text(request)
        if re.search(r"[\u3400-\u9fff]", user_text):
            protocol += (
                " 重要：用户使用中文提问，最终 JSON 中的 answer、warnings 和 "
                "follow_up_question 必须使用简体中文；不得用英文回答。"
            )
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": f"{request.instructions}\n\n{protocol}",
            }
        ]
        for item in request.input_items:
            if isinstance(item, AgentMessageItem):
                messages.append({"role": item.role, "content": item.content})
            elif isinstance(item, AgentFunctionCallItem):
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": item.call_id,
                                "type": "function",
                                "function": {
                                    "name": item.name,
                                    "arguments": item.arguments,
                                },
                            }
                        ],
                    }
                )
            elif isinstance(item, AgentFunctionOutputItem):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.call_id,
                        "content": item.output,
                    }
                )
        return messages


def _looks_like_schema_description(content: str) -> bool:
    """Detect prose that describes the private final-answer contract."""

    folded = content.casefold()
    contract_markers = (
        "agentfinaldraft",
        "json模式定义",
        "json schema",
        "maxlength",
        "minlength",
        "referenced_material_ids",
        "follow_up_question",
    )
    return sum(marker in folded for marker in contract_markers) >= 3


def _forced_literature_search_tool(
    request: MaterialAgentRequest,
) -> str | None:
    """Force provider retrieval for an explicit paper-discovery request."""

    if not request.allow_tool_calls:
        return None

    tool_names = {definition.name for definition in request.tool_definitions}
    if "literature_search" not in tool_names:
        return None

    latest_user_index = -1
    latest_user_text = ""
    for index, item in enumerate(request.input_items):
        if isinstance(item, AgentMessageItem) and item.role == "user":
            latest_user_index = index
            latest_user_text = item.content.strip()
    if latest_user_index < 0 or any(
        isinstance(item, AgentFunctionOutputItem)
        for item in request.input_items[latest_user_index + 1 :]
    ):
        return None

    normalized = latest_user_text.casefold()
    if "候选池文献预检" in normalized and "screen_candidate_literature" in tool_names:
        return "screen_candidate_literature"
    discovery_intent = bool(
        re.search(r"(?:检索|搜索|查找|推荐|寻找).{0,80}(?:论文|文献)", normalized)
        or re.search(
            r"(?:search|find|recommend).{0,30}(?:papers?|literature|articles?)",
            normalized,
        )
    )
    non_search_workflow = any(
        marker in normalized
        for marker in (
            "pdf",
            "上传",
            "预览",
            "快速阅读",
            "深度分析",
            "综合报告",
            "整合报告",
            "analyze",
            "preview",
            "synthesize",
        )
    )
    return "literature_search" if discovery_intent and not non_search_workflow else None


def _literature_detail_offset(user_text: str) -> int:
    """Resolve a bounded metadata-detail page from a search follow-up."""

    normalized = user_text.casefold()
    if any(
        marker in normalized
        for marker in (
            "深度分析",
            "全文分析",
            "实验数据",
            "analyze full",
            "deep analysis",
        )
    ):
        return 0
    numbered = re.search(
        r"第\s*(\d{1,2})\s*(?:[-–—至到]\s*\d{1,2})?\s*篇.{0,20}(?:详情|详细|信息)",
        normalized,
    )
    if numbered:
        return max(0, int(numbered.group(1)) - 1)
    if _requests_all_remaining_literature_details(user_text):
        return 5
    continuation_markers = (
        "后面几篇",
        "后续几篇",
        "其余几篇",
        "剩余几篇",
        "后面的详细",
        "继续显示详情",
        "继续展示详情",
        "remaining papers",
        "next papers",
        "more paper details",
    )
    return 5 if any(marker in normalized for marker in continuation_markers) else 0


def _requests_all_remaining_literature_details(user_text: str) -> bool:
    """Whether a follow-up asks for every remaining displayed candidate."""

    normalized = user_text.casefold()
    return bool(
        re.search(
            r"(?:后面|后续|其余|剩余).{0,20}(?:所有|全部).{0,20}(?:详情|详细|信息)",
            normalized,
        )
        or re.search(
            r"(?:所有|全部).{0,20}(?:后面|后续|其余|剩余).{0,20}(?:详情|详细|信息)",
            normalized,
        )
        or re.search(
            r"(?:all|every).{0,20}(?:remaining|rest).{0,20}(?:details?|information)",
            normalized,
        )
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
