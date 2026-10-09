"""Deterministic first-tool routing for the outlier sub-agent."""

from __future__ import annotations

import hashlib
import json

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentModel,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)


class OutlierDetectionRoutingModel:
    """Force the required tool call, then delegate final writing to the LLM.

    The outlier agent has exactly one public tool and its system contract says
    every concrete request must call it. Routing this first step in code avoids
provider-specific automatic tool-selection drift (plain text instead of
    a function call), while the wrapped model still writes the evidence-based
    final answer.
    """

    def __init__(self, final_model: MaterialAgentModel) -> None:
        self._final_model = final_model

    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        if self._has_tool_result_after_last_user(request):
            return self._final_model.generate(request)

        query = self._last_user_message(request)
        if not query:
            return self._final_model.generate(request)
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        call = AgentFunctionCallItem(
            call_id=f"outlier_{digest}",
            name="run_outlier_detection",
            arguments=json.dumps(
                {"query": query}, ensure_ascii=False, separators=(",", ":")
            ),
        )
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(call,),
            request_id=f"outlier-route-{digest}",
            provider="deterministic-router",
            model="outlier-tool-router",
        )

    @staticmethod
    def _last_user_message(request: MaterialAgentRequest) -> str | None:
        for item in reversed(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                return item.content
        return None

    @staticmethod
    def _has_tool_result_after_last_user(request: MaterialAgentRequest) -> bool:
        last_user_index = -1
        for index, item in enumerate(request.input_items):
            if isinstance(item, AgentMessageItem) and item.role == "user":
                last_user_index = index
        return any(
            isinstance(item, AgentFunctionOutputItem)
            for item in request.input_items[last_user_index + 1 :]
        )
