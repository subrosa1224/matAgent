"""Small deterministic model for offline database-agent tests."""

from __future__ import annotations

import json

from materials_screening.agent.model_base import (
    AgentModelStatus,
    MaterialAgentRequest,
    MaterialAgentResponse,
)
from materials_screening.agent.models import (
    AgentFunctionCallItem,
    AgentFunctionOutputItem,
    AgentMessageItem,
)


class MaterialsDatabaseMockModel:
    def generate(self, request: MaterialAgentRequest) -> MaterialAgentResponse:
        outputs = [
            item
            for item in request.input_items
            if isinstance(item, AgentFunctionOutputItem)
        ]
        if outputs:
            envelope = json.loads(outputs[-1].output)
            output = envelope.get("output", envelope)
            evidence = envelope.get("evidence_id") or output.get("evidence_id")
            material_ids: set[str] = set()
            _collect_material_ids(output, material_ids)
            content = {
                "status": "completed",
                "answer": json.dumps(output, ensure_ascii=False),
                "referenced_material_ids": sorted(material_ids),
                "evidence_ids": [evidence] if evidence else [],
                "warnings": [],
            }
            return MaterialAgentResponse(
                status=AgentModelStatus.COMPLETED,
                output_items=(
                    AgentMessageItem(
                        role="assistant",
                        content=json.dumps(content, ensure_ascii=False),
                    ),
                ),
                request_id="mock-db-final",
                provider="mock",
                model="mock",
            )
        args = {"fields": ["material_id", "formula_pretty"], "limit": 20}
        return MaterialAgentResponse(
            status=AgentModelStatus.COMPLETED,
            output_items=(
                AgentFunctionCallItem(
                    call_id="mock-db-search",
                    name="search_materials",
                    arguments=json.dumps(args),
                ),
            ),
            request_id="mock-db-call",
            provider="mock",
            model="mock",
        )


def _collect_material_ids(value: object, output: set[str]) -> None:
    if isinstance(value, dict):
        material_id = value.get("material_id")
        if isinstance(material_id, str) and material_id:
            output.add(material_id)
        for nested in value.values():
            _collect_material_ids(nested, output)
    elif isinstance(value, list):
        for nested in value:
            _collect_material_ids(nested, output)
