"""Regression tests for the offline literature model's final contract."""

from __future__ import annotations

import json

from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import AgentFunctionOutputItem
from materials_screening.sub_agents.literature.mock_model import LiteratureMockModel


def test_mock_literature_answer_does_not_claim_database_ids_as_paper_evidence() -> None:
    output = {
        "status": "ok",
        "evidence_id": "evidence-1",
        "output": {
            "query_id": "lit-1",
            "expanded_query": {
                "original_topic": "验证 mp-1 (TiO2) 和 mp-2 (ZnO) 的实验文献"
            },
            "papers": [],
            "warnings": ["offline"],
        },
    }
    response = LiteratureMockModel().generate(
        MaterialAgentRequest(
            input_items=(
                AgentFunctionOutputItem(
                    call_id="lit-call", output=json.dumps(output)
                ),
            ),
            instructions="literature",
            tool_definitions=(),
            final_draft_schema={"type": "object"},
        )
    )

    draft = json.loads(response.message_text)
    assert draft["referenced_material_ids"] == []
    assert "mp-1" not in draft["answer"]
    assert "mp-2" not in draft["answer"]
    assert "不能据此评价任何候选材料" in draft["answer"]
