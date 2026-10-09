"""Probe the documented Intern Chat Completions tool-call surface safely."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import SecretStr

from materials_screening.agent.intern_model import InternAgentModel
from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import AgentFinalDraft, AgentMessageItem
from materials_screening.agent.settings import AgentSettings
from materials_screening.agent.tool_base import AgentToolDefinition, ToolSideEffect
from materials_screening.sub_agents.materials_database.prompt import SYSTEM_PROMPT
from materials_screening.sub_agents.materials_database.models import (
    CompareMaterialsInput,
    DescribeMaterialsInput,
    DetectOutliersInput,
    ExportMaterialsInput,
    GetMaterialDetailsInput,
    GetQueryResultInput,
    SearchMaterialsInput,
)


def _tool(name: str, description: str, model: type[Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": model.model_json_schema(),
        },
    }


def _summarize(label: str, response: Any) -> None:
    choice = response.choices[0]
    message = choice.message
    calls = [call.function.name for call in (message.tool_calls or [])]
    content = str(message.content or "").replace("\n", " ")[:160]
    print(
        f"{label}: finish_reason={choice.finish_reason!r}; "
        f"tool_calls={calls!r}; content={content!r}"
    )


def main() -> None:
    load_dotenv()
    token = os.getenv("INTERN_API_KEY", "").strip()
    if not token:
        raise SystemExit("INTERN_API_KEY is not set")
    client = OpenAI(
        api_key=token,
        base_url=os.getenv(
            "INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/"
        ),
        timeout=120.0,
        max_retries=0,
    )
    model = os.getenv("INTERN_MODEL", "intern-s2-preview-35b")
    tools = [
        _tool("search_materials", "Search materials.", SearchMaterialsInput),
        _tool("get_material_details", "Get details.", GetMaterialDetailsInput),
        _tool("get_query_result", "Read query results.", GetQueryResultInput),
        _tool("compare_materials", "Compare materials.", CompareMaterialsInput),
        _tool("describe_materials", "Describe results.", DescribeMaterialsInput),
        _tool("detect_material_outliers", "Detect outliers.", DetectOutliersInput),
        _tool("export_materials", "Export results.", ExportMaterialsInput),
    ]
    base = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": "Call search_materials for stable oxides with band gap above 2 eV.",
        }],
        "temperature": 0.0,
        "max_tokens": 1024,
        "stream": False,
        "extra_body": {"thinking_mode": True},
    }
    _summarize("plain", client.chat.completions.create(**base))
    _summarize(
        "one_tool", client.chat.completions.create(**base, tools=tools[:1])
    )
    _summarize(
        "seven_tools", client.chat.completions.create(**base, tools=tools)
    )
    definitions = tuple(
        AgentToolDefinition(
            name=item["function"]["name"],
            description=item["function"]["description"],
            parameters=item["function"]["parameters"],
            side_effect=ToolSideEffect.READ_ONLY,
            version="1",
        )
        for item in tools
    )
    adapter = InternAgentModel(
        AgentSettings(
            agent_base_url=os.getenv(
                "INTERN_BASE_URL", "https://chat.intern-ai.org.cn/api/v1/"
            ),
            agent_model=model,
            agent_thinking_mode=True,
        ),
        client=client,
        api_key=SecretStr(token),
    )
    result = adapter.generate(MaterialAgentRequest(
        instructions=SYSTEM_PROMPT,
        input_items=(AgentMessageItem(
            role="user",
            content="寻找带隙大于 2 eV 的稳定氧化物",
        ),),
        tool_definitions=definitions,
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        max_output_tokens=8192,
        temperature=0.0,
        allow_tool_calls=True,
    ))
    print(
        "adapter: "
        f"tool_calls={[call.name for call in result.tool_calls]!r}; "
        f"message={result.message_text!r}"
    )
    language_request = MaterialAgentRequest(
        instructions=SYSTEM_PROMPT,
        input_items=(AgentMessageItem(
            role="user", content="寻找带隙大于 2 eV 的稳定氧化物"
        ),),
        tool_definitions=(),
        final_draft_schema=AgentFinalDraft.model_json_schema(),
        allow_tool_calls=False,
    )
    translated = adapter._align_final_language(
        language_request,
        AgentFinalDraft(
            status="completed",
            answer="The materials search completed successfully.",
            warnings=[],
        ),
    )
    print(f"language_repair: answer={translated.answer!r}")


if __name__ == "__main__":
    main()
