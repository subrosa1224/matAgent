"""Minimal test: inspect full DeepSeek response structure."""
import os, json, sys
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from materials_screening.agent.models import AgentFinalDraft
from materials_screening.sub_agents.outlier_detection.prompt import OUTLIER_DETECTION_SYSTEM_PROMPT

api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=60, max_retries=0)

FINAL_SCHEMA = AgentFinalDraft.model_json_schema()

mock_result = json.dumps({
    "status": "ok", "tool_name": "detect_property_outliers", "call_id": "c1",
    "evidence_id": "evt_1", "error": None,
    "output": {
        "property_name": "band_gap_ev", "method": "zscore", "threshold": 2.0,
        "records": [
            {"material_id": "mp-1", "material_label": "TiO2", "value": 3.2, "z_score": -0.3, "direction": "none", "is_outlier": False},
            {"material_id": "mp-5", "material_label": "SiO2", "value": 9.0, "z_score": 3.8, "direction": "high", "is_outlier": True},
        ],
        "distribution": {"count": 5, "mean": 4.38, "median": 3.3, "std": 2.3, "q1": 3.1, "q3": 3.3, "iqr": 0.2, "min_value": 3.0, "max_value": 9.0},
        "warnings": [], "evidence_id": "evt_1",
    },
}, ensure_ascii=False)

resp = client.responses.create(
    model="deepseek-v4-flash",
    instructions=OUTLIER_DETECTION_SYSTEM_PROMPT,
    input=[
        {"role": "user", "content": "比较 TiO2、BaTiO3、SrTiO3、ZnO、SiO2 的带隙，找出离群材料"},
        {"type": "function_call", "call_id": "c1", "name": "detect_property_outliers", "arguments": '{"property":"band_gap_ev"}', "status": "completed"},
        {"type": "function_call_output", "call_id": "c1", "output": mock_result},
    ],
    tools=[{
        "type": "function",
        "name": "detect_property_outliers",
        "description": "Detect single-property outliers (Z-score / IQR).",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }],
    tool_choice="none",
    text={"format": {"type": "json_schema", "name": "final_v1", "schema": FINAL_SCHEMA}},
    reasoning={"effort": "none"},
    temperature=0.0,
)

print(f"Status: {resp.status}")
print(f"Output text: {resp.output_text[:500] if resp.output_text else '(empty)'}")

# Inspect output items
print(f"\nOutput items ({len(resp.output)}):")
for item in resp.output:
    print(f"  type={item.type}, name={getattr(item, 'name', 'N/A')}")
    if hasattr(item, 'arguments'):
        print(f"    args={item.arguments[:200]}")
    if hasattr(item, 'content'):
        for c in item.content:
            print(f"    content type={c.type}, text={getattr(c, 'text', '')[:200]}")

# Check if output has structured function_call
for item in resp.output:
    if getattr(item, 'type', None) == 'function_call':
        print(f"\nFOUND function_call: name={item.name}, call_id={item.call_id}")
        print(f"  arguments={item.arguments[:200]}")
