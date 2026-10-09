"""Small deterministic MA-2 routing acceptance set."""

from __future__ import annotations

from materials_screening.agent.model_base import MaterialAgentRequest
from materials_screening.agent.models import AgentMessageItem
from materials_screening.master.mock_model import MasterMockAgentModel

CASES = (
    ("筛选带隙大于2 eV的稳定氧化物", "delegate_to_materials_database"),
    ("查询 mp-149 的密度和形成能", "delegate_to_materials_database"),
    ("比较 mp-13 与 mp-149", "delegate_to_materials_database"),
    ("统计这些材料的带隙", "delegate_to_materials_database"),
    ("找出查询结果中的离群材料", "delegate_to_materials_database"),
    (
        "search the material database for stable oxides",
        "delegate_to_materials_database",
    ),
    ("检索钛酸钡压电性能论文", "delegate_to_literature"),
    ("找近五年的生物陶瓷文献", "delegate_to_literature"),
    ("推荐几篇相关综述", "delegate_to_literature"),
    ("查找 DOI 和参考文献", "delegate_to_literature"),
    ("analyze recent papers about TiO2", "delegate_to_literature"),
    ("literature review on bioactive glass", "delegate_to_literature"),
    ("检查 dataset-12345678 的缺失值", "delegate_to_data_analysis"),
    ("分析上传 CSV 的数据质量", "delegate_to_data_analysis"),
    ("对实验数据做 ANOVA", "delegate_to_data_analysis"),
)


def test_auto_routing_accuracy_is_at_least_90_percent() -> None:
    model = MasterMockAgentModel()
    correct = 0
    for message, expected in CASES:
        request = MaterialAgentRequest(
            input_items=(AgentMessageItem(role="user", content=message),),
            instructions="master",
            tool_definitions=(),
            final_draft_schema={"type": "object"},
        )
        response = model.generate(request)
        correct += int(
            bool(response.tool_calls) and response.tool_calls[0].name == expected
        )
    assert correct / len(CASES) >= 0.9
