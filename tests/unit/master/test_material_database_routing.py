"""Deterministic routing tests for the unified materials database agent."""

import pytest

from materials_screening.master.master_nodes import (
    _is_data_analysis_request,
    _is_literature_evidence_request,
    _is_material_database_request,
    _literature_analysis_passthrough,
    _literature_handoff_dataset_id,
    _material_query_handoff_id,
    _material_screening_chain_passthrough,
    _materials_database_passthrough,
    _requires_material_screening_chain,
)


@pytest.mark.parametrize(
    "message",
    [
        "寻找带隙大于 2 eV 的稳定氧化物",
        "比较 mp-149 和 mp-13 的密度",
        "统计刚才这些材料的形成能",
        "导出 Materials Project 查询结果",
        "检查带隙离群值",
    ],
)
def test_material_database_requests_are_recognized(message: str) -> None:
    assert _is_material_database_request(message)


@pytest.mark.parametrize("message", ["你好", "你能做什么", "谢谢"])
def test_conversation_is_not_forced_to_database(message: str) -> None:
    assert not _is_material_database_request(message)


@pytest.mark.parametrize(
    "message",
    [
        "检查 dataset-12345678 的数据质量",
        "分析上传的 CSV 缺失值",
        "对实验数据做 ANOVA",
    ],
)
def test_data_analysis_requests_are_recognized(message: str) -> None:
    assert _is_data_analysis_request(message)


def test_synthesis_relationship_question_routes_to_literature() -> None:
    assert _is_literature_evidence_request(
        "纯、未负载nHA水热合成中，pH和温度对物相和形貌的影响"
    )


def test_plain_materials_database_query_does_not_route_to_literature() -> None:
    assert not _is_literature_evidence_request("寻找带隙大于 2 eV 的稳定氧化物")


def test_complete_screening_request_requires_three_agent_chain() -> None:
    assert _requires_material_screening_chain(
        "筛选带隙为2.8～4.5 eV、凸包能不高于0.05 eV/atom且不含Pb、Cd、Hg"
        "的氧化物半导体，分析候选的稳定性与密度分布，并结合实验文献评价"
        "前5名用于紫外光电探测的可行性。"
    )


def test_successful_database_answer_is_preserved_without_resummarizing() -> None:
    answer = (
        "查询结果如下：\n\n"
        "| 材料 ID | 化学式 | 带隙 |\n"
        "|---|---|---|\n"
        "| mp-2542 | BeO | 7.46 eV |\n\n"
        "数据库中共有约2000个符合条件的氧化物。"
    )
    draft = _materials_database_passthrough(
        {
            "model_call_count": 1,
            "executed_call_ids": ["delegate-1"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "materials_database",
                    "call_id": "delegate-1",
                    "response_text": answer,
                    "active_workflow_thread_id": None,
                    "warnings": [],
                }
            ],
        }
    )

    assert draft is not None
    assert "| mp-2542 | BeO | 7.46 eV |" in draft["answer"]
    assert "不代表数据库总数" in draft["answer"]
    assert draft["referenced_material_ids"] == ["mp-2542"]
    assert draft["evidence_ids"] == ["delegate-1"]
    assert draft["status"] == "completed"


def test_successful_literature_answer_is_preserved_without_resummarizing() -> None:
    answer = (
        "检索到 20 篇候选文献，以下按相关性列出：\n\n"
        "1. 2024 — 3D-printed bioactive glass scaffolds — "
        "DOI: 10.1000/example\n"
        "2. 2023 — Pore architecture and osteogenesis — "
        "DOI: 10.1000/example-2"
    )
    draft = _materials_database_passthrough(
        {
            "model_call_count": 1,
            "executed_call_ids": ["literature-delegation"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "literature",
                    "call_id": "literature-delegation",
                    "response_text": answer,
                    "active_workflow_thread_id": None,
                    "warnings": [],
                }
            ],
        }
    )

    assert draft is not None
    assert draft["answer"] == answer
    assert "请提供" not in draft["answer"]
    assert draft["referenced_material_ids"] == []
    assert draft["evidence_ids"] == ["literature-delegation"]
    assert draft["status"] == "completed"


def test_database_passthrough_does_not_mask_errors() -> None:
    assert (
        _materials_database_passthrough(
            {
                "model_call_count": 1,
                "executed_call_ids": ["delegate-1"],
                "sub_agent_results": [
                    {
                        "status": "error",
                        "sub_agent_name": "materials_database",
                        "call_id": "delegate-1",
                        "response_text": "failed",
                    }
                ],
            }
        )
        is None
    )


def test_literature_error_does_not_leak_master_retry_monologue() -> None:
    draft = _materials_database_passthrough(
        {
            "model_call_count": 1,
            "executed_call_ids": ["literature-delegation"],
            "sub_agent_results": [
                {
                    "status": "error",
                    "sub_agent_name": "literature",
                    "call_id": "literature-delegation",
                    "response_text": "",
                    "error": {
                        "code": "SUB_AGENT_FAILED",
                        "message": "invalid final draft",
                    },
                }
            ],
        }
    )

    assert draft is not None
    assert "无需重新描述任务" in draft["answer"]
    assert "让我尝试" not in draft["answer"]
    assert draft["warnings"] == ["invalid final draft"]


def test_previous_turn_database_result_is_not_replayed() -> None:
    assert (
        _materials_database_passthrough(
            {
                "model_call_count": 0,
                "executed_call_ids": [],
                "sub_agent_results": [
                    {
                        "status": "ok",
                        "sub_agent_name": "materials_database",
                        "call_id": "previous-delegation",
                        "response_text": "上一轮查询结果",
                        "warnings": [],
                    }
                ],
            }
        )
        is None
    )


def test_only_current_turn_database_result_is_preserved() -> None:
    draft = _materials_database_passthrough(
        {
            "model_call_count": 1,
            "executed_call_ids": ["current-delegation"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "materials_database",
                    "call_id": "previous-delegation",
                    "response_text": "上一轮筛选结果",
                    "warnings": [],
                },
                {
                    "status": "ok",
                    "sub_agent_name": "materials_database",
                    "call_id": "current-delegation",
                    "response_text": "本轮离群检测结果",
                    "warnings": [],
                },
            ],
        }
    )

    assert draft is not None
    assert draft["answer"] == "本轮离群检测结果"
    assert draft["evidence_ids"] == ["current-delegation"]


def test_literature_dataset_handoff_is_detected() -> None:
    state = {
        "executed_call_ids": ["literature-delegation"],
        "sub_agent_results": [
            {
                "status": "ok",
                "sub_agent_name": "literature",
                "call_id": "literature-delegation",
                "response_text": (
                    "已复用文献报告\n"
                    "LITERATURE_DATASET_HANDOFF: dataset_id=dataset-test"
                ),
                "warnings": [],
            }
        ],
    }

    assert _literature_handoff_dataset_id(state) == "dataset-test"
    assert _materials_database_passthrough(state) is None


def test_material_query_handoff_is_detected() -> None:
    state = {
        "executed_call_ids": ["database-delegation"],
        "sub_agent_results": [
            {
                "status": "ok",
                "sub_agent_name": "materials_database",
                "call_id": "database-delegation",
                "response_text": (
                    "候选表\nMATERIAL_QUERY_HANDOFF: query_id=query-test123"
                ),
            }
        ],
    }

    assert _material_query_handoff_id(state) == "query-test123"


def test_literature_and_data_analysis_are_combined() -> None:
    draft = _literature_analysis_passthrough(
        {
            "executed_call_ids": ["literature-delegation", "analysis-delegation"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "literature",
                    "call_id": "literature-delegation",
                    "response_text": (
                        "已复用文献报告\n"
                        "LITERATURE_DATASET_HANDOFF: dataset_id=dataset-test"
                    ),
                    "warnings": ["仅供试运行"],
                },
                {
                    "status": "ok",
                    "sub_agent_name": "data_analysis",
                    "call_id": "analysis-delegation",
                    "response_text": "描述统计已完成。",
                    "warnings": [],
                },
            ],
        }
    )

    assert draft is not None
    assert "## 文献 Agent 结果" in draft["answer"]
    assert "## 数据分析 Agent 结果" in draft["answer"]
    assert "仅供试运行" in draft["answer"]
    assert "LITERATURE_DATASET_HANDOFF" not in draft["answer"]
    assert draft["evidence_ids"] == [
        "literature-delegation",
        "analysis-delegation",
    ]


def test_full_material_screening_chain_is_combined_with_boundary() -> None:
    draft = _material_screening_chain_passthrough(
        {
            "user_message": (
                "筛选氧化物，分析分布，并结合实验文献评价前5名用于紫外光电探测的可行性"
            ),
            "executed_call_ids": ["db", "da", "lit"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "materials_database",
                    "call_id": "db",
                    "response_text": (
                        "| mp-1 | TiO2 |\n"
                        "MATERIAL_QUERY_HANDOFF: query_id=query-test123"
                    ),
                    "warnings": [],
                },
                {
                    "status": "ok",
                    "sub_agent_name": "data_analysis",
                    "call_id": "da",
                    "response_text": "密度描述统计完成。",
                    "warnings": [],
                },
                {
                    "status": "ok",
                    "sub_agent_name": "literature",
                    "call_id": "lit",
                    "response_text": "找到 TiO2 紫外探测器实验文献。",
                    "warnings": [],
                },
            ],
        }
    )

    assert draft is not None
    assert "## 1. Materials Project 初筛" in draft["answer"]
    assert "## 2. 数据库属性初排（文献预检池前 5 名）" in draft["answer"]
    assert "## 4. 候选池文献预检、证据重排与下载清单" in draft["answer"]
    assert "不是最终推荐" in draft["answer"]
    assert "MATERIAL_QUERY_HANDOFF" not in draft["answer"]
    assert draft["referenced_material_ids"] == ["mp-1"]
    assert draft["evidence_ids"] == ["db", "da", "lit"]


def test_full_material_screening_chain_reports_literature_provider_failure() -> None:
    draft = _material_screening_chain_passthrough(
        {
            "user_message": (
                "筛选氧化物，分析分布，并结合实验文献评价前5名用于紫外光电探测的可行性"
            ),
            "executed_call_ids": ["db", "da", "lit"],
            "sub_agent_results": [
                {
                    "status": "ok",
                    "sub_agent_name": "materials_database",
                    "call_id": "db",
                    "response_text": (
                        "| mp-1 | TiO2 |\n"
                        "MATERIAL_QUERY_HANDOFF: query_id=query-test123"
                    ),
                    "warnings": [],
                },
                {
                    "status": "ok",
                    "sub_agent_name": "data_analysis",
                    "call_id": "da",
                    "response_text": "密度描述统计完成。",
                    "warnings": [],
                },
                {
                    "status": "error",
                    "sub_agent_name": "literature",
                    "call_id": "lit",
                    "response_text": "",
                    "warnings": [],
                    "error": {
                        "code": "SUB_AGENT_EXECUTION_FAILED",
                        "message": "all literature providers failed",
                    },
                },
            ],
        }
    )

    assert draft is not None
    assert "文献核验本轮执行失败" in draft["answer"]
    assert "不能形成紫外探测可行性结论" in draft["answer"]
    assert "all literature providers failed" in draft["warnings"]
    assert draft["evidence_ids"] == ["db", "da", "lit"]
