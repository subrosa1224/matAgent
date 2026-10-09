"""Generate evaluator assets and score manually reviewed end-to-end runs.

This is not an agent runner: it never invokes an LLM or submits reference answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from materials_screening.evaluation.stable_acceptance import (
    asset_path,
    digest,
    frozen_expectation,
    ledger_template,
    read_json,
    score_ledger,
    validate_suite,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/research/stable_acceptance_v1"
OUTPUT_ROOT = ROOT / "outputs/stable_acceptance"


def code_fingerprint() -> str:
    checksum = hashlib.sha256()
    for path in sorted((ROOT / "src/materials_screening").rglob("*.py")):
        checksum.update(path.relative_to(ROOT).as_posix().encode())
        checksum.update(path.read_bytes())
    return checksum.hexdigest()


def protocol_fingerprint() -> dict:
    return {
        "suite_sha256": digest(FIXTURE / "suite.json"),
        "references_sha256": digest(FIXTURE / "references.json"),
        "source_code_sha256": code_fingerprint(),
    }


def save(path: Path, content: str) -> None:
    path = path.resolve()
    if not path.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Generated files must stay under outputs/stable_acceptance")
    if path.exists():
        raise ValueError(f"Output already exists; use a new run directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser(
        "prepare", help="校验输入并生成独立参考、15题清单、45条未运行记录"
    )
    build.add_argument(
        "--version", required=True, help="冻结版本标签；修改代码后新建版本"
    )
    build.add_argument("--tier", choices=("live", "offline_replay"), default="live")
    build.add_argument("--output-dir", required=True, type=Path)
    score = sub.add_parser(
        "score", help="只评分有证据且独立复核的记录，不把completed当通过"
    )
    score.add_argument("--ledger", required=True, type=Path)
    score.add_argument("--output-dir", required=True, type=Path)
    capture = sub.add_parser("capture", help="只读导出指定主控会话最新轮次，不自动评分")
    capture.add_argument("--conversation-id", required=True)
    capture.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    suite, references = (
        read_json(FIXTURE / "suite.json"),
        read_json(FIXTURE / "references.json"),
    )
    validate_suite(ROOT, suite, references)
    output = asset_path(ROOT, str(args.output_dir))
    if args.command == "capture":
        from langgraph.checkpoint.sqlite import SqliteSaver

        checkpoint = ROOT / "data/master_agent_checkpoints.sqlite"
        with sqlite3.connect(
            checkpoint.as_uri() + "?mode=ro", uri=True, check_same_thread=False
        ) as connection:
            item = SqliteSaver(connection).get_tuple(
                {"configurable": {"thread_id": "master_" + args.conversation_id}}
            )
            if item is None:
                raise ValueError("Conversation checkpoint not found")
            state = item.checkpoint["channel_values"]
            call_ids = state.get("executed_call_ids", [])
            payload = {
                "conversation_id": args.conversation_id,
                "user_turn_id": state.get("user_turn_id"),
                "checkpoint_id": item.checkpoint["id"],
                "user_message": state.get("user_message"),
                "runtime_status": state.get("status"),
                "executed_call_ids": call_ids,
                "sub_agent_results": [
                    r
                    for r in state.get("sub_agent_results", [])
                    if r.get("call_id") in call_ids
                ],
                "final_draft": state.get("final_draft"),
            }
        save(output / "master_turn.json", json_text(payload))
        print(
            json_text(
                {
                    k: payload[k]
                    for k in (
                        "conversation_id",
                        "user_turn_id",
                        "runtime_status",
                        "executed_call_ids",
                    )
                }
            )
        )
        return
    if args.command == "prepare":
        expected = {c["id"]: frozen_expectation(ROOT, suite, c) for c in suite["cases"]}
        ledger = ledger_template(suite, args.version, args.tier)
        ledger["fingerprint"] = protocol_fingerprint()
        save(output / "frozen_database_expectations.json", json_text(expected))
        save(output / "ledger.json", json_text(ledger))
        lines = [
            "# 固定15题验收清单",
            "",
            f"版本：{args.version}；主层：{args.tier}",
            "",
            "参考答案仅供验收，不传给Agent。冻结计数只适用于冻结切片；真实API要另存输入独立核对。",
            "",
            "题数15；普通题10、边界题5；开发题10、保留题5；每题3次。",
            "",
            "|题号|类型|分组|主题|冻结切片候选数|",
            "|---|---|---|---|---|",
        ]
        for case in suite["cases"]:
            lines.append(
                f"|{case['id']}|{case['kind']}|{case['split']}|{case['title']}|"
                f"{expected[case['id']]['count']}|"
            )
        for case in suite["cases"]:
            lines.extend(
                [
                    "",
                    f"## {case['id']}：{case['title']}",
                    "",
                    case["question"],
                    "",
                    f"上传全文：{', '.join(case['pdfs']) or '不上传'}。",
                ]
            )
            if case.get("followup"):
                lines.extend(["", f"固定续问：{case['followup']}"])
        save(output / "questions.md", "\n".join(lines) + "\n")
        summary = score_ledger(ROOT, suite, references, ledger)
    else:
        ledger = read_json(asset_path(ROOT, str(args.ledger)))
        if ledger.get("fingerprint") != protocol_fingerprint():
            raise ValueError(
                "Code, suite or reference changed: start a new version/campaign"
            )
        summary = score_ledger(ROOT, suite, references, ledger)
    save(output / "summary.json", json_text(summary))
    normal = summary["normal"]
    boundary = summary["boundary_primary"]
    fault = summary["boundary_fault_simulation"]
    report = (
        f"# 稳定验收状态\n\n版本：{summary['version']}；"
        f"层：{summary['primary_tier']}\n\n"
        f"门禁：**{summary['gate']}**\n\n"
        f"普通题：已确认通过 {normal['passed']}/30，确认失败 {normal['failed']}，"
        f"未审核 {normal['unreviewed']}，未运行 {normal['not_run']}。\n\n"
        f"真实/回放边界题：通过 {boundary['passed']}/12；"
        f"故障模拟另计 {fault['passed']}/3。\n\n"
        "未运行不等于失败；已确认通过占计划次数的比例不等于当前系统准确率。\n\n"
        "尚未完成45条执行与复核时，不能宣布稳定通过；历史诊断不自动填入本次固定验收。\n\n"
        f"参考级别：{summary['reference_status']}。\n\n{summary['note']}\n"
    )
    save(output / "status.md", report)
    print(json_text({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
