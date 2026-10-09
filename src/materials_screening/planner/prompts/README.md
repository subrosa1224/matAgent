# Planner Prompts

## 文件与版本

- `planner_system_v1.txt` — 系统 Prompt（对应版本 `planner-v1`）
- `planner_examples_v1.json` — few-shot 示例（随系统 Prompt 一起组装）
- `planner_system_v2.txt` — 系统 Prompt v2（默认；v1 保留用于复现）
- `planner_examples_v2.json` — few-shot 示例 v2（默认）

v2 变更：数值区间按用户原样保留（不交换/修正大小）；光伏等应用目标与有毒/稀有集合由规则层识别为 ambiguity，不写入 draft；增加注入防护规则。

`PromptBuilder.build()` 返回 `PromptSpec`：

- `version` — 当前 Prompt 版本；
- `system_prompt` — 系统 Prompt 文本（基础文本 + 示例 JSON）；
- `sha256` — `system_prompt` 的 SHA-256（UTF-8），用于审计与复现。

## 版本规则

- 任何 Prompt 或 Schema 内容变更都必须升级版本（文件名与 `version` 同步，例如 v1 → v2）。
- 同一版本内容必须完全一致：相同输入应产生相同 `sha256`。
- 版本与 `Settings.planner_prompt_version` / `planner_schema_version` 保持一致。

## 修改指南

- 语言/抽取质量问题优先改 Prompt 并升级版本；确定性规则问题改 Resolver，不改 Prompt。
- 用户文本只作为数据传入（独立 user role），不得拼入系统 Prompt。
- 不保存原始 Prompt/response（默认 `PLANNER_STORE_RAW_IO=false`）。
