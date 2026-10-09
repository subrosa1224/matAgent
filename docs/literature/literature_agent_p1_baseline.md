# LiteratureAgent P1 检索基线

> 状态：已完成，待用户确认进入 P2  
> 日期：2026-08-20  
> 上位约束：`literature_agent_execution_baseline.md`

## 1. 本阶段边界

P1 只解决一个问题：用户输入自然语言研究主题后，系统自动扩写检索词，并返回统一、
可解释、可重现的候选论文清单。本阶段不解析 PDF、不抽取实验矩阵、不生成跨论文结论，
也不新增人工金标。

## 2. 已交付能力

- 结构化查询扩写：保留原始主题，识别材料/化学式、同义词、性能词和工艺词，最多生成
  6 个检索式。扩写规则确定、可测试，不依赖某一篇论文。
- 统一检索：一个入口编排 OpenAlex 与 Semantic Scholar；任一来源失败时返回另一来源的
  有效结果，并明确记录降级原因。
- 统一结果集：优先按 DOI 去重；无 DOI 时按规范化题名与年份去重；融合摘要、作者、
  开放获取状态与来源记录，并进行确定性排序。
- 可重现快照：相同请求生成稳定 `query_id`，结果保存在
  `data/literature_queries/<query_id>/result.json`，再次调用优先读取快照。
- 正式入口：命令行 `materials-screen literature search <主题>` 与智能体工具
  `literature_search` 均已接入；普通用户无需构造 OpenAlex 或 S2 参数。
- 中文可解释字段：返回题名、年份、DOI、摘要/开放获取状态、来源和入选原因；JSON 保留
  完整程序契约。

## 3. 固定主题真实验收

| 固定主题 | query_id | top-20 | 人工核查 | DOI 重复 |
| --- | --- | ---: | --- | ---: |
| TiO2 photocatalysis hydrogen evolution | `lit-b19ad6155c60781108481267` | 20 | 前 8 篇中至少 7 篇直接相关 | 0 |
| perovskite solar cell efficiency | `lit-7a9bcc7aa4df856c778b525c` | 20 | 前 8 篇全部直接相关 | 0 |
| calcium phosphate bioceramic osteoinductivity | `lit-1eb1ee474fe31eb6094629e6` | 20 | 前 8 篇中至少 6 篇直接相关 | 0 |

三个主题均超过“top-20 至少 5 篇相关”的 P1 验收线。结果文件保留了原始扩写式、每个
来源的调用状态、记录数量、来源指纹、警告和创建时间，可供复核。

## 4. 自动验证证据

- LiteratureAgent P1 相关单元/集成测试：7 passed。
- Ruff：通过。
- LiteratureAgent 目标目录 mypy：通过，20 个源文件无错误。
- 离线测试覆盖：中英文有界扩写、DOI 跨源合并、字段融合、确定性排序、快照复用、单源
  失败降级、正式工具注册与集成结果读取。
- 三个真实结果的 DOI 重复数均为 0；重复执行相同 TiO2 请求复用了同一 `query_id` 快照。

## 5. 已知限制与处理方式

本轮真实验收中，Semantic Scholar 返回 `PROVIDER_RATE_LIMIT`，OpenAlex 正常返回结果。
系统按设计将 S2 标记为 `degraded`，没有丢弃 OpenAlex 结果，也没有将单源结果伪装为双源
成功。跨源合并逻辑已由离线测试覆盖；配置更高限额的 S2 API key 后无需修改业务代码。

“开放获取”仅表示元数据来源报告的访问状态，不等于系统已经取得 PDF。PDF 获取、解析和
正文证据抽取属于后续阶段，P1 不作完成声明。

## 6. 阶段结论

P1 的代码交付、自动测试和三个固定主题人工验收均已满足基线要求。下一步只有在用户确认
后进入 P2：通用单篇论文档案抽取；不会在确认前扩展实验矩阵或综合报告能力。
