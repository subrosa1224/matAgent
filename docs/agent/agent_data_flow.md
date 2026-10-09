# Agent 数据流详解：三条 QA 从自然语言到最终回答

> 本文用三条真实运行记录，按顺序拆解一次 `agent ask` 提问从**自然语言**
> 变成**结构化请求**、再变成**最终回答**的完整数据流：
>
> 1. **概念问题**——不调用工具，直接回答（"什么是半导体？"）；
> 2. **筛选请求**——调用工具，走完整筛选工作流（"帮我找含 Cs、Pb、I 的
>    卤化物钙钛矿…"）；
> 3. **条件有误**——调用工具但被确定性校验拦截，如实告知失败
>    （"带隙大于 2 eV 且小于 1 eV"）。
>
> 文中所有 JSON 均为真实运行产物（DeepSeek + Materials Project，采集于
> 2026-08-11），只做了截断；候选 ID / 数值会随数据库更新变化，但**链路
> 结构不变**。

---

## 0. 阅读前：三层架构与关键概念

一次提问经过三个层次，严格单向依赖：

```text
CLI / Web UI
   │  user_message
   ▼
Agent 层（MaterialAgentRunner + LangGraph 图）——负责"决策与工具编排"
   │  run_screening_workflow 工具调用
   ▼
Workflow 层（WorkflowRunner + 确定性工作流）——负责"筛选与数据"
   planner（NL→结构化）→ 检索 → 过滤 → 排名 → 验证 → 导出
```

贯穿全文的关键概念：

| 概念                         | 说明                                                 | 出现位置    |
| ---------------------------- | ---------------------------------------------------- | ----------- |
| `transcript`（input_items）  | 完整多轮对话记录，每次模型调用全量重发（模型无状态） | agent 状态  |
| `pending_tool_calls`         | 模型决定调用的工具清单（本轮最多 4 个）              | agent 状态  |
| `evidence_id`                | 一次工具执行的证据编号，最终回答必须引用             | 工具结果    |
| `AgentFinalDraft`            | 模型最终回答的结构化 JSON（必须满足此 schema）       | 模型输出    |
| `AgentResult`                | 最终返回给用户的安全结果（不含原始参数/推理）        | runner 输出 |
| `ToolResultEnvelope`         | 工具结果的安全信封（喂回给模型的 JSON）              | 工具执行后  |
| `RunScreeningWorkflowOutput` | 筛选工作流工具的具体输出结构                         | 工具内      |

Agent 图固定 7 个节点：

```text
START → prepare_turn → call_agent_model ─┬─(有工具调用)→ execute_tools ─┐
                                          │                              ▼
                                          │                        call_agent_model（第 2 次）
                                          └─(直接回答)→ validate_final ──┬─ finalize_success
                                                                          └─ finalize_error
                                                              （取消时 → finalize_cancelled）
```

- 一轮最多 4 次模型调用；**只有第 1 次允许调用工具**，之后必须给出最终回答；
- 任何回答在展示给用户前都要过 `validate_final`（证据与安全校验）。

---

## 1. QA 示例一：概念问题，不调用工具直接回答

**用户输入：** `什么是半导体？为什么电脑芯片用半导体？`

### 1.1 prepare_turn：开始新一轮

节点校验 `conversation_id` 格式与输入大小后，把用户消息追加进 transcript，
并把本轮计数清零：

```jsonc
{
  "user_turn_id": "id_4",              // 本轮唯一编号
  "input_items": [                     // transcript：目前只有一条用户消息
    {"type": "message", "role": "user",
     "content": "什么是半导体？为什么电脑芯片用半导体？"}
  ],
  "status": "running",
  "model_call_count": 0, "tool_call_count": 0, "workflow_run_count": 0,
  "pending_tool_calls": [], "final_draft": null
}
```

### 1.2 call_agent_model（第 1 次）：模型自主决策

模型收到三样东西（`MaterialAgentRequest`）：

1. **系统提示词**（固定）："You are a restricted inorganic materials
   screening agent with exactly five tools: run_screening_workflow,
   get_workflow_status, ... 你的最终回答必须是单个 JSON 对象…"
2. **5 个工具定义**（`tool_definitions`）：run_screening_workflow /
   get_workflow_status / get_workflow_history / get_screening_result /
   compare_ranked_materials 的参数 JSON Schema；
3. **transcript**（上一节的 input_items）。

`allow_tool_calls=True`（第 1 次调用允许调工具），`tool_choice="auto"`——
**调不调工具完全由模型判断**。这是一个概念问题，模型判断"不需要任何
工具"，直接输出最终回答：

```jsonc
// 模型原始输出（AgentFinalDraft JSON，节选）
{"status":"completed",
 "answer":"**什么是半导体？**\n\n半导体是一种导电能力介于导体（如铜、铝）和
           绝缘体（如橡胶、玻璃）之间的材料。它的电阻率通常在 10⁻⁴ 到
           10⁸ Ω·cm 之间。最典型的半导体材料是硅（Si）和锗（Ge），此外还有
           砷化镓（GaAs）、碳化硅（SiC）等化合物半导体。\n\n半导体的关键
           特性是它的导电性可以通过多种方式精确调控：…",
 "active_workflow_thread_id": null,    // 没有工作流
 "referenced_material_ids": [],        // 没有引用任何材料
 "evidence_ids": [],                   // 没有证据
 "warnings": [],
 "follow_up_question": null}
```

模型没有产出 `function_call`，所以 `pending_tool_calls` 为空，路由直接
走到 `validate_final`——**一次模型调用即完成**。

### 1.3 validate_final：校验清单

`FinalValidator` 逐项检查（全部通过）：

| 检查项                          | 本案例                                                 |
| ------------------------------- | ------------------------------------------------------ |
| Schema 合法（AgentFinalDraft）  | ✓ 通过                                                 |
| 回答长度 ≤ 8000 字符            | ✓                                                      |
| 不含密钥模式（sk-…、api_key=…） | ✓                                                      |
| evidence_ids 必须来自本轮会话   | 空列表 ✓（无证据要求）                                 |
| 引用的材料 ID 必须在工具证据中  | 空列表 ✓                                               |
| 回答中的 mp-xxx 必须全部被引用  | 无 ✓                                                   |
| **任务事实必须有证据**          | 概念回答不含任务事实（无材料/带隙/筛选等标记）→ 免检 ✓ |
| 工作流状态一致性                | 无工作流 → 免检 ✓                                      |

> 关键规则：`completed` 状态的回答如果声称了"任务事实"（列材料、报带隙、
> 报筛选结果），就必须携带 evidence_ids；纯概念回答不需要证据。

### 1.4 finalize_success：构造最终输出

节点把 `final_draft["answer"]` 发布为 `final_response`，runner 汇总成
`AgentResult`（安全结果，不含原始参数/推理过程）：

```jsonc
{"conversation_id": "xxxx", "user_turn_id": "id_4",
 "status": "completed", "final_status": "completed",
 "response_text": "**什么是半导体？**\n\n半导体是一种导电能力介于…",
 "active_workflow_thread_id": null,
 "selected_tools": [],          // 本轮没有调用任何工具
 "tool_call_count": 0, "model_call_count": 1,
 "evidence_ids": [], "warnings": [], "error": null}
```

CLI 显示 `Status: completed`，无 `Tools:` 行，`Agent>` 后即为回答。

### 1.5 验证要点

- `Status: completed`、`Tools:` 行为空（或不存在）；
- 回答是连贯的中文讲解，不含"请到导出目录查看"类表述；
- 无 `thread` 生成（`data/workflow_runs/` 下没有新目录）。

---

## 2. QA 示例二：筛选请求，调用工具回答

**用户输入：** `帮我找含 Cs、Pb、I 的卤化物钙钛矿光伏材料，带隙 1.2 到 1.7 eV`

### 2.1 prepare_turn

同 1.1，transcript 追加用户消息。

### 2.2 call_agent_model（第 1 次）：模型决定调用工具

模型判断这是一个筛选请求，输出**函数调用**（而不是回答）：

```jsonc
// 模型原始输出（function_call，arguments 是 JSON 字符串）
{"call_id": "call_00_2Gn1GxU7JH3hawFVYFLA1153",
 "name": "run_screening_workflow",
 "arguments": "{\"query\": \"Find halide perovskite photovoltaic materials
               containing Cs, Pb, and I with band gap between 1.2 and
               1.7 eV\"}"}
```

> **注意**：模型把中文查询改写成了英文再传给工具。这是真实行为——查询
> 文本在到达 workflow 之前可能被模型翻译/改写，后续解析以改写后的文本为准。

节点把该调用存入 `pending_tool_calls`，路由到 `execute_tools`。

### 2.3 execute_tools：执行前的守卫

`ToolExecutor` 对调用做 10 步确定性检查（顺序）：

1. Registry 查工具（必须在 5 工具白名单内）→ ✓
2. 参数原始字节 ≤ 8 KB → ✓
3. JSON 可解析且为对象 → ✓
4. pydantic 校验（`RunScreeningWorkflowInput`，`extra="forbid"`）→ ✓
5. **Policy**：call_id 非空且不重复、每轮工具 ≤ 4 个、每轮工作流运行
   ≤ 1 次（`CREATE_WORKFLOW_RUN` 副作用限制）→ ✓
6. 幂等 ledger：`(user_turn_id, query_hash)` 未执行过 → 执行
7. 执行工具 → 内部调用 `WorkflowRunner.run(WorkflowInput(query=...))`

### 2.4 workflow 内部：自然语言如何变成结构化请求

`WorkflowRunner.run` 走固定 11 节点 DAG。关键在 `resolve_request` 节点：

**第 1 步：planner（真实 DeepSeek）解析出草稿。** 查询（已是英文）交给
`PlannerService.parse()`——模型按 `PlannerDraft` schema 输出字段草稿：

```jsonc
// PlannerDraft（节选）
{"status": "extracted",
 "required_elements": ["Cs", "I", "Pb"],
 "excluded_elements": [], "chemsys": null,
 "band_gap_min": 1.2, "band_gap_max": 1.7, "band_gap_unit": "eV",
 "is_metal": null, "is_stable": null, "limit": null,
 "ambiguities": [], "conflicts": [], "unsupported_requirements": [], …}
```

**第 2 步：确定性 resolver 校验并转成 ScreeningRequest。** `PlannerResolver`
不调用任何 LLM/数据库，只做：单位换算、元素合法性（pymatgen）、合理性
边界（带隙 0–20 eV、hull 0–2 eV/atom、密度 0–25 g/cm³）、冲突/歧义检测。
通过后产出**结构化请求**（就是 `screening_result.json` 里的 `request`）：

```jsonc
{"band_gap_ev": {"min": 1.2, "max": 1.7}, "chemsys": null,
 "crystal_system": null, "density_g_cm3": null,
 "energy_above_hull_ev_atom": null, "excluded_elements": [],
 "formula": null, "is_metal": null, "is_stable": null, "limit": 10,
 "required_elements": ["Cs", "I", "Pb"], "spacegroup_numbers": [],
 "target_band_gap_ev": null, "theoretical": null}
```

> 到这里，"帮我找含 Cs、Pb、I 的卤化物钙钛矿光伏材料，带隙 1.2 到
> 1.7 eV" 这句话已经变成了一组**字段**。应用意图词（"光伏"）被解析为
> 带隙目标，不产生硬约束。

**第 3 步：检索。** `retrieve_materials` 用该 request 调 Materials Project
summary API（有界分页，最多 2000 条；查询过宽会被 `is_broad_request`
提前拒绝）。本案例检索到 **3 条**（CsPbI3 等），写入 `retrieval.json`。

**第 4 步：14 步硬过滤。** `filter_materials` 按固定顺序逐步过滤，每步
记录前后数量（`filter_trace.json` 真实数据）：

```text
deprecated 3→3 | duplicate_id 3→3 | excluded_elements 3→3 |
required_elements 3→3 | chemsys 3→3 | formula 3→3 | is_metal 3→3 |
is_stable 3→3 | theoretical 3→3 | crystal_system 3→3 | spacegroup 3→3 |
band_gap 3→3 | energy_above_hull 3→3 | density 3→3
```

（MP 服务端已按元素/带隙过滤，本地过滤全部通过；宽查询时这里会逐条
淘汰并记录拒绝原因。）

**第 5 步：加权排名。** `rank_materials` 按固定权重打分并给出**分数分解**：

```text
总分 = 稳定性 0.45 + 带隙匹配 0.40 + 数据完整度 0.10 + 直接带隙 0.05
```

（`ranked.json` 真实数据，节选）：

```jsonc
{"rank": 1,
 "record": {"material_id": "mp-1069538", "formula_pretty": "CsPbI3",
            "band_gap_ev": 1.4788, "energy_above_hull_ev_atom": 0.025,
            "density_g_cm3": 4.844, "is_gap_direct": true, …},
 "total_score": 0.9…}
```

**第 6 步：验证与导出。** `validate_results` 重新校验（ID 唯一、分数单调、
provenance 完整等），`export_results` 写入 `exports/`（request.json /
result.json / candidates.csv / report.md），并登记 artifact 清单。

### 2.5 工具结果：envelope 喂回给模型

工具返回 `RunScreeningWorkflowOutput`，executor 包成 `ToolResultEnvelope`
（真实片段）：

```jsonc
{"status": "ok", "tool_name": "run_screening_workflow",
 "call_id": "call_00_2Gn1GxU7JH3hawFVYFLA1153",
 "evidence_id": "a1ef",                       // ← 本轮证据编号
 "output": {"status": "completed",
            "thread_id": "ff19615e-b917-439e-8438-2c75f2396a5a",
            "planner_status": "ready",
            "retrieved_count": 3, "filtered_count": 3, "returned_count": 3,
            "validation_passed": true,
            "evidence_id": "a1ef",
            "top_candidates": [{"rank": 1, "material_id": "mp-1069538",
                                "formula_pretty": "CsPbI3",
                                "band_gap_ev": 1.4788, …}, …]}}
```

该 envelope 作为 `function_call_output` 追加进 transcript；同时证据
`(evidence_id=a1ef, result_json=…)` 登记进本轮 ledger。

### 2.6 call_agent_model（第 2 次）：基于证据生成最终回答

`allow_tool_calls=False`（`tool_choice="none"`，模型**不能再调工具**），
transcript 现在包含：用户消息 + 函数调用 + 工具结果。模型输出最终 draft
（真实 JSON，节选）：

```jsonc
{"status": "completed",
 "answer": "筛选完成，找到 3 个符合条件的含 Cs、Pb、I 的卤化物钙钛矿
            光伏材料（带隙 1.2–1.7 eV）：\n\n1. **mp-1069538 —
            CsPbI3**（排名 1）\n   - 带隙：1.479 eV\n   - 能量高于
            凸包：0.025 eV/atom\n   - 密度：4.844 g/cm³\n   - 综合
            评分：0.9…\n\n2. …",
 "active_workflow_thread_id": "ff19615e-b917-439e-8438-2c75f2396a5a",
 "referenced_material_ids": ["mp-1069538", …],
 "evidence_ids": ["a1ef"],
 "warnings": [], "follow_up_question": null}
```

### 2.7 validate_final：证据与事实一致性校验

与 1.3 相同清单，本案例的关键几项：

| 检查项                                         | 本案例                                                                             |
| ---------------------------------------------- | ---------------------------------------------------------------------------------- |
| evidence_ids 属于本轮会话                      | `a1ef` 在 ledger 中 ✓                                                              |
| referenced_material_ids 必须出现在证据 JSON 里 | mp-1069538 出现在 top_candidates ✓                                                 |
| 回答里出现的 mp-xxx 必须全部被引用             | ✓                                                                                  |
| **工作流状态一致性**                           | draft 声称 completed，证据中 workflow status=completed 且 validation_passed=true ✓ |

> 校验器会**递归扫描证据 JSON** 收集其中的 material_id，回答引用的每个
> ID 都必须来自证据——模型"编造"一个不存在的 mp-xxx 会在这里被拦截，
> 整轮转为 FINAL_VALIDATION_FAILED。

### 2.8 finalize_success → AgentResult

```jsonc
{"status": "completed", "final_status": "completed",
 "response_text": "筛选完成，找到 3 个符合条件的含 Cs、Pb、I 的…",
 "active_workflow_thread_id": "ff19615e-…",
 "selected_tools": ["run_screening_workflow"],
 "tool_call_count": 1, "model_call_count": 2,
 "evidence_ids": ["a1ef"], "error": null}
```

runner 还会把该 workflow thread 与 conversation 建立所有权关联
（`store.link_workflow`），后续追问 `get_screening_result` 只能读自己
会话的 thread。

### 2.9 验证要点

- `Status: completed`，`Tools:` 行含 `run_screening_workflow`；
- 回答直接列出候选 ID/公式/带隙（来自工具摘要，不是模型编造）；
- `data/workflow_runs/<thread>/` 下有完整 artifacts（request /
  retrieval / filter_trace / ranked / screening_result / validation）与
  exports；`manifest.json` 记录了每个产物的 SHA-256；
- 回答中的材料 ID 与 `ranked.json` 的 top 一致；evidence_ids 与工具
  envelope 一致。

---

## 3. QA 示例三：条件有误，调用后失败并如实告知

**用户输入：** `帮我找带隙大于 2 eV 且小于 1 eV 的半导体材料`

### 3.1–3.3 前置环节

同 2.1–2.3：模型决定调用 `run_screening_workflow`，参数为原查询：

```jsonc
{"call_id": "call_00_J23w9A4yeqMyRiz17lnW2392",
 "name": "run_screening_workflow",
 "arguments": "{\"query\": \"帮我找带隙大于 2 eV 且小于 1 eV 的半导体材料\"}"}
```

### 3.4 workflow 内部：矛盾条件被确定性拦截

`resolve_request` 节点：

1. **planner** 解析出草稿：`band_gap_min=2.0, band_gap_max=1.0`（单位
   eV，都在合理性边界内，模型层面"看起来正常"）；
2. **resolver 检测到冲突**：`min(2.0) > max(1.0)` 触发
   `ConflictCode.BAND_GAP_MIN_ABOVE_MAX` → 状态 **INVALID**；
3. 路由到 `finalize_planner_stop`——**不访问数据库**。证据：该运行目录
   的 `manifest.json` 中 `artifacts` 为空 `{}`，没有 retrieval /
   filter / rank 任何产物：

```jsonc
{"artifacts": {}, "created_at": "2026-08-11T01:21:44Z",
 "metadata": {"input_mode": "query", "workflow_version": "workflow-v1"},
 "run_id": "76fcec78-…", "schema_version": "artifact-manifest-v1"}
```

> 设计要点：**非 READY 的计划（需澄清/无效/不支持）在触碰数据源之前就
> 终止**，既省成本也避免误导。

### 3.5 工具结果 envelope

```jsonc
{"status": "ok", "tool_name": "run_screening_workflow",
 "call_id": "call_00_J23w9A4yeqMyRiz17lnW2392",
 "evidence_id": "6aa7",
 "output": {"status": "invalid_request",          // ← 注意这里
            "thread_id": "76fcec78-…",
            "planner_status": "invalid",
            "retrieved_count": 0, "filtered_count": 0, "returned_count": 0,
            "validation_passed": null, "top_candidates": [], …}}
```

（`invalid_request` 是 workflow 对 planner INVALID 的终止状态之一；
其他终止状态还有 `no_results`、`needs_clarification` 等。）

### 3.6 模型基于失败结果如实回答

第 2 次模型调用（`allow_tool_calls=False`）收到失败 envelope 后，输出
**error 状态的 final draft**——不编造任何候选：

```jsonc
{"status": "error",
 "answer": "您的筛选条件存在逻辑矛盾：带隙大于 2 eV 且小于 1 eV 是不可能
            同时满足的（一个材料的带隙不可能既大于 2 eV 又小于 1 eV）。
            请重新提供合理的带隙范围条件，例如：\n- 带隙大于 2 eV（宽带隙
            半导体）\n- 带隙小于 1 eV（窄带隙半导体）\n- 带隙在某个区间内
            （如 1-2 eV）\n\n请明确您需要的带隙条件后…",
 "active_workflow_thread_id": null,
 "referenced_material_ids": [], "evidence_ids": [], …}
```

> 注意状态选择：失败时 draft 用 `status="error"` 而非 `completed`——这
> 满足校验器的"工作流状态一致性"规则（completed 声明要求证据中的 workflow
> 状态是 completed/no_results 且验证通过，这里明显不满足，所以必须如实
> 报 error）。

### 3.7 validate_final 与 finalize

- error 状态的 draft 不触发"任务事实必须有证据"与"工作流状态一致性"检查
  （这些只约束 completed 声明）；
- 校验通过 → `finalize_success` 发布；`AgentResult`：

```jsonc
{"status": "completed", "final_status": "error",   // 回合完成，但最终回答是 error 声明
 "response_text": "您的筛选条件存在逻辑矛盾：…",
 "selected_tools": ["run_screening_workflow"],
 "model_call_count": 2, "evidence_ids": [], "error": null}
```

CLI 显示 `Status: completed`、`Final status: error`；Web UI 对 error
声明会显示中文说明（见 UI 错误解释表）。

### 3.8 验证要点

- 回答**如实说明条件矛盾**，绝不假装筛选成功或编造候选；
- `data/workflow_runs/<thread>/` 下没有 retrieval/ranked 产物（未访问
  数据库）；
- 其他失败形态同理：条件不足 → needs_clarification 追问；不支持的任务
  （如"用 DFT 计算"）→ unsupported；网络/限流 → 工具层错误 envelope。

---

## 4. 输出结构速查

| 结构                         | 关键字段                                                                                                                     | 产生者                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `AgentFinalDraft`            | status / answer / active_workflow_thread_id / referenced_material_ids / evidence_ids / warnings / follow_up_question         | 模型（最终回答）            |
| `AgentResult`                | status / final_status / response_text / selected_tools / tool_call_count / model_call_count / evidence_ids / error           | runner（用户可见）          |
| `ToolResultEnvelope`         | status(ok/error) / tool_name / call_id / evidence_id / output / error                                                        | 工具执行器                  |
| `RunScreeningWorkflowOutput` | status / thread_id / planner_status / retrieved_count / filtered_count / returned_count / validation_passed / top_candidates | run_screening_workflow 工具 |
| `WorkflowOutput`             | status / thread_id / planner_status / request / counts / exports / warnings / error                                          | workflow runner             |

## 5. 如何验证一次 Agent 运行是否正确

按顺序检查六点（真实模式下）：

1. **状态**：`Status: completed`；有失败声明时 `Final status: error` 且
   回答如实说明原因；
2. **工具**：`Tools:` 行与预期一致（筛选→run_screening_workflow；追问
   结果→get_screening_result；比较→compare_ranked_materials）；
3. **证据**：回答末尾/内部的证据 ID 与工具执行一致
   （`agent show` / 调试日志可见）；
4. **材料 ID**：回答中的每个 mp-xxx 都出现在工具的 top_candidates 里；
5. **工作流一致性**：completed 声明对应的 thread 状态为
   completed/no_results 且 validation_passed=true；
6. **产物可追溯**：`data/workflow_runs/<thread>/` 的 manifest.json 校验
   通过（SHA-256），`workflow status --thread-id <id>` 可复查。

常用命令：`materials-screen agent ask/show/tools/inspect`、
`materials-screen workflow status/history/replay`。

## 6. 真实运行中的两个常见现象

1. **模型可能改写查询文本**（如把中文翻译成英文再传给工具）——工作流
   按改写后的文本解析，结果以工具输出为准；
2. **偶发 MODEL_ERROR**：DeepSeek 在 tools 与 json_schema 并用时，最终
   回答不总是严格 JSON；模型层会确定性提取并自动重试一次，仍失败则整轮
   报错（工作流结果不丢失，见 UI 的中文错误说明与 thread 保存位置）。
