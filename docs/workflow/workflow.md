# 材料筛选系统第三阶段技术细节文档

> 项目：`materials-screening-core`  
> 阶段：第三阶段——LangGraph 固定工作流编排层  
> 文档版本：v1.0  
> 日期：2026-08-06  
> 前置条件：第一阶段确定性筛选内核和第二阶段 DeepSeek-V4-Flash Planner 均已完成并通过测试。

---

## 1. 阶段定位

第三阶段不立即创建具有独立人格和自由规划能力的多智能体，而是先把前两阶段能力编排为一个状态明确、可持久化、可恢复、可观察、可测试的固定工作流。

```text
用户输入 query 或 request
        ↓
initialize_run
        ↓
resolve_request
        ↓
条件路由
├─ READY
│   ↓
│ retrieve_materials
│   ↓
│ filter_materials
│   ├─ 零候选 → finalize_no_results
│   └─ 有候选
│       ↓
│     rank_materials
│       ↓
│     validate_results
│       ├─ 失败 → finalize_failure
│       └─ 通过 → export_results → finalize_success
├─ NEEDS_CLARIFICATION → finalize_planner_stop
├─ INVALID             → finalize_planner_stop
├─ UNSUPPORTED         → finalize_planner_stop
└─ 预期业务错误         → finalize_failure
```

本阶段的节点只是受控工作流步骤。真正的角色化多智能体放在第四阶段。

---

## 2. 核心原则

### 2.1 LangGraph 只负责编排

节点复用已有服务：

```text
PlannerService
MaterialsRepository
FilterService
RankingService
ValidationService
ExportService
```

不得在节点中重新实现材料过滤、排序或科学验证规则。

### 2.2 图状态保持轻量

LangGraph checkpointer 会在 super-step 边界保存状态。如果把大量 `MaterialRecord`、晶体结构或 CIF 内容放进状态，每一步都会重复持久化，导致 SQLite 膨胀和恢复变慢。

采用：

```text
轻量 WorkflowState
+
RunArtifactStore 中间文件
```

状态只保存 `ArtifactRef`，完整记录写入：

```text
data/workflow_runs/<run_id>/artifacts/
```

### 2.3 节点必须可重执行

节点失败、恢复或 replay 时可能从头执行。因此：

- 文件名必须确定；
- 写入必须原子化；
- 相同内容重复写入直接复用；
- 同名但 hash 不同必须失败；
- export 不得每次生成新的时间戳目录；
- 不得重复追加同一事件；
- 外部副作用必须有幂等键。

### 2.4 单一重试所有者

第一、二阶段的 DeepSeek Provider 与 Materials Repository 已经有有限重试。第三阶段初始版本不在 LangGraph 节点上叠加 `RetryPolicy`，避免调用次数成倍增加。

若未来改为 LangGraph 重试，必须先把底层服务尝试次数设为 1，并精确限定 `retry_on`。

### 2.5 不允许自由循环

第三阶段图没有业务循环。运行时 `recursion_limit` 推荐为 32。若触发 recursion limit，应视为图定义、恢复或路由错误，而不是正常业务状态。

---

## 3. 本阶段完成后的命令

自然语言工作流：

```bash
uv run materials-screen workflow run \
  --query "寻找不含 Pb、Cd、Hg，带隙 1.2 到 2.0 eV 的非金属材料" \
  --llm-provider deepseek \
  --materials-repository materials-project
```

结构化工作流：

```bash
uv run materials-screen workflow run \
  --request examples/semiconductor_request.json \
  --materials-repository materials-project
```

离线演示：

```bash
uv run materials-screen workflow run \
  --query "不含 Pb，带隙 1 到 2 eV" \
  --llm-provider mock \
  --materials-repository mock \
  --planner-fixture tests/fixtures/planner_drafts.json \
  --materials-fixture tests/fixtures/mp_documents.json \
  --stream
```

状态和历史：

```bash
uv run materials-screen workflow status <thread_id>
uv run materials-screen workflow history <thread_id>
```

受控 replay：

```bash
uv run materials-screen workflow replay \
  --thread-id <thread_id> \
  --checkpoint-id <checkpoint_id> \
  --confirm-remote-calls
```

---

## 4. 本阶段不实现

- 多智能体人格与自由规划；
- Agent-to-Agent 对话；
- 动态创建智能体；
- RAG；
- Reporter LLM；
- 多数据库；
- 属性预测；
- DFT 作业；
- FastAPI、Web UI 或 LangGraph Server；
- PostgreSQL、Redis；
- 子图；
- 正式 Human-in-the-loop；
- 并行材料分片；
- 自动放宽零结果条件。

---

## 5. 开发前基线保护

```bash
git status
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not real_api and not real_llm and not real_deepseek"
```

建议：

```bash
git tag stage2-complete
git switch -c stage3-langgraph-workflow
```

创建 `docs/STAGE2_BASELINE.md`，记录 Python、依赖、测试数量、覆盖率、CLI、真实 smoke 状态和已知限制。

---

## 6. 依赖与当前 API

推荐：

```toml
[project.optional-dependencies]
workflow = [
    "langgraph>=1.2,<2",
    "langgraph-checkpoint-sqlite",
]
```

安装：

```bash
uv sync --extra deepseek --extra workflow
```

实现前必须检查当前本地包：

```bash
uv run python scripts/inspect_langgraph.py
```

至少核对：

- `StateGraph`；
- `START`、`END`；
- `add_node`；
- `add_conditional_edges`；
- `compile`；
- `Runtime` 与 `context_schema`；
- `invoke(..., context=...)`；
- `stream(..., stream_mode="updates", version="v2")`；
- `get_state()`；
- `get_state_history()`；
- `update_state()`；
- `InMemorySaver`；
- `SqliteSaver`。

不得根据旧博客使用废弃接口。

---

## 7. 推荐目录

```text
src/materials_screening/
├── workflow/
│   ├── __init__.py
│   ├── state.py
│   ├── input_output.py
│   ├── context.py
│   ├── events.py
│   ├── errors.py
│   ├── artifact_store.py
│   ├── nodes.py
│   ├── routing.py
│   ├── graph_builder.py
│   ├── checkpointer.py
│   ├── runner.py
│   ├── history.py
│   ├── visualization.py
│   └── version.py
├── planner/
├── llm/
├── repositories/
├── services/
└── cli.py
```

测试：

```text
tests/
├── unit/workflow/
│   ├── test_workflow_state.py
│   ├── test_artifact_store.py
│   ├── test_workflow_nodes.py
│   ├── test_workflow_routing.py
│   ├── test_graph_builder.py
│   ├── test_checkpointer_factory.py
│   └── test_workflow_runner.py
├── integration/
│   ├── test_workflow_query_success.py
│   ├── test_workflow_request_success.py
│   ├── test_workflow_planner_stops.py
│   ├── test_workflow_no_results.py
│   ├── test_workflow_failures.py
│   ├── test_workflow_persistence.py
│   ├── test_workflow_replay.py
│   ├── test_workflow_streaming.py
│   └── test_direct_workflow_equivalence.py
└── fixtures/
```

脚本：

```text
scripts/inspect_langgraph.py
scripts/draw_workflow.py
scripts/inspect_checkpoints.py
```

---

## 8. 状态枚举

```python
class WorkflowStatus(str, Enum):
    INITIALIZING = "initializing"
    RESOLVING_REQUEST = "resolving_request"
    READY_FOR_RETRIEVAL = "ready_for_retrieval"
    RETRIEVING = "retrieving"
    FILTERING = "filtering"
    RANKING = "ranking"
    VALIDATING = "validating"
    EXPORTING = "exporting"

    NEEDS_CLARIFICATION = "needs_clarification"
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_REQUEST = "unsupported_request"
    NO_RESULTS = "no_results"
    COMPLETED = "completed"
    FAILED = "failed"
```

终态：

```python
TERMINAL_WORKFLOW_STATUSES = {
    WorkflowStatus.NEEDS_CLARIFICATION,
    WorkflowStatus.INVALID_REQUEST,
    WorkflowStatus.UNSUPPORTED_REQUEST,
    WorkflowStatus.NO_RESULTS,
    WorkflowStatus.COMPLETED,
    WorkflowStatus.FAILED,
}
```

---

## 9. 输入与输出 Schema

### 9.1 WorkflowInput

```python
class WorkflowInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str | None = None
    request: dict | None = None
    output_root: str = "data/workflow_runs"
    export_cif: bool = True

    @model_validator(mode="after")
    def validate_mode(self) -> "WorkflowInput":
        if (self.query is None) == (self.request is None):
            raise ValueError("Exactly one of query or request must be provided")
        return self
```

### 9.2 WorkflowOutput

```python
class WorkflowOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    thread_id: str
    status: WorkflowStatus
    planner_status: str | None
    request: dict | None
    retrieved_count: int
    filtered_count: int
    returned_count: int
    validation_passed: bool | None
    exports: tuple[str, ...]
    warnings: tuple[str, ...]
    error: dict | None
    clarification_question: str | None
```

应显式配置 graph input/output schema，避免内部 artifact 和错误细节被直接返回。

---

## 10. ArtifactRef

```python
class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    relative_path: str
    sha256: str
    media_type: str
    size_bytes: int
    item_count: int | None = None
    schema_name: str | None = None
    schema_version: str | None = None
```

状态中只保存：

```text
retrieval_ref
filtered_ref
filter_trace_ref
ranked_ref
validation_ref
screening_result_ref
export_manifest_ref
```

---

## 11. 事件与 reducer

```python
class WorkflowEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    run_id: str
    node: str
    event_type: str
    created_at: str
    status: str
    message: str
    metrics: dict[str, int | float | str | bool | None]
```

```python
from operator import add
from typing import Annotated


class WorkflowState(TypedDict, total=False):
    events: Annotated[list[dict], add]
```

节点只返回本次新增事件，不返回完整旧列表。

---

## 12. WorkflowState

所有字段必须 JSON 可序列化：

```python
class WorkflowState(TypedDict, total=False):
    run_id: str
    workflow_version: str
    input_mode: str

    user_query: str | None
    raw_request: dict | None
    output_root: str
    export_cif: bool

    status: str
    current_node: str
    started_at: str
    finished_at: str | None

    planner_result: dict | None
    planner_status: str | None
    clarification_question: str | None
    screening_request: dict | None

    retrieval_ref: dict | None
    filtered_ref: dict | None
    filter_trace_ref: dict | None
    ranked_ref: dict | None
    validation_ref: dict | None
    screening_result_ref: dict | None
    export_manifest_ref: dict | None

    retrieved_count: int
    filtered_count: int
    returned_count: int
    validation_passed: bool | None

    exports: list[str]
    warnings: list[str]
    error: dict | None
    events: Annotated[list[dict], add]
```

禁止放入状态：

- API Key；
- DeepSeek 原始响应和 reasoning；
- Materials Project 大型原始响应；
- service 实例；
- sqlite connection；
- file handles；
- exception/traceback；
- `Path`、`datetime` 等未转换对象；
- 完整结构列表。

---

## 13. WorkflowContext

```python
@dataclass(frozen=True)
class WorkflowContext:
    planner_service: PlannerService
    materials_repository: MaterialsRepository
    filter_service: FilterService
    ranking_service: RankingService
    validation_service: ValidationService
    export_service: ExportService
    artifact_store: RunArtifactStore
    clock: Clock
    id_generator: IdGenerator
```

构图：

```python
builder = StateGraph(
    WorkflowState,
    context_schema=WorkflowContext,
    input_schema=WorkflowGraphInput,
    output_schema=WorkflowOutput,
)
```

节点：

```python
def retrieve_materials_node(
    state: WorkflowState,
    runtime: Runtime[WorkflowContext],
) -> dict:
    repository = runtime.context.materials_repository
    ...
```

调用：

```python
graph.invoke(
    graph_input,
    config=config,
    context=workflow_context,
)
```

Context 不进入 checkpoint。

---

## 14. RunArtifactStore

### 14.1 接口

```python
class RunArtifactStore(Protocol):
    def initialize_run(self, *, run_id: str, metadata: dict) -> None: ...

    def put_json(
        self,
        *,
        run_id: str,
        name: str,
        value: object,
        schema_name: str | None = None,
        schema_version: str | None = None,
        item_count: int | None = None,
    ) -> ArtifactRef: ...

    def get_json(self, ref: ArtifactRef) -> object: ...
    def exists(self, ref: ArtifactRef) -> bool: ...
    def verify(self, ref: ArtifactRef) -> bool: ...
```

### 14.2 布局

```text
data/workflow_runs/<run_id>/
├── manifest.json
├── artifacts/
│   ├── retrieval.json
│   ├── filtered.json
│   ├── filter_trace.json
│   ├── ranked.json
│   ├── validation.json
│   └── screening_result.json
└── exports/
```

### 14.3 幂等写入

1. 将对象转换为 JSON-compatible 数据；
2. 使用稳定序列化；
3. 计算 SHA-256；
4. 写临时文件；
5. flush 并原子 replace；
6. 已存在且 hash 相同则复用；
7. 已存在但 hash 不同则 `ArtifactConflictError`。

路径必须经过白名单和 root containment 检查。

读取前必须 `verify(ref)`，hash 不一致时停止工作流。

---

## 15. run_id 与 thread_id

推荐：

```text
run_id = UUIDv7 或 UUID4
thread_id = run_id
```

规则：

- 每次新运行只生成一次；
- initialize 后不再生成；
- `thread_id` 通过 config 传递；
- `run_id` 保存在 state；
- 长度低于 255；
- replay 使用同一 thread 与 checkpoint；
- 新的独立重跑使用新 thread。

```python
config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": 32,
    "tags": ["materials-screening", "workflow-v1"],
    "metadata": {
        "run_id": run_id,
        "workflow_version": "workflow-v1",
    },
}
```

`recursion_limit` 不放进 `configurable`。

---

## 16. 节点清单

```text
initialize_run
resolve_request
retrieve_materials
filter_materials
rank_materials
validate_results
export_results
finalize_success
finalize_no_results
finalize_planner_stop
finalize_failure
```

每个节点：

- 单一职责；
- 返回 partial state；
- 写 `current_node` 和 status；
- 添加事件；
- 不返回未修改字段；
- 不吞未知编程错误；
- 只把预期领域异常转换为安全错误；
- 读取 artifact 时校验 hash。

---

## 17. initialize_run

职责：

- 验证内部 graph input；
- 初始化计数和状态；
- 创建 run 目录和 manifest；
- 不调用 DeepSeek 或 MP。

`run_id` 应由 Runner 在 invoke 前注入，节点不得自行生成新的 ID。

---

## 18. resolve_request

### query 模式

调用：

```python
planner_result = planner_service.parse(user_query)
```

映射：

```text
READY               → READY_FOR_RETRIEVAL
NEEDS_CLARIFICATION → NEEDS_CLARIFICATION
INVALID             → INVALID_REQUEST
UNSUPPORTED         → UNSUPPORTED_REQUEST
```

### request 模式

```python
request = ScreeningRequest.model_validate(raw_request)
```

校验失败转为 `INVALID_REQUEST`。

DeepSeek 配置、认证或服务错误转为 `FAILED`。未知编程错误继续抛出。

---

## 19. retrieve_materials

1. 从 state 重建 `ScreeningRequest`；
2. 调用 Repository；
3. 将 `RetrievalResult` 写入 artifact；
4. state 只保存 ArtifactRef 和 count。

不得：

- 提前 limit；
- 重写过滤；
- 把 records 放进 state；
- 生成报告。

---

## 20. filter_materials

1. 验证并读取 retrieval artifact；
2. 调用 FilterService；
3. 写 `filtered.json` 和 `filter_trace.json`；
4. 更新 filtered_count；
5. 零候选正常路由，不视为异常。

---

## 21. rank_materials

1. 读取 filtered artifact；
2. 对全部候选调用 RankingService；
3. 再按 request.limit 截断；
4. 写 ranked artifact；
5. 更新 returned_count。

不得在 Repository 或 Filter 提前 limit。

---

## 22. validate_results

1. 读取 request、retrieval、trace 和 ranked artifacts；
2. 构造初步 `ScreeningResult`；
3. 调用 ValidationService；
4. 写 validation 和完整 result artifact；
5. 更新 `validation_passed`。

若验证失败，必须路由 `finalize_failure`，不得删除不合格候选后伪装成功。

---

## 23. export_results

Export 是副作用节点，必须幂等。

优先新增 `WorkflowExportAdapter`，不要破坏第一阶段 ExportService 旧接口。

固定目录：

```text
data/workflow_runs/<run_id>/exports/
```

已有 manifest 时：

- request/result hash 相同：复用；
- hash 不同：冲突失败；
- 文件缺失或损坏：失败。

不允许 replay 时创建新的时间戳目录。

---

## 24. Finalize 节点

### finalize_success

要求：

- validation_passed=True；
- result 和 export manifest 存在；
- 至少 JSON 和 Markdown 存在；
- 设置 COMPLETED 和 finished_at。

### finalize_no_results

生成零结果摘要：

- 原始条件；
- retrieved_count；
- filter trace；
- 哪一步归零；
- 仅提出可选放宽建议，不自动执行。

### finalize_planner_stop

不调用 MP，不创建正常 exports。

### finalize_failure

保存安全错误摘要，不保存 Key、HTTP body 或 traceback。

---

## 25. 错误模型

```python
class WorkflowErrorData(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    node: str
    message: str
    retryable: bool
    exception_type: str
    occurred_at: str
```

错误码：

```text
WORKFLOW_INPUT_INVALID
PLANNER_FAILED
REQUEST_INVALID
RETRIEVAL_FAILED
ARTIFACT_WRITE_FAILED
ARTIFACT_READ_FAILED
ARTIFACT_INTEGRITY_FAILED
FILTER_FAILED
RANKING_FAILED
VALIDATION_FAILED
EXPORT_FAILED
CHECKPOINT_FAILED
WORKFLOW_INVARIANT_FAILED
WORKFLOW_RECURSION_LIMIT
UNEXPECTED_WORKFLOW_ERROR
```

---

## 26. 路由函数

路由函数必须纯函数、无 I/O、无状态修改，并返回 `Literal`。

```python
def route_after_request(
    state: WorkflowState,
) -> Literal[
    "retrieve_materials",
    "finalize_planner_stop",
    "finalize_failure",
]:
    status = WorkflowStatus(state["status"])

    if status is WorkflowStatus.READY_FOR_RETRIEVAL:
        return "retrieve_materials"

    if status in {
        WorkflowStatus.NEEDS_CLARIFICATION,
        WorkflowStatus.INVALID_REQUEST,
        WorkflowStatus.UNSUPPORTED_REQUEST,
    }:
        return "finalize_planner_stop"

    if status is WorkflowStatus.FAILED:
        return "finalize_failure"

    raise WorkflowInvariantError(...)
```

每个分支必须有测试。

---

## 27. 图定义

```python
from langgraph.graph import END, START, StateGraph


def create_workflow_builder() -> StateGraph:
    builder = StateGraph(
        WorkflowState,
        context_schema=WorkflowContext,
        input_schema=WorkflowGraphInput,
        output_schema=WorkflowOutput,
    )

    builder.add_node("initialize_run", initialize_run_node)
    builder.add_node("resolve_request", resolve_request_node)
    builder.add_node("retrieve_materials", retrieve_materials_node)
    builder.add_node("filter_materials", filter_materials_node)
    builder.add_node("rank_materials", rank_materials_node)
    builder.add_node("validate_results", validate_results_node)
    builder.add_node("export_results", export_results_node)
    builder.add_node("finalize_success", finalize_success_node)
    builder.add_node("finalize_no_results", finalize_no_results_node)
    builder.add_node("finalize_planner_stop", finalize_planner_stop_node)
    builder.add_node("finalize_failure", finalize_failure_node)

    builder.add_edge(START, "initialize_run")
    builder.add_edge("initialize_run", "resolve_request")

    builder.add_conditional_edges(
        "resolve_request",
        route_after_request,
        {
            "retrieve_materials": "retrieve_materials",
            "finalize_planner_stop": "finalize_planner_stop",
            "finalize_failure": "finalize_failure",
        },
    )

    builder.add_conditional_edges(
        "retrieve_materials",
        route_after_retrieval,
        {
            "filter_materials": "filter_materials",
            "finalize_failure": "finalize_failure",
        },
    )

    builder.add_conditional_edges(
        "filter_materials",
        route_after_filter,
        {
            "rank_materials": "rank_materials",
            "finalize_no_results": "finalize_no_results",
            "finalize_failure": "finalize_failure",
        },
    )

    builder.add_conditional_edges(
        "rank_materials",
        route_after_ranking,
        {
            "validate_results": "validate_results",
            "finalize_failure": "finalize_failure",
        },
    )

    builder.add_conditional_edges(
        "validate_results",
        route_after_validation,
        {
            "export_results": "export_results",
            "finalize_failure": "finalize_failure",
        },
    )

    builder.add_conditional_edges(
        "export_results",
        route_after_export,
        {
            "finalize_success": "finalize_success",
            "finalize_failure": "finalize_failure",
        },
    )

    builder.add_edge("finalize_success", END)
    builder.add_edge("finalize_no_results", END)
    builder.add_edge("finalize_planner_stop", END)
    builder.add_edge("finalize_failure", END)

    return builder
```

---

## 28. Checkpointer

### 测试

```python
from langgraph.checkpoint.memory import InMemorySaver
```

每个测试创建新的 saver 和 compiled graph。

### 本地开发

使用 `langgraph-checkpoint-sqlite` 的 `SqliteSaver`：

```text
data/workflow_checkpoints.sqlite
```

要求：

- connection 生命周期覆盖 invoke/stream；
- 不在每个节点创建 saver；
- connection 不放入 state；
- Runner/handle 负责关闭；
- 测试使用 `tmp_path`；
- SQLite 文件加入 `.gitignore`。

建议工厂返回具有 `close()` 或 context manager 的 handle，避免 saver 被过早关闭。

---

## 29. 编译图

```python
def compile_workflow(*, checkpointer: object):
    builder = create_workflow_builder()
    builder.validate()
    return builder.compile(
        checkpointer=checkpointer,
        name="materials-screening-workflow-v1",
    )
```

编译时不创建 DeepSeek 或 MP 客户端；依赖由 Context 注入。

---

## 30. WorkflowRunner

```python
class WorkflowRunner:
    def run(
        self,
        workflow_input: WorkflowInput,
        *,
        stream: bool = False,
    ) -> WorkflowOutput: ...

    def get_state(self, thread_id: str) -> WorkflowStateView: ...

    def get_history(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> tuple[WorkflowCheckpointView, ...]: ...

    def replay(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
    ) -> WorkflowOutput: ...
```

Runner 负责：

- 生成 run/thread ID；
- 构造 input/config/context；
- invoke/stream；
- 状态和历史转换；
- replay；
- 捕获 `GraphRecursionError` 与 checkpoint I/O；
- 不吞测试环境中的未知异常。

---

## 31. Streaming

使用：

```python
graph.stream(
    graph_input,
    config=config,
    context=context,
    stream_mode="updates",
    version="v2",
)
```

CLI 只显示节点级安全摘要：

```text
[initialize_run] resolving_request
[resolve_request] ready_for_retrieval
[retrieve_materials] retrieved=186
[filter_materials] filtered=23
[rank_materials] returned=10
[validate_results] passed
[export_results] files=14
[finalize_success] completed
```

不默认使用 `values`，因为它会暴露完整内部状态。

---

## 32. 状态与历史

状态：

```python
snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
```

历史：

```python
history = list(graph.get_state_history(config, limit=limit))
```

对外只返回安全 view，不直接打印完整 `StateSnapshot.values`。

Checkpoint 表格字段：

```text
step
checkpoint_id
source
current_node
status
next_nodes
created_at
```

---

## 33. Replay

流程：

1. 从历史查指定 checkpoint；
2. 验证属于 thread；
3. 验证 artifact 完整性；
4. 使用 checkpoint config 调用 graph；
5. checkpoint 之前的节点跳过，之后重新执行；
6. 幂等 artifact 和 export 防止重复副作用。

概念示例：

```python
result = graph.invoke(
    None,
    config=checkpoint.config,
    context=context,
)
```

具体签名以当前安装版本为准并由集成测试确认。

Replay 可能重新触发 checkpoint 后的 DeepSeek 或 MP 调用，因此 CLI 必须要求 `--confirm-remote-calls`。

---

## 34. update_state 边界

第三阶段不向普通用户开放任意 state 修改，否则可能绕过 Validator。

`update_state()` 只作为未来 Human-in-the-loop 的受控接口预留，不实现诸如直接设置：

```json
{"validation_passed": true}
```

---

## 35. 图可视化

```bash
uv run python scripts/draw_workflow.py
```

至少生成：

```text
docs/workflow_v1.mmd
```

PNG 可选，不强制安装外部渲染器。

---

## 36. 旧入口兼容

保留直接模式：

```bash
materials-screen parse
materials-screen screen --query
materials-screen screen --request
```

新增 `workflow run`，不要静默改变旧命令行为。

同一结构化请求和 fixture 下，Direct 与 Workflow 必须得到相同：

- 候选 ID；
- 排名；
- 分数；
- 验证结果；
- 核心 CSV 数据。

允许 run metadata、事件和目录不同。

---

## 37. 异常策略

节点只捕获预期领域异常：

```python
except (
    LLMError,
    RepositoryError,
    ArtifactStoreError,
    ValidationFailedError,
    ExportError,
) as exc:
    return failure_update(...)
```

禁止：

```python
except Exception:
    return {"status": "failed"}
```

未知编程错误应让 invoke 抛出，避免 bug 被伪装成普通失败。

---

## 38. 配置

`.env.example` 增加：

```dotenv
WORKFLOW_ENABLED=true
WORKFLOW_VERSION=workflow-v1
WORKFLOW_CHECKPOINTER_BACKEND=sqlite
WORKFLOW_CHECKPOINT_DB=data/workflow_checkpoints.sqlite
WORKFLOW_RUN_ROOT=data/workflow_runs
WORKFLOW_RECURSION_LIMIT=32
WORKFLOW_HISTORY_LIMIT=50
WORKFLOW_STREAM_MODE=updates
WORKFLOW_STORE_FULL_STATE_LOGS=false
WORKFLOW_ALLOW_REPLAY=true
```

校验：

- recursion 8–256；
- history 1–500；
- stream mode 本阶段只允许 updates；
- SQLite 和 run root 必须在允许的数据目录；
- full state logs 默认 false。

---

## 39. CLI

```text
materials-screen workflow run
materials-screen workflow status
materials-screen workflow history
materials-screen workflow replay
materials-screen workflow draw
materials-screen workflow inspect
```

`workflow run`：

```text
--query / --request 严格二选一
--llm-provider mock|deepseek
--materials-repository mock|materials-project
--planner-fixture
--materials-fixture
--output
--stream
--no-cif
--thread-id（高级）
```

用户指定已存在 thread-id 且不是 replay/resume 时必须拒绝。

---

## 40. 测试矩阵

### State

- JSON 可序列化；
- reducer；
- terminal status；
- input/output schema；
- 不允许 Secret、Path、exception object。

### ArtifactStore

- 原子写；
- hash；
- 相同内容幂等；
- 不同内容冲突；
- 路径穿越；
- 损坏检测；
- tmp 清理。

### Nodes

每个节点测试：

- 正常；
- 缺前置字段；
- artifact 缺失；
- hash 失败；
- 预期 service 错误；
- event/status；
- state 中无大型对象。

### Routing

每个状态对应唯一边。

### Persistence

1. tmp SQLite 运行；
2. 关闭 Runner；
3. 新建 Runner；
4. `get_state`；
5. `get_state_history`；
6. 状态仍存在。

### Replay

从 filter 后 checkpoint replay：

- retrieve 不重新运行；
- rank/validate/export 重新执行；
- artifact hash 一致；
- export 不新建重复目录；
- 结果相同。

### Direct 等价

建立 `test_direct_workflow_equivalence.py`。

---

## 41. 端到端场景

至少覆盖：

1. query + Mock DeepSeek + Mock MP 成功；
2. request + Mock MP 成功；
3. NEEDS_CLARIFICATION；
4. INVALID；
5. UNSUPPORTED；
6. DeepSeek 失败；
7. MP 失败；
8. 零候选；
9. Validator 失败；
10. Export 失败；
11. SQLite 重启恢复；
12. replay；
13. streaming；
14. artifact 篡改；
15. thread ID 冲突；
16. direct/workflow 等价；
17. recursion limit 防护。

---

## 42. 质量门

```bash
uv sync --extra deepseek --extra workflow
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest \
  -m "not real_api and not real_llm and not real_deepseek" \
  --cov=materials_screening \
  --cov-report=term-missing
```

目标：

```text
第一阶段测试全部通过
第二阶段测试全部通过
第三阶段测试全部通过
总覆盖率 >= 90%
workflow/state/routing/artifact_store >= 95%
```

---

## 43. 实施里程碑

### S3-M0 基线保护

Stage 2 tag、旧测试、CLI 快照、分支。

### S3-M1 LangGraph 依赖与接口核对

依赖、inspect 脚本、兼容文档、最小 compile 测试。

### S3-M2 State、Context 和错误模型

input/output、state、event、ArtifactRef、context。

### S3-M3 ArtifactStore

幂等文件存储、hash、manifest、完整性测试。

### S3-M4 Nodes 与 Routing

所有节点和条件路由，独立测试。

### S3-M5 GraphBuilder 与 Checkpointer

StateGraph、InMemorySaver、SqliteSaver、持久化测试。

### S3-M6 Runner、Streaming 与 CLI

run、status、history、draw、streaming、旧入口兼容。

### S3-M7 Replay 与恢复

checkpoint 查找、replay、幂等导出、重启恢复。

### S3-M8 完整评测和最终审查

端到端矩阵、direct/workflow 等价、真实小规模 smoke、文档。

---

## 44. 完成定义

- [ ] 使用当前 `StateGraph`、`START`、`END` 和条件边；
- [ ] 显式 input/output schema；
- [ ] 依赖通过 Runtime context 注入；
- [ ] SQLite checkpointer；
- [ ] 每次 invoke 传 thread_id；
- [ ] state 中没有大型材料对象；
- [ ] 中间结果通过 ArtifactRef；
- [ ] artifact 与 export 幂等；
- [ ] 不叠加远端重试；
- [ ] 无业务循环；
- [ ] recursion limit 有限制；
- [ ] 非 READY 不访问 MP；
- [ ] 零结果正常终止；
- [ ] Validator 失败不能成功；
- [ ] 支持 updates streaming；
- [ ] 支持 status/history；
- [ ] 支持受控 replay；
- [ ] SQLite 重启后可恢复查看；
- [ ] Direct 与 Workflow 结果一致；
- [ ] 第一、二阶段全部测试通过；
- [ ] 普通测试不联网；
- [ ] 不包含自治多智能体；
- [ ] 覆盖率达标。

---

## 45. 官方资料

1. LangGraph Overview  
   https://docs.langchain.com/oss/python/langgraph/overview

2. Graph API  
   https://docs.langchain.com/oss/python/langgraph/graph-api

3. Persistence  
   https://docs.langchain.com/oss/python/langgraph/persistence

4. Checkpointer integrations  
   https://docs.langchain.com/oss/python/integrations/checkpointers

5. Streaming  
   https://docs.langchain.com/oss/python/langgraph/streaming

6. Testing  
   https://docs.langchain.com/oss/python/langgraph/test

7. Interrupts  
   https://docs.langchain.com/oss/python/langgraph/interrupts

8. StateGraph Reference  
   https://reference.langchain.com/python/langgraph/graph/state/StateGraph

9. SqliteSaver Reference  
   https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver

10. RetryPolicy Reference  
    https://reference.langchain.com/python/langgraph/types/RetryPolicy
