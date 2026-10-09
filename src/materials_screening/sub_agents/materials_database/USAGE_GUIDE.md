# Materials Database Agent 使用指南

本文说明如何配置和使用统一材料数据库查询 Agent。当前版本的数据源仅为 Materials Project，生产大模型仅接入书生 `intern-s2-preview-35b`。

## 1. 环境配置

请在项目配置方式支持的位置设置以下环境变量，不要把 Token 提交到代码仓库。

```dotenv
LLM_PROVIDER=intern
INTERN_API_KEY=你的书生API_TOKEN
INTERN_BASE_URL=https://chat.intern-ai.org.cn/api/v1/
INTERN_MODEL=intern-s2-preview-35b
INTERN_THINKING_MODE=true
MP_API_KEY=你的Materials_Project_API_KEY
```

书生 Token 只填写 Token 本身，不需要添加 `Bearer ` 前缀。离线测试可使用项目现有 mock 配置，无需真实调用外部服务。

## 2. 启动方式

以项目当前入口为准启动 Master Agent 或 Gradio 界面。可先查看命令帮助：

```powershell
uv run materials-screen --help
uv run materials-screen master ask -m "寻找带隙大于 2 eV 的稳定氧化物" --progress
```

如果已经把项目安装到当前 Python 环境，也可以直接运行 `materials-screen`。

Master Agent 会把相关请求路由到 `materials_database`。调用方不应再直接依赖旧的材料筛选或离群检测 Agent 名称。

## 3. 常用提问

### 条件筛选

```text
查找含 Li、Fe、O，带隙大于 1 eV，凸包上能量不超过 0.05 eV/atom 的材料。
```

可以组合元素、化学式和多个数值条件。结果会返回 `query_id`，供后续操作复用。

### 排序和 Top-K

```text
查询含 Si 和 O 的材料，按带隙从高到低排序，只看前 10 个。
```

请明确排序属性、方向和数量。未要求排序时，Agent 不应擅自将某属性作为排名依据。

### 材料详情

```text
查看 mp-149 的材料详情。
```

详情请求适用于明确的 Materials Project ID。

### 多材料比较

```text
比较 mp-149、mp-13 和 mp-2534 的带隙、密度、形成能和凸包上能量。
```

未指定比较字段时，Agent 可返回默认关键字段；指定字段时只重点展示用户关心的属性。

### 描述统计

```text
对刚才的查询结果统计带隙和密度的均值、中位数、标准差及范围。
```

推荐基于上一轮返回的 `query_id` 统计，避免重新查询导致样本集合变化。

### 离群检测

```text
检查刚才结果中带隙的离群材料。
```

离群检测必须有明确意图和目标属性。仅提到“带隙”不等于要求检测带隙离群值。例如：

- “查询带隙大于 2 eV 的材料”只执行筛选。
- “找出带隙异常的材料”才执行离群检测。

### 结果导出

```text
把刚才的结果导出为 CSV。
```

当前支持 CSV 和 JSON。导出的是指定 `query_id` 对应的结构化材料结果，不是聊天记录、模型推理过程或自然语言回答。文件默认写入：

```text
data/material_queries/exports/
```

## 4. 推荐的多轮工作流

```text
用户筛选材料
  → Agent 返回结果摘要和 query_id
  → 用户继续分页、查看详情或比较
  → 用户按需请求统计或离群检测
  → 用户确认后导出 CSV/JSON
```

示例：

```text
用户：筛选含 Li 的稳定材料，按形成能升序取前 20 个。
用户：比较其中前 5 个的带隙和密度。
用户：统计这 20 个材料的带隙分布。
用户：再检查带隙离群值。
用户：把完整的 20 条结果导出为 CSV。
```

## 5. 工具语义

| 工具 | 何时调用 | 关键输出 |
| --- | --- | --- |
| `search_materials` | 新查询、筛选、排序、Top-K | `query_id`、总数、首批记录 |
| `get_query_result` | 查看同一结果集的后续页面 | 分页记录 |
| `get_material_details` | 用户给出单个材料 ID | 完整材料详情 |
| `compare_materials` | 用户要求比较多个材料 | 对齐后的属性表 |
| `describe_materials` | 用户要求统计分析 | 描述统计量 |
| `detect_material_outliers` | 用户明确要求异常/离群分析 | 离群记录及方法信息 |
| `export_materials` | 用户明确要求下载或导出 | 文件路径和格式 |

一次复杂请求可能连续调用多个工具。数据库 Agent 已允许有限的多步工具调用，但应避免无关调用和重复查询。

## 6. 支持字段

当前可稳定用于数值筛选、排序、比较、统计和离群检测的字段为：

```text
band_gap_ev
density_g_cm3
formation_energy_ev_atom
energy_above_hull_ev_atom
```

自然语言中的“带隙、密度、形成能、凸包上能量”应分别映射到以上字段。单位必须保持一致，不要在没有显式换算的情况下混用。

## 7. 使用约束

- 第一版只查询 Materials Project，不做跨数据库聚合。
- 必须提供有效的 `MP_API_KEY` 才能进行在线查询。
- 不把书生 Token 或 Materials Project Key 写入前端、日志、导出文件或版本库。
- 统计与离群结果受查询样本量、缺失值和检测方法影响，应同时展示样本数和方法。
- 大结果集优先使用分页与查询快照，不要把全部记录一次塞入模型上下文。
- 导出是用户明确请求后的动作，不应为每次查询自动生成文件。

## 8. 常见问题

### 查询结果为空

检查元素组合和数值范围是否过严，再逐步放宽条件；同时确认 Materials Project API Key 有效。

### 书生模型鉴权失败

确认 `INTERN_API_KEY` 未包含 `Bearer `，模型名为 `intern-s2-preview-35b`，并检查 Token 是否过期。

### 后续操作找不到结果

确认使用的是当前会话返回的有效 `query_id`，并检查 `data/material_queries/` 中的快照是否仍存在。

### 为什么没有自动离群检测

这是预期行为。离群检测属于分析动作，只有用户明确提出相关意图时才执行。

### 导出的内容是什么

导出内容是查询快照中的结构化材料记录；导出范围应与对应 `query_id` 保持一致。

## 9. 开发验证

```powershell
pytest -q -k "materials_database or material_database"
```

修改字段映射、查询逻辑或工具参数后，应至少覆盖筛选、排序、分页、统计、离群检测、导出以及旧入口兼容性测试。

## 10. 全功能命令清单

以下命令均在项目根目录执行，并显式使用书生模型和 Materials
Project。PowerShell 中建议将每条命令写在一行，避免把终端显示的 `>>`
提示符复制进命令。

### 10.1 条件筛选

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project -m "筛选含 Li、Fe、O，带隙大于 1 eV，energy_above_hull_ev_atom 不超过 0.05 eV/atom 的稳定材料，返回材料 ID、化学式、带隙和凸包上能量" --progress
```

### 10.2 排序和 Top-K

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project -m "查询含 Si 和 O 的稳定材料，按 band_gap_ev 从高到低排序，返回前 10 个及其材料 ID、化学式和带隙" --progress
```

### 10.3 材料详情

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project -m "查询 mp-149 的材料详情，返回化学式、元素、带隙、密度、形成能、凸包上能量、稳定性、晶系和空间群" --progress
```

### 10.4 多材料比较

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project -m "比较 mp-149、mp-13 和 mp-2534 的 band_gap_ev、density_g_cm3、formation_energy_ev_atom 和 energy_above_hull_ev_atom，并用表格展示" --progress
```

### 10.5 描述统计

描述统计应复用一次查询产生的快照。先运行：

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project -m "筛选含 Li 和 O 的稳定材料，返回前 100 个的材料 ID、化学式、band_gap_ev 和 density_g_cm3" --progress
```

从输出中复制 `Conversation` 值，替换下面的 `<CONVERSATION_ID>`：

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "对刚才的查询结果统计 band_gap_ev 和 density_g_cm3 的样本数、缺失数、均值、标准差、最小值、中位数和最大值" --progress
```

### 10.6 离群检测

先执行 10.5 的初始筛选，再在同一会话中运行：

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "对刚才的查询结果执行 band_gap_ev 离群检测，列出离群材料、检测方法、阈值和样本数" --progress
```

只有明确出现“离群检测”“异常值”等意图时才会调用离群检测工具；普通带隙筛选不会自动执行该功能。

### 10.7 结果导出

先执行任一筛选命令，再用相同的会话 ID 导出 CSV：

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "把刚才查询快照中的完整结构化材料结果导出为 CSV，并返回文件路径" --progress
```

导出 JSON：

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "把刚才查询快照中的完整结构化材料结果导出为 JSON，并返回文件路径" --progress
```

导出的内容是查询快照中的结构化材料记录，不包含聊天记录、模型推理过程或自然语言回答。默认导出目录为 `data/material_queries/exports/`。

### 10.8 一套连续覆盖全部能力的会话

依次运行下面的提问，并从第二条开始始终传入第一条命令返回的同一个
`Conversation` 值：

```powershell
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project -m "筛选含 Li 和 O 的稳定材料，按 formation_energy_ev_atom 从低到高排序，取前 20 个，并返回材料 ID、化学式、带隙、密度、形成能和凸包上能量" --progress
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "查看结果中第一个材料的详细信息" --progress
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "比较结果中的前 3 个材料的带隙、密度、形成能和凸包上能量" --progress
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "统计这 20 个材料的带隙和密度" --progress
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "检测这 20 个材料中带隙的离群值" --progress
uv run materials-screen master ask --llm-provider intern --materials-repository materials-project --conversation-id <CONVERSATION_ID> -m "将这 20 个材料的完整查询结果导出为 CSV" --progress
```
