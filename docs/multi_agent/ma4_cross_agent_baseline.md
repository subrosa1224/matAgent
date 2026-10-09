# MA-4 跨 Agent 最小闭环验收记录

> 对应计划：`unified_multi_agent_ui_plan.md`  
> 阶段：MA-4  
> 日期：2026-08-25  
> 状态：已验收

## 1. 首个闭环

```text
论文 dossier
  → MaterialClue（题名、元素、式样线索、原文、页码、审核状态）
  → SearchMaterialsInput（required_elements、稳定性、排序、字段、限额）
  → Materials Project 查询快照
  → 双来源联合报告
```

## 2. 关键边界

- 子 Agent 之间不传递自由文本材料约束；
- 数据库只接收经过 Pydantic 校验的元素符号；
- 论文中的化学式只保留为 `formula_candidates`，不作为精确数据库公式过滤；
- 默认允许数据库候选包含额外元素；
- 数据库候选必须满足结构化的稳定性过滤；
- 每条线索保留 document ID、chunk ID、页码和 dossier 审核状态；
- 联合报告明确声明“论文组成仅是检索线索”；
- 某篇线索缺失或数据库失败时返回部分结果，不伪造候选。

## 3. 真实验收案例

论文：`Characterization of 3D-printed gelatin/sodium alginate/58S bioactive
glass scaffold and osteogenesis in rabbit cranial defects`

- 原始安全线索：`SiO2`、`CaO`、`P2O5`；
- 转换元素体系：`Ca–O–P–Si`；
- 交联剂 `CaCl2` 已被排除，没有把 `Cl` 混入查询；
- Materials Project 返回候选：`mp-1227400`，`Ca5Si(PO6)2`；
- 查询快照：`query-100bba427ebf4a709fcfd3e6f4f8cae6`；
- 由于论文 dossier 尚未人工批准，结果状态为 `partial` 并显示待审核警告；
- 报告没有将论文中的 58S 生物活性玻璃等同于该数据库候选。

## 4. 自动测试

- Master、跨 Agent 契约、Renderer 和统一 UI 定向测试通过；
- 覆盖无有效线索、待审核线索、结构化查询、候选渲染和部分失败；
- 真实 Materials Project 只读查询通过。

## 5. 使用方式

```powershell
uv run materials-screen master ui --port 8501
```

先上传并分析论文，再在“自动判断”模式输入：

```text
根据这些论文的组成查询Materials Project候选
```
