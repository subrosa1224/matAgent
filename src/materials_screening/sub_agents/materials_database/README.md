# Materials Database Agent

`materials_database` 是项目中统一的材料数据库查询 Agent。第一版仅接入 Materials Project，并吸收原“材料筛选”和“离群检测”Agent 的能力。

## 能力范围

- 条件筛选：按元素、化学式及数值属性筛选材料
- 排序和 Top-K：按指定属性升序或降序返回前 K 项
- 材料详情：按 Materials Project ID 查询单个材料
- 多材料比较：将多个材料的关键属性并排比较
- 描述统计：计算计数、均值、标准差、最小值、中位数和最大值
- 离群检测：对用户明确指定的属性执行离群分析
- 结果导出：将当前查询快照导出为 CSV 或 JSON

如果用户只要求查询带隙等属性，Agent 不会自动执行离群检测；只有用户明确表达“离群、异常值、异常点”等意图时才调用离群工具。

## 目录结构

```text
materials_database/
├── __init__.py       # 对外导出
├── models.py         # 工具参数和返回模型
├── tools.py          # 七类数据库工具
├── prompt.py         # Agent 系统提示词
├── spec_factory.py   # AgentSpec 与兼容入口工厂
├── mock_model.py     # 离线开发模型
├── README.md
└── USAGE_GUIDE.md
```

底层服务位于：

- `services/material_database_service.py`：查询、比较、统计、离群检测和导出
- `services/query_result_store.py`：查询快照及分页结果存储

## 对外工具

| 工具 | 用途 |
| --- | --- |
| `search_materials` | 条件筛选、排序、Top-K，并生成查询快照 |
| `get_material_details` | 获取单个材料详情 |
| `get_query_result` | 按查询 ID 分页读取结果 |
| `compare_materials` | 比较多个材料 |
| `describe_materials` | 对查询结果做描述统计 |
| `detect_material_outliers` | 对查询结果中的指定属性做离群检测 |
| `export_materials` | 导出查询结果 |

## 当前稳定属性

| 字段 | 含义 | 单位 |
| --- | --- | --- |
| `band_gap_ev` | 带隙 | eV |
| `density_g_cm3` | 密度 | g/cm³ |
| `formation_energy_ev_atom` | 每原子形成能 | eV/atom |
| `energy_above_hull_ev_atom` | 每原子凸包上能量 | eV/atom |

材料记录还可包含材料 ID、化学式、元素集合、晶系及稳定性等非数值信息。新增属性时，应同步修改数据模型、Materials Project 映射、筛选白名单、排序/统计逻辑和测试。

## 查询快照

首次查询会生成 `query_id`，后续分页、比较、统计、离群检测和导出都应尽量复用该 ID，以保证处理的是同一批数据。

默认目录：

```text
data/material_queries/
└── exports/
```

## 与旧 Agent 的关系

`materials_database` 是正式且唯一的新入口。旧入口暂时保留为兼容别名：

- `materials_screening`
- `outlier_detection`

它们最终委托给同一个 Materials Database Agent 和底层服务，方便旧调用方平滑迁移；稳定后可删除兼容入口。

## 开发验证

在项目根目录运行：

```powershell
pytest -q
```

若只验证本模块，可按测试文件或关键字运行：

```powershell
pytest -q -k "materials_database or material_database"
```

具体配置、提问示例和操作流程见 [USAGE_GUIDE.md](./USAGE_GUIDE.md)。

其中 [USAGE_GUIDE.md](./USAGE_GUIDE.md) 的“全功能命令清单”提供了可直接运行的 PowerShell 命令，覆盖条件筛选、排序和 Top-K、材料详情、多材料比较、描述统计、离群检测与 CSV/JSON 结果导出。
