import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/Desktop/matAgent/outputs/data_collection_plan";
const outputPath = `${outputDir}/material_agent_data_requirements.xlsx`;
const previewDir = `${outputDir}/preview`;
const font = "Arial";

const workbook = Workbook.create();
const overview = workbook.worksheets.add("数据总览");
const master = workbook.worksheets.add("完整数据清单");
const schema = workbook.worksheets.add("字段字典");
const phases = workbook.worksheets.add("分阶段计划");
const workload = workbook.worksheets.add("专家工作量");

for (const sheet of [overview, master, schema, phases, workload]) {
  sheet.showGridLines = false;
}

const masterHeaders = [
  "编号", "数据类别", "具体数据项", "计数单位", "最低可运行数量", "推荐固定目标",
  "单任务参考量", "主要来源", "必须字段", "建议格式", "专家责任",
  "主要支持模块", "优先级", "验收标准", "已收集数量", "状态", "备注"
];

const masterRows = [
  ["D01", "科研任务", "端到端晶态无机材料科研任务", "个任务", 12, 30, "1个完整问题", "专业人员历史课题、公开案例", "问题、背景、应用场景、材料范围、任务边界、预期输出", "DOCX/MD/XLSX", "定义并最终确认任务", "全系统", "P0", "每个任务能够转化为明确筛选流程，且不存在未说明的关键边界", "", "未开始", "30个任务覆盖5类任务，每类6个"],
  ["D02", "科研任务", "深度标注任务", "个任务", 6, 12, "从D01中选取", "D01任务子集", "完整证据、规则、推理图、专家报告", "文件夹+XLSX", "完成深度审核", "推理图、STaR、报告", "P1", "每个任务具备从问题到最终结论的完整可审计链条", "", "未开始", "属于D01子集，不重复计入任务总量"],
  ["D03", "任务定义", "结构化任务合同", "份", 12, 30, "1份/任务", "专业人员与系统共同整理", "硬条件、软条件、单位、优先级、缺失值策略、排除规则、停止条件", "XLSX/JSON", "审核全部科研条件", "主控Agent、推理图", "P0", "每项条件都能映射为可执行查询、排序、核验或人工复核动作", "", "未开始", "不得使用“性能好”等无法执行的模糊条件"],
  ["D04", "任务定义", "固定数据库快照和查询版本", "份", 12, 30, "1份/任务", "Materials Project等公开数据库", "数据库名、版本、查询日期、字段定义、查询参数、原始返回", "JSON/CSV", "确认属性含义与任务适配性", "数据库Agent、可复现性", "P0", "相同快照和参数能够复现同一初始候选集", "", "未开始", "数据库升级后仍保留原始快照"],
  ["D05", "文献", "唯一论文全文", "篇", 60, 220, "平均8–12篇/任务，可复用", "论文正文与补充材料", "DOI、题名、作者、年份、全文、许可信息", "PDF/XML/TXT", "确认是否与材料和任务相关", "增强RAG", "P0", "全文可解析，元数据完整，关键任务至少包含实验或直接证据论文", "", "未开始", "220是唯一论文数量；同一论文可支持多个任务"],
  ["D06", "文献", "任务—论文相关性关系", "条关系", 120, 330, "11条/任务", "D05论文与D01任务配对", "task_id、paper_id、相关等级、对应材料、作用、判断理由", "XLSX/JSONL", "标注或审核相关性", "增强RAG、评价", "P0", "每条关系具有相关/背景/负例标签及简短理由", "", "未开始", "与唯一论文数不是同一计数口径"],
  ["D07", "候选材料", "完整初始候选记录", "条候选—任务记录", 300, 1500, "平均50条/任务", "数据库筛选、人工历史清单", "task_id、material_id、化学式、晶型、空间群、属性、来源、数据库版本", "CSV/Parquet", "审核边界候选，不必逐条手工录入", "数据库Agent、候选漏斗", "P0", "保留未经过人工删减的完整候选并可追溯来源", "", "未开始", "任务间重复材料仍分别计数"],
  ["D08", "专家金标", "专家最终候选判断", "条候选—任务判断", 300, 1500, "覆盖D07全部记录", "材料专业人员", "保留/排除/待定、等级、理由、证据、置信度、建议", "XLSX/JSONL", "逐项确认最终状态", "端到端评价、SFT", "P0", "D07每条候选均有唯一状态；不确定项明确标为待定", "", "未开始", "不能只提供最终入选材料"],
  ["D09", "材料身份", "规范材料身份实体", "个唯一实体", 200, 700, "平均20–30个唯一实体/任务", "数据库、CIF、论文", "规范名称、化学式、材料ID、晶型、空间群、别名、掺杂、缺陷", "CSV/JSON", "审核别名、晶型和结构对应", "身份感知RAG、校验器", "P0", "每个关键候选能够映射到明确身份；无法确定时保留歧义", "", "未开始", "实体数量小于候选—任务记录数量"],
  ["D10", "材料身份", "易混淆材料身份对", "对", 100, 350, "约10–12对/任务", "同式异相、别名、掺杂/纯相案例", "entity_a、entity_b、是否同一身份、差异字段、依据", "XLSX/JSONL", "必须审核每对结论", "身份校验、困难负例", "P0", "覆盖同化学式异晶型、近似名称、掺杂与纯相等主要错误", "", "未开始", "至少一半应为“不能合并”的负例"],
  ["D11", "证据", "正向Claim–Evidence记录", "条", 300, 1200, "40条/任务", "论文正文、表格、图、数据库记录", "claim、材料身份、原文、页码、证据类型、条件、支持强度、来源", "JSONL/XLSX", "确认关键结论与证据是否匹配", "增强RAG、证据账本", "P0", "所有证据定位到原文或固定数据库记录，且能直接或有条件支持结论", "", "未开始", "重要候选尽量有两个独立来源"],
  ["D12", "证据", "困难负例或错误证据", "条", 300, 1200, "40条/任务", "相似但不支持的论文片段", "query/claim、错误证据、错误类型、材料差异、条件差异、原因", "JSONL/XLSX", "审核高相似度困难负例", "检索重排、事实核查", "P0", "负例在文本上具有一定相关性，但科研上不能支持目标结论", "", "未开始", "与D11约1:1，避免只训练正样本"],
  ["D13", "研究条件", "实验条件记录", "条结果—条件记录", 200, 1000, "约30–40条/实验型任务", "实验论文、补充材料、用户数据", "材料身份、制备法、原料、温度、压力、时间、气氛、样品、测试方法", "CSV/JSONL", "审核影响判断的关键条件", "条件感知RAG、实验建议", "P0", "每条实验性质或合成结论都保留足以判断身份和可比性的条件", "", "未开始", "不要求摘录与筛选无关的全部仪器参数"],
  ["D14", "研究条件", "计算条件记录", "条结果—条件记录", 100, 500, "约15–20条/计算型任务", "计算论文、数据库方法说明", "泛函、U值、赝势、k点、磁性设置、参考相、软件、数据库版本", "CSV/JSONL", "审核关键计算方法差异", "条件感知RAG、计算事实核查", "P1", "计算性质均可区分方法体系，不把不同体系数值无条件合并", "", "未开始", "只有涉及计算证据的任务才需要"],
  ["D15", "历史数据", "文献结构化原始—修改—最终三联数据", "组", 100, 500, "不按任务平均", "合作团队以前的结构化任务", "原始输入、模型初稿、人工修改、最终版本、修改原因、审核者", "JSONL/XLSX", "最终版本和关键修改需确认", "抽取SFT、错误学习", "P0", "三版可通过稳定ID对齐，能够定位修改字段和原因", "", "未开始", "优先获取已有历史数据，可大幅降低重新标注量"],
  ["D16", "冲突核查", "文献证据冲突对", "对", 30, 150, "5对/任务", "不同论文或同文不同结果", "证据A、证据B、是否真冲突、条件差异、来源等级、处理建议", "XLSX/JSONL", "必须判断是否构成真实冲突", "事实核查、结论降级", "P1", "至少覆盖真实冲突、表面冲突和证据不足三类", "", "未开始", "不同方法导致的差异不应自动判为冲突"],
  ["D17", "可比性", "实验/计算结果可比性判断对", "对", 100, 350, "约10–12对/任务", "论文、数据库、用户实验", "记录A、记录B、可比等级、关键差异、允许的分析动作", "XLSX/JSONL", "审核全部困难判断", "校验器、数据分析Agent", "P1", "能够区分直接可比、有条件可比、不可比和信息不足", "", "未开始", "覆盖晶型、温压、测试法、实验/计算差异"],
  ["D18", "专业规则", "物理化学与事实核查核心规则", "条规则", 30, 54, "全局规则库", "专家知识、数据库说明、标准、方法文献", "规则、适用范围、输入字段、触发条件、动作、例外、依据", "XLSX/JSON", "制定或逐条确认", "约束校验算法", "P1", "规则可执行、适用范围明确、存在依据且定义例外处理", "", "未开始", "建议：身份10、单位8、实验/计算8、可比性10、稳定性8、证据边界10"],
  ["D19", "专业规则", "规则通过案例", "条案例", 150, 270, "5条/规则", "D18规则的真实或构造案例", "rule_id、输入、预期状态、理由、来源", "JSONL/XLSX", "审核困难案例", "校验器测试", "P1", "每条规则至少5个应通过案例，边界值单独覆盖", "", "未开始", "用于测量误报率"],
  ["D20", "专业规则", "规则违反或降级案例", "条案例", 150, 270, "5条/规则", "历史错误、模型生成、专家构造", "rule_id、错误输入、违反点、预期动作、正确修正", "JSONL/XLSX", "审核全部严重错误", "校验器测试、奖励惩罚", "P1", "每条规则至少5个阻断、降级或待复核案例", "", "未开始", "用于测量漏报率"],
  ["D21", "结论控制", "科研结论强度标注", "条结论", 300, 600, "20条/任务", "系统输出与专家报告", "结论、证据集合、确认/支持/推测/不足/错误/待复核、理由", "JSONL/XLSX", "必须确认标签", "事实核查、报告生成", "P1", "不同证据强度下的允许措辞边界明确", "", "未开始", "重点覆盖过度因果和过度泛化"],
  ["D22", "机理知识", "过程—结构—性质—性能关系", "条关系", 150, 500, "约15–20条/任务", "机理论文、综述、专家判断", "源实体、关系、目标实体、方向、条件、证据、强度、来源", "JSONL/图谱CSV", "审核关系和适用条件", "显式推理图、机理校验", "P1", "每条关系可追溯，且不把相关性自动写为因果", "", "未开始", "采用Processing→Structure→Property→Performance骨架"],
  ["D23", "机理知识", "因果证据等级标签", "条", 150, 500, "与D22一一对应", "D22关系", "实验干预/统计相关/机理解释/专家推测/争议、置信度、依据", "JSONL/XLSX", "必须确认等级", "因果边界控制", "P1", "D22全部关系具有证据等级和允许表述", "", "未开始", "无干预数据时不得标记为严格因果发现"],
  ["D24", "推理数据", "专家标准显式推理图", "张", 6, 12, "1张/深标任务", "专家解决D02任务的过程", "节点、边、输入、动作、证据、状态、输出、停止条件", "JSON/GraphML", "必须构建或审核", "创新一原型", "P1", "从任务定义到候选结论全程可追溯，无悬空节点", "", "未开始", "不要求记录自由形式内部思维链"],
  ["D25", "推理数据", "系统候选推理轨迹", "条", 120, 800, "约25–30条/任务", "现有系统和升级模型生成", "任务、推理图、工具调用、候选变化、证据、结果、自动评分", "JSONL", "只审核进入训练集和高风险轨迹", "STaR自举", "P2", "轨迹可重放，并能定位每一步工具、证据和状态变化", "", "未开始", "由系统批量生成，不应要求专家手写"],
  ["D26", "推理数据", "专家修正轨迹", "条", 60, 240, "20条/深标任务", "D25错误或次优轨迹", "错误步骤、错误类型、修正动作、正确结果、修改理由", "JSONL/XLSX", "必须审核", "STaR、错误诊断", "P2", "每条修正能明确指出首个关键错误及正确替代步骤", "", "未开始", "优先修正身份、证据和规则错误"],
  ["D27", "偏好数据", "回答或推理轨迹偏好对", "对", 300, 1200, "40对/任务", "同任务不同系统输出", "输出A、输出B、优选项、理由、严重错误、审核者", "JSONL/XLSX", "必须判断科研优劣", "偏好优化、可选RL", "P2", "偏好理由能够映射到正确性、证据、约束、完整性等维度", "", "未开始", "完整RL若不稳定，应停留在SFT/偏好实验"],
  ["D28", "训练样本", "SFT/LoRA结构化训练样本", "条", 1500, 6500, "由全部数据聚合", "D03、D11、D15、D24–D26转换", "instruction、context、tools、structured_output、evidence、quality_status", "JSONL", "专家确认20%–30%，高风险样本全审", "模型微调", "P2", "去重、无测试泄漏、证据可追溯、训练格式可解析", "", "未开始", "建议组成：合同500、规划500、工具800、检索1200、抽取1200、候选1000、纠错800、总结500"],
  ["D29", "数值数据", "结构化数值数据集", "份数据集", 8, 20, "按设计类型配置", "用户实验、公开数据、论文补充材料、数据库、模拟边界数据", "样本ID、材料ID、分组、属性、值、单位、来源、条件、批次", "CSV/XLSX/Parquet", "审核实验设计与科研解释", "数据分析Agent", "P1", "每份数据具有数据字典、来源、分析许可和预期分析类型", "", "未开始", "建议5份真实实验、5份公开/补充、5份数据库/文献、5份异常测试"],
  ["D30", "数值数据", "实验或研究设计说明", "份", 8, 20, "1份/数值数据集", "实验记录、论文方法、数据集说明", "样本单位、独立/配对/重复、批次、对照、随机化、主要指标", "DOCX/MD/XLSX", "必须确认设计含义", "数据分析Agent", "P1", "能够据此确定允许的统计分析；未知设计不得强行推断", "", "未开始", "与D29一一对应"],
  ["D31", "报告金标", "专家版完整科研报告", "份", 6, 12, "1份/深标任务", "专家既有报告或根据模板审核", "任务、方法、候选、证据、冲突、不确定性、结论、验证建议", "DOCX/PDF/MD", "提供或完整审核", "报告生成、端到端评价", "P1", "报告中的关键结论均能映射到金标候选和证据", "", "未开始", "不要求统一文风，但必须保持科研边界"],
  ["D32", "测试集", "完全独立端到端测试任务", "个任务", 3, 6, "D01中的6个保留任务", "未进入训练和提示开发的新任务", "完整任务合同、候选、证据、专家结果", "冻结目录", "最终盲审", "全部创新评价", "P0", "任务、论文来源和核心候选不进入训练；测试前冻结", "", "未开始", "属于D01的子集，不额外增加任务总数"],
  ["D33", "合规与溯源", "数据来源、许可与审核记录", "批次", 1, 10, "每次交付1批次", "所有数据提供方", "提供者、来源、许可、用途限制、日期、版本、审核者", "XLSX/MD", "提供方确认授权范围", "训练合规、复现", "P0", "每份文件能够追溯来源并明确是否允许检索、训练和发表", "", "未开始", "批次数可随实际交付调整；推荐按10批次管理"],
];

const schemaHeaders = ["对象", "主键建议", "必须字段", "可选字段", "一行/一条代表什么", "禁止混淆", "主要输出格式"];
const schemaRows = [
  ["科研任务", "task_id", "问题、材料范围、应用、边界、预期输出", "时间范围、成本限制", "一个完整科研决策问题", "不能把问题改写成单纯数据库过滤", "JSON/XLSX"],
  ["任务合同", "contract_id", "task_id、硬条件、软条件、单位、缺失策略、停止条件", "条件权重、人工门禁", "一个冻结版本的可执行研究设定", "硬条件与排序偏好必须分开", "JSON"],
  ["材料身份", "entity_id", "化学式、晶型、空间群、材料ID、来源", "CIF、别名、掺杂、缺陷", "一个规范材料身份", "相同化学式不等于相同材料", "CSV/JSON"],
  ["候选判断", "task_id+entity_id", "初始属性、状态、理由、证据、置信度", "排序、验证建议", "某材料在某任务中的一次决策", "任务间重复候选不能直接去重", "CSV/JSONL"],
  ["论文", "paper_id/DOI", "题名、年份、全文、许可、来源", "摘要、关键词", "一篇唯一文献", "唯一论文数与任务—论文关系数不同", "PDF+JSON"],
  ["证据", "evidence_id", "claim、原文、位置、材料身份、关系、条件、来源", "证据等级、冲突组", "一条能被定位和审核的证据", "摘要不能作为关键候选的唯一直接证据", "JSONL"],
  ["实验条件", "condition_id", "材料、制备/测试法、温度、压力、样品、来源", "气氛、时间、仪器", "一项结果对应的一组关键实验条件", "不同样品或晶型不能共用条件", "CSV/JSONL"],
  ["计算条件", "condition_id", "材料、泛函、软件、U值、参考相、来源", "k点、截断能、赝势", "一项计算结果对应的方法条件", "PBE、HSE、实验值不能无标记合并", "CSV/JSONL"],
  ["专业规则", "rule_id", "规则、范围、触发、动作、例外、依据", "严重度、版本", "一个可执行校验规则", "专家经验与可验证事实应分别标记", "JSON/XLSX"],
  ["机理关系", "relation_id", "源、关系、目标、方向、条件、证据等级、来源", "置信度、争议状态", "一个有条件的材料机理关系", "相关、机理解释与因果干预不可混称", "JSONL/图谱CSV"],
  ["推理图", "graph_id", "task_id、节点、边、证据引用、状态、输出", "替代路径、失败原因", "一个完整任务的结构化科研路径", "不保存不可审核的自由形式长思维链", "JSON/GraphML"],
  ["数值数据", "dataset_id+sample_id", "样本、材料、属性、值、单位、来源、条件", "批次、异常标记", "一次独立观测或明确层级的汇总值", "单次测量、样本均值和论文均值不可混为一行", "CSV/Parquet"],
  ["偏好对", "preference_id", "task_id、输出A、输出B、选择、理由、审核者", "维度评分", "同一任务下的一次成对比较", "不能只记录选择而不记录关键理由", "JSONL"],
];

const phaseHeaders = ["阶段", "目标", "必须达到的固定数量", "允许暂缺", "进入下一阶段的门槛", "主要责任人"];
const phaseRows = [
  ["阶段A：最小闭环", "证明流程可运行", "12任务；60论文；300候选判断；300正证据；300负例；100身份对；8数值数据集", "完整规则库、RL偏好", "至少3个任务完整跑通且全部关键结论可追溯", "项目人员+材料专家"],
  ["阶段B：核心RAG完整实现", "完成身份与条件感知多源RAG", "30任务；220唯一论文；330任务—论文关系；1200正证据；1200负例；700身份实体；350身份对", "大规模偏好训练", "相对普通RAG在独立测试任务上提高证据召回、身份准确和引用正确性", "项目人员，专家审核"],
  ["阶段C：校验闭环", "完成物理化学与事实核查核心模块", "54规则；270通过案例；270违反案例；150冲突对；350可比性对；600结论强度标签", "全面机理规则库", "已知严重错误被阻断，证据不足能正确降级，报告误报和漏报", "材料专家+项目人员"],
  ["阶段D：推理与训练原型", "完成显式推理图和STaR/SFT", "12推理图；800系统轨迹；240修正轨迹；1200偏好对；6500 SFT样本", "完整策略强化学习", "SFT版本在独立任务上优于未训练版本，且硬约束违反率不升高", "项目人员，专家审核关键样本"],
  ["阶段E：冻结评价", "形成论文级结果", "6独立测试任务；12专家报告；完整消融M0–M4", "M5强化学习", "测试任务、论文和候选无训练泄漏；完成错误分类和局限性报告", "导师+材料专家+项目人员"],
];

const workloadHeaders = ["专家工作", "目标数量", "单条预计审核时间（分钟）", "预计总小时", "普通人员可预处理", "专家必须完成的判断", "说明"];
const workloadRows = [
  ["任务与合同确认", 30, 60, "=B5*C5/60", "整理背景与模板", "确认研究边界和筛选逻辑", "按已有课题材料充分时估算"],
  ["候选最终判断", 1500, 1.5, "=B6*C6/60", "预填数据库属性和证据", "保留、排除、待定及关键理由", "普通候选可批量确认，困难项单独复核"],
  ["正证据审核", 1200, 1.5, "=B7*C7/60", "定位原文与录入条件", "判断证据是否支持结论", "不包含从零阅读整篇论文时间"],
  ["身份困难对审核", 350, 2, "=B8*C8/60", "准备材料ID和结构字段", "判断能否合并", "重点覆盖同式异相"],
  ["冲突与可比性审核", 500, 2, "=B9*C9/60", "并列原始记录", "判断真冲突及可比等级", "D16和D17合计"],
  ["核心规则确认", 54, 15, "=B10*C10/60", "整理规则草案与出处", "确认适用范围、例外和动作", "规则测试案例另做抽样"],
  ["标准推理图审核", 12, 120, "=B11*C11/60", "系统生成初稿", "确认步骤、依赖和停止条件", "每张按2小时估算"],
  ["专家修正轨迹", 240, 3, "=B12*C12/60", "自动标出疑似错误", "确认首个关键错误和修正", "优先高风险轨迹"],
  ["偏好对判断", 1200, 0.75, "=B13*C13/60", "并列输出并突出差异", "选择较优输出并给出原因", "每对45秒是界面良好时的目标"],
  ["专家报告审核", 12, 120, "=B14*C14/60", "系统生成报告草稿", "核验候选、证据和科研边界", "每份按2小时估算"],
  ["预计专家总工作量", "", "", "=SUM(D5:D14)", "", "", "为规划估计，不包含会议、返工和从零选题时间"],
];

function writeTitle(sheet, title, subtitle, endCol) {
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: font, size: 16, bold: true, color: "#183B56" };
  sheet.getRange(`A3:${endCol}3`).format.borders = { bottom: { style: "medium", color: "#4F81BD" } };
  sheet.getRange("A3").values = [[subtitle]];
  sheet.getRange("A3").format.font = { name: font, size: 10, italic: true, color: "#5B6573" };
}

function formatTableSheet(sheet, title, subtitle, headers, rows, tableRange, tableName, widths, priorityCol = null) {
  writeTitle(sheet, title, subtitle, String.fromCharCode(64 + headers.length));
  sheet.getRange("A4").write([headers, ...rows]);
  const table = sheet.tables.add(tableRange, true, tableName);
  table.style = "TableStyleMedium2";
  sheet.getRange(tableRange).format.font = { name: font, size: 10, color: "#1F2933" };
  sheet.getRange(tableRange).format.verticalAlignment = "center";
  sheet.getRange(tableRange).format.wrapText = true;
  sheet.getRange(`A4:${String.fromCharCode(64 + headers.length)}4`).format.font = { name: font, size: 10, bold: true, color: "#FFFFFF" };
  sheet.getRange(`A4:${String.fromCharCode(64 + headers.length)}4`).format.fill = "#2F5597";
  sheet.getRange(`A5:${String.fromCharCode(64 + headers.length)}${rows.length + 4}`).format.rowHeight = 44;
  widths.forEach((width, idx) => sheet.getRangeByIndexes(0, idx, rows.length + 4, 1).format.columnWidth = width);
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(2);
  if (priorityCol) {
    const range = sheet.getRange(`${priorityCol}5:${priorityCol}${rows.length + 4}`);
    range.conditionalFormats.add("containsText", { text: "P0", format: { fill: "#FDE9D9", font: { color: "#9C0006", bold: true } } });
    range.conditionalFormats.add("containsText", { text: "P1", format: { fill: "#FFF2CC", font: { color: "#7F6000", bold: true } } });
    range.conditionalFormats.add("containsText", { text: "P2", format: { fill: "#E2F0D9", font: { color: "#375623", bold: true } } });
  }
}

formatTableSheet(
  master,
  "材料多智能体项目完整数据清单",
  "推荐目标用于论文级系统建设；最低数量仅用于原型验证。已收集数量和状态为后续可编辑字段。",
  masterHeaders,
  masterRows,
  `A4:Q${masterRows.length + 4}`,
  "DataRequirementsTable",
  [8, 13, 24, 15, 13, 13, 18, 22, 42, 15, 24, 20, 9, 38, 13, 12, 30],
  "M"
);
master.getRange(`O5:O${masterRows.length + 4}`).format.fill = "#FFF2CC";
master.getRange(`P5:P${masterRows.length + 4}`).format.fill = "#FFF2CC";
master.getRange(`P5:P${masterRows.length + 4}`).dataValidation = { rule: { type: "list", values: ["未开始", "收集中", "待专家审核", "已完成", "不适用"] } };
master.getRange(`E5:F${masterRows.length + 4}`).format.numberFormat = "#,##0";
master.getRange(`O5:O${masterRows.length + 4}`).format.numberFormat = "#,##0";

formatTableSheet(
  schema,
  "数据字段字典",
  "用于统一材料、文献、证据、规则和训练数据的记录口径。",
  schemaHeaders,
  schemaRows,
  `A4:G${schemaRows.length + 4}`,
  "FieldDictionaryTable",
  [17, 22, 48, 32, 30, 38, 18]
);

formatTableSheet(
  phases,
  "分阶段数据收集与验收",
  "先完成核心RAG，再扩展校验和推理训练；不得使用测试任务反复调提示词。",
  phaseHeaders,
  phaseRows,
  `A4:F${phaseRows.length + 4}`,
  "CollectionPhasesTable",
  [24, 28, 68, 28, 58, 25]
);

formatTableSheet(
  workload,
  "材料专家工作量估算",
  "估算假设普通人员已完成文献整理、字段预填和原文定位；实际时间会受任务复杂度影响。",
  workloadHeaders,
  workloadRows,
  `A4:G${workloadRows.length + 4}`,
  "ExpertWorkloadTable",
  [25, 15, 22, 15, 30, 38, 40]
);
workload.getRange("D5:D15").format.numberFormat = "0.0";
workload.getRange("A15:G15").format.font = { name: font, size: 10, bold: true, color: "#183B56" };
workload.getRange("A15:G15").format.fill = "#D9EAF7";

writeTitle(overview, "材料多智能体项目数据需求总览", "方案假设：创新③完整实现；创新②完成核心校验闭环；创新①完成推理图与STaR原型。", "H");
overview.getRange("A5:B15").values = [
  ["推荐固定目标", "数量"],
  ["端到端任务", 30],
  ["深度标注任务", 12],
  ["唯一论文", 220],
  ["候选判断", 1500],
  ["正向证据", 1200],
  ["困难负例", 1200],
  ["核心规则", 54],
  ["标准推理图", 12],
  ["SFT/LoRA样本", 6500],
  ["独立测试任务", 6],
];
overview.getRange("A5:B15").format.font = { name: font, size: 11, color: "#1F2933" };
overview.getRange("A5:B5").format = { fill: "#2F5597", font: { name: font, size: 11, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
overview.getRange("A5:B15").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
overview.getRange("B6:B15").format.numberFormat = "#,##0";
overview.getRange("A5:A15").format.columnWidth = 24;
overview.getRange("B5:B15").format.columnWidth = 15;

overview.getRange("D5:G10").values = [
  ["优先级", "含义", "第一轮重点", "是否依赖专家"],
  ["P0", "启动和核心RAG必需", "任务、候选、论文、证据、身份、许可", "是"],
  ["P1", "校验和推理图建设", "规则、冲突、条件、机理、数值数据", "是"],
  ["P2", "模型训练阶段", "轨迹、修正、偏好、SFT数据", "关键样本需要"],
  ["数量边界", "推荐目标不是统计学通用定律", "任务复杂度改变时应重新核算", "导师确认"],
  ["数据划分", "按整个任务和论文来源拆分", "18训练、6验证、6测试", "测试集盲审"],
];
overview.getRange("D5:G10").format.font = { name: font, size: 10, color: "#1F2933" };
overview.getRange("D5:G5").format = { fill: "#2F5597", font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
overview.getRange("D5:G10").format.wrapText = true;
overview.getRange("D5:G10").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
overview.getRange("D5:D10").format.columnWidth = 16;
overview.getRange("E5:E10").format.columnWidth = 30;
overview.getRange("F5:F10").format.columnWidth = 42;
overview.getRange("G5:G10").format.columnWidth = 20;
overview.getRange("D6:D6").format.fill = "#FDE9D9";
overview.getRange("D7:D7").format.fill = "#FFF2CC";
overview.getRange("D8:D8").format.fill = "#E2F0D9";

overview.getRange("A18:D22").values = [
  ["收集进度", "数据项数量", "已完成项", "完成率"],
  ["P0", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P0\")", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P0\",'完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B19=0,0,C19/B19)"],
  ["P1", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P1\")", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P1\",'完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B20=0,0,C20/B20)"],
  ["P2", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P2\")", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P2\",'完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B21=0,0,C21/B21)"],
  ["全部", "=COUNTA('完整数据清单'!$A$5:$A$37)", "=COUNTIFS('完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B22=0,0,C22/B22)"],
];
overview.getRange("B19:D22").formulas = [
  ["=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P0\")", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P0\",'完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B19=0,0,C19/B19)"],
  ["=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P1\")", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P1\",'完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B20=0,0,C20/B20)"],
  ["=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P2\")", "=COUNTIFS('完整数据清单'!$M$5:$M$37,\"P2\",'完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B21=0,0,C21/B21)"],
  ["=COUNTA('完整数据清单'!$A$5:$A$37)", "=COUNTIFS('完整数据清单'!$P$5:$P$37,\"已完成\")", "=IF(B22=0,0,C22/B22)"],
];
overview.getRange("A18:D22").format.font = { name: font, size: 10, color: "#1F2933" };
overview.getRange("A18:D18").format = { fill: "#2F5597", font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center" };
overview.getRange("A18:D22").format.borders = { insideHorizontal: { style: "thin", color: "#D9E2F3" }, bottom: { style: "thin", color: "#D9E2F3" } };
overview.getRange("A18:D22").format.columnWidth = 18;
overview.getRange("D19:D22").format.numberFormat = "0.0%";
overview.getRange("A25:G29").values = [
  ["使用说明", "内容", "", "", "", "", ""],
  ["数量含义", "最低数量用于原型运行；推荐固定目标用于论文级建设，不代表任何材料任务的统计学通用下限。", "", "", "", "", ""],
  ["专家数据", "专家主要提供任务边界、最终候选、身份/证据/机理审核；PDF整理和字段预填可由普通人员完成。", "", "", "", "", ""],
  ["实验数据", "用户真实实验数据不是启动前提。论文补充材料、公开数据和数据库表也可供数据分析Agent使用，但必须标明来源与允许的推断强度。", "", "", "", "", ""],
  ["测试隔离", "测试任务、对应论文和关键候选必须在训练前冻结，不能用于反复修改提示词、规则或模型。", "", "", "", "", ""],
];
overview.getRange("A25:G29").format.font = { name: font, size: 10, color: "#1F2933" };
overview.getRange("A25:G25").format = { fill: "#D9EAF7", font: { name: font, size: 10, bold: true, color: "#183B56" } };
overview.getRange("A25:G29").format.wrapText = true;
overview.getRange("A26:A29").format.font = { name: font, size: 10, bold: true, color: "#183B56" };
overview.getRange("A25:A29").format.columnWidth = 18;
overview.getRange("B25:B29").format.columnWidth = 90;
overview.getRange("A26:G29").format.rowHeight = 38;
overview.freezePanes.freezeRows(3);

workbook.recalculate();
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const inspections = [];
for (const [sheetName, range] of [
  ["数据总览", "A1:H29"],
  ["完整数据清单", `A1:Q${masterRows.length + 4}`],
  ["字段字典", `A1:G${schemaRows.length + 4}`],
  ["分阶段计划", `A1:F${phaseRows.length + 4}`],
  ["专家工作量", `A1:G${workloadRows.length + 4}`],
]) {
  const check = await workbook.inspect({ kind: "table", range: `${sheetName}!${range.split("!").pop()}`, include: "values,formulas", tableMaxRows: 8, tableMaxCols: 17, maxChars: 5000 });
  inspections.push({ sheetName, check: check.ndjson });
  const preview = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${sheetName}.png`, new Uint8Array(await preview.arrayBuffer()));
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
});

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.writeFile(`${outputDir}/verification.json`, JSON.stringify({ inspections, errors: errors.ndjson }, null, 2), "utf8");
console.log(JSON.stringify({ outputPath, previewDir, masterRowCount: masterRows.length, errors: errors.ndjson }, null, 2));
