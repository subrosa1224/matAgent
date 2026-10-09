import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/Desktop/matAgent/outputs/data_collection_plan";
const outputPath = `${outputDir}/material_agent_data_requirements_simple.xlsx`;
const previewPath = `${outputDir}/material_agent_data_requirements_simple.png`;

const headers = [
  "数据类型",
  "具体需要哪些方面",
  "最低可运行数量",
  "推荐固定目标",
  "可以从哪里获得",
  "材料专家需要做什么",
  "用于项目哪个部分",
  "第一批是否需要",
];

const rows = [
  [
    "1. 科研任务与筛选设定",
    "科研问题\n应用场景\n材料范围\n硬筛选条件\n软排序条件\n属性单位\n缺失值处理\n排除规则\n停止条件\n预期输出",
    "12个任务",
    "30个任务\n其中12个做深度标注",
    "专业人员做过的课题\n公开文献中的筛选案例\n合作团队历史任务",
    "确认任务是否真实\n确定筛选条件和科研边界\n说明哪些情况不能下结论",
    "主控Agent\n任务规划\n端到端评价",
    "是",
  ],
  [
    "2. 数据库快照",
    "数据库名称\n数据库版本\n查询日期\n查询参数\n属性定义\n属性单位\n原始返回记录\n缺失字段说明",
    "12份",
    "30份\n每个任务1份",
    "Materials Project\n其他公开材料数据库\n项目本地数据库",
    "确认属性含义\n确认属性是否适合当前筛选任务",
    "数据库Agent\n筛选复现\n属性校验",
    "是",
  ],
  [
    "3. 初始候选材料",
    "任务ID\n材料ID\n化学式\n材料名称\n晶型\n空间群\n关键属性\n数据来源\n数据库版本\n初始排序",
    "300条候选记录",
    "1,500条候选记录\n平均50条/任务",
    "数据库查询结果\n专家历史候选清单",
    "审核异常候选和身份边界\n不需要手工录入数据库已有字段",
    "数据库筛选\n候选漏斗\n召回率评价",
    "是",
  ],
  [
    "4. 专家最终选择",
    "保留/排除/待定\n推荐等级或排序\n选择理由\n违反的条件\n采用的证据\n置信度\n后续验证建议",
    "300条判断\n覆盖全部初始候选",
    "1,500条判断\n覆盖全部初始候选",
    "材料专家针对初始候选独立判断",
    "逐项确认最终状态\n说明保留、排除或待定的关键原因",
    "最终结果金标\n候选排序\nSFT训练",
    "是",
  ],
  [
    "5. 相关论文与补充材料",
    "PDF全文\n补充材料\nDOI\n题名\n作者\n年份\n期刊\n对应任务\n对应材料\n论文作用\n使用许可",
    "60篇唯一论文\n120条任务—论文关系",
    "220篇唯一论文\n330条任务—论文关系",
    "专业人员推荐\n公开论文库\n项目已有文献",
    "确认论文是否真正相关\n区分直接证据、背景和负例",
    "文献Agent\n材料领域RAG",
    "是",
  ],
  [
    "6. 结论与证据标注",
    "科研结论\n对应材料身份\n支持或反对\n证据原文\n页码/段落\n表格或图编号\n证据类型\n证据强度\n实验或计算\n适用条件",
    "300条正证据\n300条困难负例",
    "1,200条正证据\n1,200条困难负例",
    "论文正文\n补充材料\n固定数据库记录",
    "确认原文是否真正支持结论\n审核高相似但错误的困难负例",
    "增强RAG\nClaim–Evidence账本\n事实核查",
    "是",
  ],
  [
    "7. 材料身份数据",
    "规范名称\n化学式\n材料ID\n晶型\n空间群\n别名\n结构文件\n掺杂情况\n缺陷情况\n不同身份能否合并",
    "200个身份实体\n100对易混淆案例",
    "700个身份实体\n350对易混淆案例",
    "材料数据库\nCIF结构文件\n论文\n专家知识",
    "确认别名和晶型对应关系\n审核同化学式不同物相等困难案例",
    "身份感知RAG\n候选去重\n身份校验",
    "是",
  ],
  [
    "8. 实验条件数据",
    "材料和样品身份\n制备方法\n原料\n温度\n压力\n时间\n气氛\n样品形态\n测试方法\n测试温度\n测量值和单位\n证据位置",
    "200条结果—条件记录",
    "1,000条结果—条件记录",
    "实验论文\n论文补充材料\n用户或合作团队实验数据",
    "确认哪些条件会影响材料身份、数据可比性和筛选结论",
    "条件感知RAG\n合成可行性\n实验建议",
    "是",
  ],
  [
    "9. 计算条件数据",
    "材料身份\n计算软件\n交换关联泛函\nU值\n赝势\nk点\n截断能\n磁性设置\n参考相\n计算结果和单位\n数据库版本",
    "100条结果—条件记录",
    "500条结果—条件记录",
    "计算论文\n数据库方法说明\n补充材料",
    "确认不同计算方法的数据能否比较\n指出必须保留的关键参数",
    "计算事实核查\n数据可比性\n结论降级",
    "涉及计算证据时需要",
  ],
  [
    "10. 冲突与可比性案例",
    "两条待比较记录\n是否同一材料\n实验或计算方法差异\n温度压力差异\n是否真正冲突\n可比等级\n不能比较的原因\n推荐处理方式",
    "30对冲突案例\n100对可比性判断",
    "150对冲突案例\n350对可比性判断",
    "不同论文结果\n数据库与实验结果\n历史争议案例",
    "判断直接可比、有条件可比、不可比或信息不足",
    "事实核查\n冲突检测\n数据分析Agent",
    "第二批需要",
  ],
  [
    "11. 物理化学与事实规则",
    "规则名称\n适用材料范围\n输入字段\n触发条件\n判断逻辑\n通过/阻断/降级动作\n例外情况\n严重程度\n规则依据\n版本",
    "30条规则\n每条5个通过和5个违反案例",
    "54条规则\n270个通过案例\n270个违反案例",
    "专家经验\n数据库说明\n标准\n方法论文\n综述",
    "制定或审核规则\n确认适用范围、例外和错误处理动作",
    "物理化学校验器\n领域事实核查",
    "第二批需要",
  ],
  [
    "12. 机理与因果关系",
    "原因或过程\n作用方向\n结构变化\n性质变化\n性能结果\n成立条件\n证据位置\n关系强度\n因果等级\n争议状态",
    "150条关系",
    "500条关系",
    "机理论文\n综述\n控制变量实验\n专家判断",
    "审核关系方向和成立条件\n区分实验干预、相关、机理解释和推测",
    "显式推理图\n机理约束\n因果边界控制",
    "第二批需要",
  ],
  [
    "13. 推理图、错误轨迹与偏好",
    "任务步骤\n步骤依赖\n工具调用\n候选变化\n使用证据\n错误步骤\n错误类型\n正确修正\n输出A/B\n专家偏好及理由",
    "6张推理图\n120条系统轨迹\n60条修正轨迹\n300对偏好",
    "12张推理图\n800条系统轨迹\n240条修正轨迹\n1,200对偏好",
    "系统批量生成轨迹\n专家解决任务的步骤\n系统不同版本输出",
    "审核标准推理图和高风险轨迹\n指出第一个关键错误\n判断两个输出哪个更好",
    "STaR原型\nSFT\n可选偏好优化或RL",
    "训练阶段需要",
  ],
  [
    "14. 历史结构化结果与微调样本",
    "原始论文输入\n模型初稿\n人工修改版\n最终正确版\n修改字段\n修改原因\n任务指令\n上下文\n工具调用\n结构化标准输出\n质量状态",
    "100组三联数据\n1,500条SFT样本",
    "500组三联数据\n6,500条SFT样本",
    "合作团队以前的文献结构化任务\n本项目金标和系统轨迹",
    "确认最终版本\n高风险训练样本全部审核\n普通样本抽查20%–30%",
    "文献抽取微调\n工具调用微调\n报告生成微调",
    "训练阶段需要",
  ],
  [
    "15. 数值数据与实验设计",
    "样本ID\n材料ID\n分组或处理条件\n属性名称\n数值\n单位\n数据来源\n实验/计算\n批次\n独立/配对/重复测量\n对照组\n主要指标\n异常和缺失说明",
    "8份数据集\n每份1份设计说明",
    "20份数据集\n每份1份设计说明",
    "用户真实实验CSV\n公开数据集\n论文补充材料\n材料数据库\n模拟边界数据",
    "确认每一行代表什么\n确认实验设计和允许的统计推断",
    "数据分析Agent\n统计分析\n候选属性分析",
    "有数据分析任务时需要",
  ],
  [
    "16. 独立测试、报告与授权",
    "冻结测试任务\n测试论文和候选\n专家标准结果\n专家标准报告\n数据提供者\n数据来源\n使用许可\n是否允许训练\n是否允许发表\n版本与审核日期",
    "3个独立测试任务\n6份专家报告\n每批数据1份授权记录",
    "6个独立测试任务\n12份专家报告\n按10个交付批次管理授权",
    "未进入训练的新任务\n专家历史报告\n数据提供方授权说明",
    "对测试任务进行盲审\n确认数据允许的使用范围",
    "最终评价\n报告评价\n防止数据泄漏\n合规复现",
    "是",
  ],
];

const workbook = Workbook.create();
const sheet = workbook.worksheets.add("数据需求表");
sheet.showGridLines = false;

sheet.getRange("A1").values = [["材料多智能体项目数据需求表"]];
sheet.getRange("A1").format.font = { name: "Arial", size: 15, bold: true, color: "#222222" };
sheet.getRange("A2").values = [["数量按“核心RAG完整实现、校验闭环、推理与STaR原型”计算。最低数量用于原型，推荐数量用于论文级建设。"]];
sheet.getRange("A2").format.font = { name: "Arial", size: 10, italic: true, color: "#555555" };

sheet.getRange("A4").write([headers, ...rows]);
const endRow = rows.length + 4;
const table = sheet.tables.add(`A4:H${endRow}`, true, "SimpleDataRequirementsTable");
table.style = "TableStyleLight1";
table.showBandedRows = false;

sheet.getRange(`A4:H${endRow}`).format.font = { name: "Arial", size: 10, color: "#222222" };
sheet.getRange(`A4:H${endRow}`).format.verticalAlignment = "top";
sheet.getRange(`A4:H${endRow}`).format.wrapText = true;
sheet.getRange("A4:H4").format = {
  fill: "#E7E7E7",
  font: { name: "Arial", size: 10, bold: true, color: "#111111" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { bottom: { style: "medium", color: "#666666" } },
};
sheet.getRange(`A5:H${endRow}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#D9D9D9" },
  bottom: { style: "thin", color: "#D9D9D9" },
};
sheet.getRange(`A5:A${endRow}`).format.font = { name: "Arial", size: 10, bold: true, color: "#222222" };
sheet.getRange(`H5:H${endRow}`).format.horizontalAlignment = "center";

const widths = [24, 48, 22, 25, 28, 34, 27, 18];
widths.forEach((width, index) => {
  sheet.getRangeByIndexes(0, index, endRow, 1).format.columnWidth = width;
});
sheet.getRange(`A5:H${endRow}`).format.rowHeight = 128;
sheet.getRange("A4:H4").format.rowHeight = 30;
sheet.freezePanes.freezeRows(4);
sheet.freezePanes.freezeColumns(1);

workbook.recalculate();
await fs.mkdir(outputDir, { recursive: true });

const inspect = await workbook.inspect({
  kind: "table",
  range: `数据需求表!A1:H${endRow}`,
  include: "values,formulas",
  tableMaxRows: 20,
  tableMaxCols: 8,
  maxChars: 16000,
});

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});

const preview = await workbook.render({
  sheetName: "数据需求表",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
await fs.writeFile(
  `${outputDir}/material_agent_data_requirements_simple_verification.json`,
  JSON.stringify({ inspect: inspect.ndjson, errors: errors.ndjson }, null, 2),
  "utf8",
);

console.log(JSON.stringify({ outputPath, previewPath, rowCount: rows.length, errors: errors.ndjson }, null, 2));
