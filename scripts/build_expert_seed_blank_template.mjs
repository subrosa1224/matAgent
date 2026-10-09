import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/Desktop/matAgent/outputs/expert_seed_templates";
const outputPath = `${outputDir}/expert_seed_data_blank_template.xlsx`;
const previewDir = `${outputDir}/previews`;

const workbook = Workbook.create();
const fontName = "Arial";
const headerFill = "#E7E7E7";
const headerText = "#111111";
const bodyText = "#222222";
const lineColor = "#D9D9D9";

function colName(index) {
  let n = index + 1;
  let s = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    s = String.fromCharCode(65 + r) + s;
    n = Math.floor((n - 1) / 26);
  }
  return s;
}

function createInputSheet({ name, title, instruction, headers, blankRows, widths, freezeColumns = 2, tableName }) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;

  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: fontName, size: 14, bold: true, color: bodyText };
  sheet.getRange("A2").values = [[instruction]];
  sheet.getRange("A2").format.font = { name: fontName, size: 10, italic: true, color: "#555555" };

  const rows = [headers, ...Array.from({ length: blankRows }, () => Array(headers.length).fill(null))];
  sheet.getRange("A4").write(rows);
  const lastCol = colName(headers.length - 1);
  const endRow = 4 + blankRows;
  const table = sheet.tables.add(`A4:${lastCol}${endRow}`, true, tableName);
  table.style = "TableStyleLight1";
  table.showBandedRows = false;

  sheet.getRange(`A4:${lastCol}${endRow}`).format.font = { name: fontName, size: 10, color: bodyText };
  sheet.getRange(`A4:${lastCol}${endRow}`).format.verticalAlignment = "top";
  sheet.getRange(`A4:${lastCol}${endRow}`).format.wrapText = false;
  sheet.getRange(`A4:${lastCol}4`).format = {
    fill: headerFill,
    font: { name: fontName, size: 10, bold: true, color: headerText },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
    borders: { bottom: { style: "medium", color: "#777777" } },
  };
  sheet.getRange(`A5:${lastCol}${endRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: lineColor },
    bottom: { style: "thin", color: lineColor },
  };
  sheet.getRange(`A5:${lastCol}${endRow}`).format.rowHeight = 24;
  sheet.getRange(`A4:${lastCol}4`).format.rowHeight = 42;
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, endRow, 1).format.columnWidth = width;
  });
  sheet.freezePanes.freezeRows(4);
  if (freezeColumns > 0) sheet.freezePanes.freezeColumns(freezeColumns);
  return { sheet, endRow, lastCol };
}

function setListValidation(sheet, range, values) {
  sheet.getRange(range).dataValidation = { rule: { type: "list", values } };
}

// 1. 填写说明
const guide = workbook.worksheets.add("填写说明");
guide.showGridLines = false;
guide.getRange("A1").values = [["材料筛选项目第一轮专家数据模板"]];
guide.getRange("A1").format.font = { name: fontName, size: 15, bold: true, color: bodyText };
guide.getRange("A2").values = [["用于收集 5 个真实端到端任务。专家先独立提供科研问题、筛选标准、核心文献和已有人工结论；项目组随后完成结构化、数据库查询与系统运行。"]];
guide.getRange("A2").format.font = { name: fontName, size: 10, italic: true, color: "#555555" };

const guideHeaders = ["填写顺序", "工作表", "需要填写什么", "建议数量", "是否必须", "说明"];
const guideRows = [
  [1, "科研任务", "真实科研问题、材料范围、边界和预期输出", "5 个任务", "必须", "一行一个任务；任务编号建议使用 TASK-01 至 TASK-05。"],
  [2, "筛选条件", "硬条件、排序条件、排除条件、单位和缺失值处理", "每个任务 5–10 条", "必须", "一行一条条件；无法确定时填写“待确认”，不要自行猜测。"],
  [3, "论文目录", "每个任务的核心论文及其作用", "每个任务至少 5 篇", "必须", "优先提供全文和补充材料；没有 DOI 时可填写题名和文件名。"],
  [4, "候选判断", "已有人工保留、排除或待定结果及原因", "每个任务 10–15 个关键候选", "必须", "若暂无完整候选表，只填已有关键候选；项目组后续会预填系统候选供复核。"],
  [5, "关键证据", "支持或反对结论的原文、页码和适用条件", "每个任务 8–10 条", "必须", "证据尽量定位到页码、段落、表格或图片；摘要只能作为背景。"],
  [6, "专家总结", "最终推荐、排除、待定、主要不确定性和后续实验", "每个任务 1 条", "必须", "结论必须写清适用范围和不能下结论的部分。"],
  [7, "可选数据目录", "历史结构化结果、原始实验数据或补充材料数据", "有多少填多少", "可选", "仅填写目录和说明，不要求在此表中粘贴全部原始数据。"],
];
guide.getRange("A4").write([guideHeaders, ...guideRows]);
const guideEnd = 4 + guideRows.length;
const guideTable = guide.tables.add(`A4:F${guideEnd}`, true, "GuideTable");
guideTable.style = "TableStyleLight1";
guideTable.showBandedRows = false;
guide.getRange(`A4:F${guideEnd}`).format.font = { name: fontName, size: 10, color: bodyText };
guide.getRange(`A4:F${guideEnd}`).format.verticalAlignment = "top";
guide.getRange(`A4:F${guideEnd}`).format.wrapText = true;
guide.getRange("A4:F4").format = {
  fill: headerFill,
  font: { name: fontName, size: 10, bold: true, color: headerText },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { bottom: { style: "medium", color: "#777777" } },
};
guide.getRange(`A5:F${guideEnd}`).format.borders = {
  insideHorizontal: { style: "thin", color: lineColor },
  bottom: { style: "thin", color: lineColor },
};
[12, 38, 44, 20, 14, 58].forEach((w, i) => guide.getRangeByIndexes(0, i, guideEnd, 1).format.columnWidth = w);
guide.getRange(`A5:F${guideEnd}`).format.rowHeight = 54;
guide.getRange("A4:F4").format.rowHeight = 28;
guide.freezePanes.freezeRows(4);

guide.getRange("A14").values = [["统一填写规则"]];
guide.getRange("A14").format.font = { name: fontName, size: 11, bold: true, color: bodyText };
guide.getRange("A15:B20").values = [
  ["编号", "任务、论文、候选和证据必须使用稳定编号，后续表之间用编号关联。"],
  ["单位", "数值与单位分开填写；同一属性尽量使用同一单位。"],
  ["未知信息", "不知道时填写“待确认”；不适用时填写“不适用”；不要用 0 代表缺失。"],
  ["材料身份", "同一化学式的不同晶型、空间群、掺杂或缺陷状态应分开记录。"],
  ["独立金标", "第一轮请先给出已有人工选择，不要先看系统结果，以免影响后续评价。"],
  ["授权", "论文和数据需注明是否允许项目内部使用、训练使用和论文发表。"],
];
guide.getRange("A15:B20").format.font = { name: fontName, size: 10, color: bodyText };
guide.getRange("A15:A20").format.font = { name: fontName, size: 10, bold: true, color: bodyText };
guide.getRange("A15:B20").format.wrapText = true;
guide.getRange("A15:B20").format.verticalAlignment = "top";
guide.getRange("A15:B20").format.borders = { insideHorizontal: { style: "thin", color: lineColor } };
guide.getRange("A15:B20").format.rowHeight = 50;

// 2. 科研任务
const task = createInputSheet({
  name: "科研任务",
  title: "科研任务",
  instruction: "每行填写一个真实任务，共 5 个。带 * 的字段为第一轮必填。",
  headers: ["任务编号*", "科研问题*", "应用场景*", "材料范围*", "必须满足*", "必须排除*", "证据要求*", "预期输出*", "停止条件*", "专家补充说明", "提供者*", "版本*", "提交日期*"],
  blankRows: 5,
  widths: [15, 38, 26, 28, 34, 34, 34, 30, 30, 34, 18, 12, 16],
  tableName: "ResearchTaskTable",
});
task.sheet.getRange(`M5:M${task.endRow}`).format.numberFormat = "yyyy-mm-dd";

// 3. 筛选条件
const criteria = createInputSheet({
  name: "筛选条件",
  title: "筛选条件",
  instruction: "每行填写一条筛选或排序规则。上下限不适用时留空；数值和单位分开填写。",
  headers: ["任务编号*", "条件编号*", "条件名称*", "条件类型*", "属性或概念*", "比较关系*", "下限", "上限", "单位", "优先级*", "首选数据来源", "缺失值处理*", "科学依据*", "例外和备注"],
  blankRows: 50,
  widths: [15, 15, 24, 18, 24, 16, 12, 12, 12, 12, 24, 18, 40, 34],
  tableName: "ScreeningCriteriaTable",
});
setListValidation(criteria.sheet, `D5:D${criteria.endRow}`, ["硬条件", "软排序条件", "排除条件"]);
setListValidation(criteria.sheet, `F5:F${criteria.endRow}`, ["等于", "不等于", "大于", "大于等于", "小于", "小于等于", "区间", "包含", "不包含", "专家判断"]);
setListValidation(criteria.sheet, `J5:J${criteria.endRow}`, ["高", "中", "低"]);
setListValidation(criteria.sheet, `L5:L${criteria.endRow}`, ["保留待查", "排除", "降低等级", "不适用"]);
criteria.sheet.getRange(`G5:H${criteria.endRow}`).format.numberFormat = "0.########";

// 4. 论文目录
const papers = createInputSheet({
  name: "论文目录",
  title: "核心论文目录",
  instruction: "每个任务至少 5 篇核心论文。论文全文和补充材料单独作为文件交付，本表记录对应关系。",
  headers: ["任务编号*", "论文编号*", "DOI", "题名*", "年份", "期刊", "对应材料", "论文作用*", "相关程度*", "全文文件名", "补充材料文件名", "入选原因*", "允许内部使用*", "允许训练使用*", "允许论文发表*", "备注"],
  blankRows: 25,
  widths: [15, 15, 24, 44, 10, 24, 24, 18, 18, 28, 28, 36, 18, 18, 18, 28],
  tableName: "PaperCatalogTable",
});
setListValidation(papers.sheet, `H5:H${papers.endRow}`, ["实验合成", "性质测量", "计算预测", "机理研究", "综述背景", "反例或冲突"]);
setListValidation(papers.sheet, `I5:I${papers.endRow}`, ["直接证据", "间接证据", "背景", "困难负例"]);
for (const col of ["M", "N", "O"]) setListValidation(papers.sheet, `${col}5:${col}${papers.endRow}`, ["是", "否", "待确认"]);
papers.sheet.getRange(`E5:E${papers.endRow}`).format.numberFormat = "0";

// 5. 候选判断
const candidates = createInputSheet({
  name: "候选判断",
  title: "候选材料人工判断",
  instruction: "第一轮填写已有关键候选。后续项目组会把系统候选预填到本表，再请专家复核。每行一个明确材料身份。",
  headers: ["任务编号*", "候选编号*", "材料名称", "化学式*", "晶型", "空间群", "数据库编号", "专家状态*", "专家等级", "主要原因*", "满足或违反的条件", "采用的证据编号", "判断置信度*", "后续验证建议", "专家姓名或编号*", "审核日期*"],
  blankRows: 75,
  widths: [15, 16, 24, 16, 18, 16, 20, 14, 14, 40, 32, 28, 16, 34, 20, 16],
  tableName: "CandidateDecisionTable",
});
setListValidation(candidates.sheet, `H5:H${candidates.endRow}`, ["保留", "排除", "待定"]);
setListValidation(candidates.sheet, `I5:I${candidates.endRow}`, ["A", "B", "C", "D", "待定"]);
setListValidation(candidates.sheet, `M5:M${candidates.endRow}`, ["高", "中", "低"]);
candidates.sheet.getRange(`P5:P${candidates.endRow}`).format.numberFormat = "yyyy-mm-dd";

// 6. 关键证据
const evidence = createInputSheet({
  name: "关键证据",
  title: "关键证据",
  instruction: "每行记录一条可核查证据。原文应与页码、表图或段落位置对应，并写明材料身份和条件。",
  headers: ["任务编号*", "证据编号*", "待支持或反对的结论*", "材料名称", "化学式*", "晶型", "论文编号*", "证据原文*", "页码*", "章节或段落", "表格或图片编号", "证据类型*", "证据关系*", "证据强度*", "关键实验或计算条件*", "可比性说明*", "专家确认*", "专家备注"],
  blankRows: 50,
  widths: [15, 15, 38, 22, 16, 18, 15, 52, 12, 24, 20, 16, 16, 14, 40, 34, 16, 30],
  tableName: "EvidenceTable",
});
setListValidation(evidence.sheet, `L5:L${evidence.endRow}`, ["实验", "计算", "数据库", "综述"]);
setListValidation(evidence.sheet, `M5:M${evidence.endRow}`, ["支持", "反对", "部分支持", "证据不足"]);
setListValidation(evidence.sheet, `N5:N${evidence.endRow}`, ["强", "中", "弱"]);
setListValidation(evidence.sheet, `Q5:Q${evidence.endRow}`, ["是", "否", "待确认"]);

// 7. 专家总结
const summary = createInputSheet({
  name: "专家总结",
  title: "任务级专家总结",
  instruction: "每个任务填写一行。重点写清为什么选、为什么不选、还缺什么证据，以及结论边界。",
  headers: ["任务编号*", "推荐候选*", "排除候选*", "待定候选*", "核心证据总结*", "主要不确定性*", "建议的下一步实验或计算*", "总体结论*", "结论适用范围和边界*", "专家姓名或编号*", "审核日期*"],
  blankRows: 5,
  widths: [15, 32, 32, 28, 52, 42, 42, 48, 44, 20, 16],
  freezeColumns: 1,
  tableName: "ExpertSummaryTable",
});
summary.sheet.getRange(`K5:K${summary.endRow}`).format.numberFormat = "yyyy-mm-dd";

// 8. 可选数据目录
const optional = createInputSheet({
  name: "可选数据目录",
  title: "可选历史数据和原始数据目录",
  instruction: "只登记可提供的数据文件及其含义。历史结构化结果、原始实验数据和补充材料数据均可登记。",
  headers: ["任务编号", "数据编号*", "数据类型*", "文件名*", "数据说明*", "记录数", "变量或字段说明", "单位说明", "实验或计算设计", "数据来源*", "允许内部使用*", "允许训练使用*", "允许论文发表*", "备注"],
  blankRows: 20,
  widths: [15, 15, 24, 30, 40, 14, 38, 24, 36, 30, 18, 18, 18, 30],
  tableName: "OptionalDataCatalogTable",
});
setListValidation(optional.sheet, `C5:C${optional.endRow}`, ["历史文献结构化数据", "原始实验数据", "补充材料数据", "其他"]);
for (const col of ["K", "L", "M"]) setListValidation(optional.sheet, `${col}5:${col}${optional.endRow}`, ["是", "否", "待确认"]);
optional.sheet.getRange(`F5:F${optional.endRow}`).format.numberFormat = "#,##0";

workbook.recalculate();
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const sheetNames = ["填写说明", "科研任务", "筛选条件", "论文目录", "候选判断", "关键证据", "专家总结", "可选数据目录"];
const inspections = [];
for (const sheetName of sheetNames) {
  const inspect = await workbook.inspect({
    kind: "region",
    sheetId: sheetName,
    range: "A1:R12",
    maxChars: 4500,
  });
  inspections.push({ sheetName, inspect: inspect.ndjson });
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
await fs.writeFile(
  `${outputDir}/verification.json`,
  JSON.stringify({ inspections, errors: errors.ndjson }, null, 2),
  "utf8",
);

console.log(JSON.stringify({ outputPath, previewDir, sheets: sheetNames, errors: errors.ndjson }, null, 2));
