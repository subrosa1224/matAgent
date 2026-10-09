import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const outputDir = "D:/Desktop/matAgent/outputs/expert_seed_templates";
const outputPath = `${outputDir}/expert_first_round_collection_simple.xlsx`;
const previewDir = `${outputDir}/simple_previews`;

const workbook = Workbook.create();
const fontName = "Arial";
const textColor = "#222222";
const gray = "#E7E7E7";
const darkGray = "#666666";
const lightLine = "#D9D9D9";
const inputFill = "#FFF9E6";

function titleBlock(sheet, title, note) {
  sheet.showGridLines = false;
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A1").format.font = { name: fontName, size: 14, bold: true, color: textColor };
  sheet.getRange("A2").values = [[note]];
  sheet.getRange("A2").format.font = { name: fontName, size: 10, italic: true, color: "#555555" };
}

function setListValidation(sheet, range, values) {
  sheet.getRange(range).dataValidation = { rule: { type: "list", values } };
}

// 1. 填写说明
const guide = workbook.worksheets.add("填写说明");
titleBlock(
  guide,
  "材料筛选项目第一轮信息收集",
  "专家只需要描述科研任务、推荐核心资料和已有判断。项目组负责数据库字段、证据编号和系统格式。",
);

guide.getRange("A4:D4").values = [["顺序", "专家需要做什么", "建议数量", "完成标准"]];
guide.getRange("A5:D7").values = [
  [1, "在“任务问答”中用自然语言回答每个任务的问题", "5 个任务", "每个问题能让不了解课题的人读懂即可"],
  [2, "在“相关资料”中列出最重要的论文或资料", "每个任务至少 5 篇", "写清文件名和为什么重要"],
  [3, "在“已有结论”中填写以前选择或排除过的材料", "每个任务约 10 条；没有就少填", "写明推荐、不推荐或不确定及主要原因"],
];
const guideTable = guide.tables.add("A4:D7", true, "SimpleGuideTable");
guideTable.style = "TableStyleLight1";
guideTable.showBandedRows = false;
guide.getRange("A4:D7").format.font = { name: fontName, size: 10, color: textColor };
guide.getRange("A4:D4").format = {
  fill: gray,
  font: { name: fontName, size: 10, bold: true, color: textColor },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { bottom: { style: "medium", color: darkGray } },
};
guide.getRange("A5:D7").format.wrapText = true;
guide.getRange("A5:D7").format.verticalAlignment = "top";
guide.getRange("A5:D7").format.rowHeight = 54;
[10, 60, 25, 48].forEach((w, i) => guide.getRangeByIndexes(0, i, 8, 1).format.columnWidth = w);

guide.getRange("A10").values = [["暂时不需要专家填写的内容"]];
guide.getRange("A10").format.font = { name: fontName, size: 11, bold: true, color: textColor };
guide.getRange("A11:B15").values = [
  ["不需要", "数据库字段名、JSON、材料数据库编号"],
  ["不需要", "证据编号、逐条页码、可比性等级"],
  ["不需要", "完整候选材料清单"],
  ["不需要", "从空白开始审核系统结果"],
  ["项目组处理", "文献结构化、候选生成、证据定位和数据格式转换"],
];
guide.getRange("A11:B15").format.font = { name: fontName, size: 10, color: textColor };
guide.getRange("A11:A15").format.font = { name: fontName, size: 10, bold: true, color: textColor };
guide.getRange("A11:B15").format.borders = { insideHorizontal: { style: "thin", color: lightLine } };
guide.getRange("A11:B15").format.rowHeight = 30;

guide.getRange("A18").values = [["之后会怎样"]];
guide.getRange("A18").format.font = { name: fontName, size: 11, bold: true, color: textColor };
guide.getRange("A19:B22").values = [
  ["第二步", "项目组把本文件整理成系统可以读取的数据。"],
  ["第三步", "系统检索数据库和论文，生成候选、证据和初步结论。"],
  ["第四步", "项目组生成已经预填的复核表。"],
  ["专家复核", "专家只需选择同意、不同意或不确定，并修改关键错误。"],
];
guide.getRange("A19:B22").format.font = { name: fontName, size: 10, color: textColor };
guide.getRange("A19:A22").format.font = { name: fontName, size: 10, bold: true, color: textColor };
guide.getRange("A19:B22").format.borders = { insideHorizontal: { style: "thin", color: lightLine } };
guide.getRange("A19:B22").format.rowHeight = 30;

// 2. 任务问答：每个任务使用一个纵向问答块
const tasks = workbook.worksheets.add("任务问答");
titleBlock(tasks, "任务问答", "按照自己的科研语言填写即可。不知道的内容可写“暂不确定”，不需要使用统计或数据库术语。");
tasks.getRange("A:A").format.columnWidth = 42;
tasks.getRange("B:B").format.columnWidth = 92;

const taskQuestions = [
  "任务简称",
  "1. 你真正想解决的科研问题是什么？",
  "2. 这个结果最终用于什么场景？",
  "3. 想筛选哪一类材料？范围可以写宽一些。",
  "4. 候选材料必须满足哪些条件？有数值范围和单位就写，没有也可以。",
  "5. 哪些材料或情况一定要排除？",
  "6. 如果候选很多，你通常优先比较哪些方面？",
  "7. 最终希望系统给出什么结果？",
  "8. 如果以前做过，最后选择了哪些材料？",
  "9. 哪些地方目前仍不确定，或者不能直接下结论？",
  "填写人",
  "填写日期",
];

let startRow = 4;
for (let taskIndex = 1; taskIndex <= 5; taskIndex += 1) {
  const taskId = `TASK-${String(taskIndex).padStart(2, "0")}`;
  tasks.getRange(`A${startRow}:B${startRow}`).values = [[taskId, "请在右侧浅黄色单元格中填写"]];
  tasks.getRange(`A${startRow}:B${startRow}`).format = {
    fill: gray,
    font: { name: fontName, size: 10, bold: true, color: textColor },
    verticalAlignment: "center",
    borders: { bottom: { style: "medium", color: darkGray } },
  };
  tasks.getRange(`A${startRow}:B${startRow}`).format.rowHeight = 28;

  const qRows = taskQuestions.map((q) => [q, null]);
  tasks.getRange(`A${startRow + 1}`).write(qRows);
  const blockEnd = startRow + taskQuestions.length;
  tasks.getRange(`A${startRow + 1}:B${blockEnd}`).format.font = { name: fontName, size: 10, color: textColor };
  tasks.getRange(`A${startRow + 1}:B${blockEnd}`).format.verticalAlignment = "top";
  tasks.getRange(`A${startRow + 1}:B${blockEnd}`).format.wrapText = true;
  tasks.getRange(`A${startRow + 1}:B${blockEnd}`).format.borders = {
    insideHorizontal: { style: "thin", color: lightLine },
    bottom: { style: "thin", color: lightLine },
  };
  tasks.getRange(`A${startRow + 1}:A${blockEnd}`).format.font = { name: fontName, size: 10, bold: true, color: textColor };
  tasks.getRange(`B${startRow + 1}:B${blockEnd}`).format.fill = inputFill;
  tasks.getRange(`A${startRow + 1}:B${blockEnd}`).format.rowHeight = 48;
  tasks.getRange(`B${blockEnd}`).format.numberFormat = "yyyy-mm-dd";
  startRow = blockEnd + 2;
}
tasks.freezePanes.freezeRows(3);
tasks.freezePanes.freezeColumns(1);

// 3. 相关资料
const papers = workbook.worksheets.add("相关资料");
titleBlock(papers, "相关资料", "每个任务先列 5 篇最重要的论文或资料。PDF 和补充材料文件单独放在文件夹中，本表只记录名称和用途。");
const paperHeaders = ["任务编号", "论文或资料名称*", "交付文件名*", "DOI 或链接（可不填）", "为什么与任务相关？*", "希望系统重点提取什么？*", "有全文吗？", "有补充材料吗？", "其他说明"];
const paperRows = [];
for (let taskIndex = 1; taskIndex <= 5; taskIndex += 1) {
  const taskId = `TASK-${String(taskIndex).padStart(2, "0")}`;
  for (let i = 0; i < 5; i += 1) paperRows.push([taskId, null, null, null, null, null, null, null, null]);
}
papers.getRange("A4").write([paperHeaders, ...paperRows]);
const paperEnd = 4 + paperRows.length;
const paperTable = papers.tables.add(`A4:I${paperEnd}`, true, "SimplePaperTable");
paperTable.style = "TableStyleLight1";
paperTable.showBandedRows = false;
papers.getRange(`A4:I${paperEnd}`).format.font = { name: fontName, size: 10, color: textColor };
papers.getRange("A4:I4").format = {
  fill: gray,
  font: { name: fontName, size: 10, bold: true, color: textColor },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: darkGray } },
};
papers.getRange(`A5:I${paperEnd}`).format.verticalAlignment = "top";
papers.getRange(`A5:I${paperEnd}`).format.borders = { insideHorizontal: { style: "thin", color: lightLine } };
papers.getRange(`B5:I${paperEnd}`).format.fill = inputFill;
papers.getRange(`A5:I${paperEnd}`).format.rowHeight = 28;
[14, 42, 28, 28, 48, 42, 16, 18, 30].forEach((w, i) => papers.getRangeByIndexes(0, i, paperEnd, 1).format.columnWidth = w);
setListValidation(papers, `G5:G${paperEnd}`, ["有", "没有", "不确定"]);
setListValidation(papers, `H5:H${paperEnd}`, ["有", "没有", "不确定"]);
papers.freezePanes.freezeRows(4);
papers.freezePanes.freezeColumns(1);

// 4. 已有结论
const decisions = workbook.worksheets.add("已有结论");
titleBlock(decisions, "已有人工结论", "填写以前真实选择、排除或暂时无法判断的材料。没有完整候选表时，只填最关键的即可。");
const decisionHeaders = ["任务编号", "材料名称或化学式*", "晶型或结构（不清楚可不填）", "你的判断*", "最主要的原因*", "对应论文或资料文件名", "目前还缺什么证据？", "其他说明"];
const decisionRows = [];
for (let taskIndex = 1; taskIndex <= 5; taskIndex += 1) {
  const taskId = `TASK-${String(taskIndex).padStart(2, "0")}`;
  for (let i = 0; i < 10; i += 1) decisionRows.push([taskId, null, null, null, null, null, null, null]);
}
decisions.getRange("A4").write([decisionHeaders, ...decisionRows]);
const decisionEnd = 4 + decisionRows.length;
const decisionTable = decisions.tables.add(`A4:H${decisionEnd}`, true, "SimpleDecisionTable");
decisionTable.style = "TableStyleLight1";
decisionTable.showBandedRows = false;
decisions.getRange(`A4:H${decisionEnd}`).format.font = { name: fontName, size: 10, color: textColor };
decisions.getRange("A4:H4").format = {
  fill: gray,
  font: { name: fontName, size: 10, bold: true, color: textColor },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  wrapText: true,
  borders: { bottom: { style: "medium", color: darkGray } },
};
decisions.getRange(`A5:H${decisionEnd}`).format.verticalAlignment = "top";
decisions.getRange(`A5:H${decisionEnd}`).format.borders = { insideHorizontal: { style: "thin", color: lightLine } };
decisions.getRange(`B5:H${decisionEnd}`).format.fill = inputFill;
decisions.getRange(`A5:H${decisionEnd}`).format.rowHeight = 28;
[14, 28, 30, 16, 48, 32, 38, 30].forEach((w, i) => decisions.getRangeByIndexes(0, i, decisionEnd, 1).format.columnWidth = w);
setListValidation(decisions, `D5:D${decisionEnd}`, ["推荐", "不推荐", "不确定"]);
decisions.freezePanes.freezeRows(4);
decisions.freezePanes.freezeColumns(1);

workbook.recalculate();
await fs.mkdir(outputDir, { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const sheetNames = ["填写说明", "任务问答", "相关资料", "已有结论"];
const inspections = [];
for (const sheetName of sheetNames) {
  const inspect = await workbook.inspect({ kind: "region", sheetId: sheetName, range: "A1:J25", maxChars: 5000 });
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
await fs.writeFile(`${outputDir}/simple_verification.json`, JSON.stringify({ inspections, errors: errors.ndjson }, null, 2), "utf8");

console.log(JSON.stringify({ outputPath, previewDir, sheetNames, errors: errors.ndjson }, null, 2));
