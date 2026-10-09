import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const benchmarkDir = path.join(root, "tests", "fixtures", "research", "benchmark_v1");
const outputDir = path.join(root, "outputs", "ra0-crystalline-inorganic");

const readJson = async (...parts) =>
  JSON.parse(await fs.readFile(path.join(benchmarkDir, ...parts), "utf8"));

const screeningCases = await readJson("screening_cases", "cases.json");
const baseline = await readJson("expected", "current_baseline.json");
const identityCases = await readJson("identity_cases", "cases.json");
const literature = await readJson("literature_evidence", "index.json");
const literatureCandidates = await readJson("literature_evidence", "candidates.json");
const endToEndTasks = await readJson("end_to_end_tasks", "tasks.json");

const workbook = Workbook.create();
const instructions = workbook.worksheets.add("使用说明");
const screening = workbook.worksheets.add("筛选用例");
const identity = workbook.worksheets.add("身份复核");
const literatureSheet = workbook.worksheets.add("文献复核");
const tasksSheet = workbook.worksheets.add("端到端任务");
const signoff = workbook.worksheets.add("签字确认");

const colors = {
  navy: "#17365D",
  blue: "#D9EAF7",
  paleBlue: "#EEF5FB",
  green: "#E2F0D9",
  amber: "#FFF2CC",
  red: "#FCE4D6",
  gray: "#E7E6E6",
  text: "#1F2937",
  white: "#FFFFFF",
  border: "#B4C6D7",
};

function title(sheet, range, value) {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range);
  cell.values = [[value]];
  cell.format = {
    fill: colors.navy,
    font: { bold: true, color: colors.white, size: 16 },
    verticalAlignment: "center",
  };
  cell.format.rowHeight = 34;
}

function header(range) {
  range.format = {
    fill: colors.blue,
    font: { bold: true, color: colors.text },
    borders: { preset: "outside", style: "thin", color: colors.border },
    wrapText: true,
    verticalAlignment: "center",
  };
  range.format.rowHeight = 30;
}

function body(range) {
  range.format = {
    font: { color: colors.text, size: 10 },
    borders: {
      insideHorizontal: { style: "thin", color: "#DCE6F1" },
      bottom: { style: "thin", color: colors.border },
    },
    verticalAlignment: "top",
    wrapText: true,
  };
}

function setWidths(sheet, widths) {
  for (const [column, width] of Object.entries(widths)) {
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
  }
}

function addStatusFormatting(range) {
  range.conditionalFormats.add("containsText", {
    text: "通过",
    format: { fill: colors.green, font: { color: "#375623" } },
  });
  range.conditionalFormats.add("containsText", {
    text: "待复核",
    format: { fill: colors.amber, font: { color: "#7F6000" } },
  });
  range.conditionalFormats.add("containsText", {
    text: "不通过",
    format: { fill: colors.red, font: { color: "#9C0006" } },
  });
}

for (const sheet of [instructions, screening, identity, literatureSheet, tasksSheet, signoff]) {
  sheet.showGridLines = false;
}

// 使用说明与动态摘要
title(instructions, "A1:H1", "RA-0 晶态无机材料筛选基准｜专业复核工作簿");
instructions.getRange("A3:H7").values = [
  ["用途", "把程序自动生成的候选、身份边界、文献证据和端到端答案转成可审计的科研金标。", null, null, null, null, null, null],
  ["填写对象", "晶态无机材料方向研究人员；数据库逻辑用例可由第二位独立复核人确认。", null, null, null, null, null, null],
  ["填写原则", "只确认有证据支持的内容；不确定时选择“需补证据”，不要凭经验补全。", null, null, null, null, null, null],
  ["最小完成条件", "10 个多晶型身份边界完成专业复核；有效文献卷宗不少于 8 篇；3 个端到端任务均有专业结论。", null, null, null, null, null, null],
  ["当前状态", "RA-0 进行中：数据与自动化层已建立，科研金标尚未完成专业签字。", null, null, null, null, null, null],
];
for (let row = 3; row <= 7; row += 1) instructions.getRange(`B${row}:H${row}`).merge();
instructions.getRange("A3:A7").format = { fill: colors.paleBlue, font: { bold: true, color: colors.navy }, wrapText: true };
body(instructions.getRange("A3:H7"));

instructions.getRange("A9:B9").values = [["复核指标", "当前值"]];
header(instructions.getRange("A9:B9"));
instructions.getRange("A10:A16").values = [
  ["筛选用例总数"],
  ["自动筛选精确通过"],
  ["身份边界待专业复核"],
  ["身份边界已通过"],
  ["有效文献卷宗"],
  ["端到端任务已通过"],
  ["签字项已完成"],
];
instructions.getRange("B10:B16").formulas = [
  ["=COUNTA('筛选用例'!$A$4:$A$18)"],
  ["=COUNTIF('筛选用例'!$J$4:$J$18,\"通过\")"],
  ["=COUNTIF('身份复核'!$J$4:$J$23,\"待复核\")"],
  ["=COUNTIF('身份复核'!$J$4:$J$23,\"通过\")"],
  ["=COUNTIF('文献复核'!$H$4:$H$11,\"通过\")"],
  ["=COUNTIF('端到端任务'!$H$4:$H$6,\"通过\")"],
  ["=COUNTIF('签字确认'!$E$4:$E$7,\"已确认\")"],
];
body(instructions.getRange("A10:B16"));
instructions.getRange("B10:B16").format.numberFormat = "0";
instructions.getRange("D9:H9").merge();
instructions.getRange("D9:H9").values = [["完成判定（由表内填写结果自动更新）"]];
header(instructions.getRange("D9:H9"));
instructions.getRange("D10:H13").merge(true);
instructions.getRange("D10:D13").formulas = [
  ["=IF(B13=20,\"身份边界：已完成\",\"身份边界：还需 \"&(20-B13)&\" 项通过\")"],
  ["=IF(B14>=8,\"文献卷宗：达到最低数量\",\"文献卷宗：还缺 \"&(8-B14)&\" 篇有效证据\")"],
  ["=IF(B15=3,\"端到端任务：已完成\",\"端到端任务：尚未完成\")"],
  ["=IF(AND(B13=20,B14>=8,B15=3,B16=4),\"RA-0 可申请关闭\",\"RA-0 暂不可关闭\")"],
];
body(instructions.getRange("D10:H13"));
addStatusFormatting(instructions.getRange("D10:H13"));
instructions.freezePanes.freezeRows(1);
setWidths(instructions, { A: 20, B: 17, C: 3, D: 19, E: 18, F: 18, G: 18, H: 18 });

// 筛选用例：自动层基线，仅需第二人确认逻辑。
title(screening, "A1:K1", "筛选用例与当前程序基线");
screening.getRange("A2:K2").merge();
screening.getRange("A2:K2").values = [["说明：这里验证数据库过滤结果是否可重复；“排序支持”单独列出，不能因过滤正确就视为完整科研任务已完成。"]];
screening.getRange("A2:K2").format = { fill: colors.amber, wrapText: true, font: { color: colors.text } };
screening.getRange("A3:K3").values = [["用例", "科研问题", "约束条件", "排序字段", "方向", "限制数", "预期命中", "实际命中", "过滤状态", "复核结论", "复核备注"]];
header(screening.getRange("A3:K3"));
const baselineById = new Map(baseline.cases.map((item) => [item.case_id, item]));
const screeningRows = screeningCases.map((item) => {
  const current = baselineById.get(item.case_id);
  return [
    item.case_id,
    item.question,
    JSON.stringify(item.constraints),
    item.sort.field,
    item.sort.direction,
    item.limit,
    current.expected_match_count ?? null,
    current.actual_match_count ?? null,
    current.status === "evaluated" && current.filter_exact ? "精确" : `不支持：${(current.unsupported_constraints ?? []).join("、")}`,
    current.status === "evaluated" && current.filter_exact ? "通过" : "待复核",
    current.task_specific_sort_supported ? "已支持指定排序" : "当前尚不支持任务指定排序",
  ];
});
screening.getRange("A4:K18").values = screeningRows;
body(screening.getRange("A4:K18"));
screening.getRange("F4:H18").format.numberFormat = "0";
screening.getRange("J4:J18").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过"] } };
addStatusFormatting(screening.getRange("J4:J18"));
screening.freezePanes.freezeRows(3);
screening.freezePanes.freezeColumns(1);
setWidths(screening, { A: 13, B: 42, C: 55, D: 24, E: 12, F: 10, G: 12, H: 12, I: 20, J: 14, K: 34 });

// 身份边界：程序规则与材料学判断分开记录。
title(identity, "A1:L1", "材料身份与多晶型边界复核");
identity.getRange("A2:L2").merge();
identity.getRange("A2:L2").values = [["关键判断：同一化学式但空间群不同，不应自动当作同一物相；专业人员需确认是否可作为“不同相/不同结构记录”的金标边界。"]];
identity.getRange("A2:L2").format = { fill: colors.amber, wrapText: true, font: { color: colors.text } };
identity.getRange("A3:L3").values = [["用例", "化学式/体系", "左侧材料 ID", "左空间群", "右侧材料 ID", "右空间群", "预期关系", "自动状态", "专业判断", "复核结论", "证据或理由", "复核人"]];
header(identity.getRange("A3:L3"));
identity.getRange("A4:L23").values = identityCases.map((item) => [
  item.case_id,
  item.formula ?? item.chemsys ?? "同一来源记录",
  item.left_material_id,
  item.left_spacegroup ?? null,
  item.right_material_id,
  item.right_spacegroup ?? null,
  item.expected_relationship,
  item.review_status === "deterministic" ? "规则确定" : "需材料专业复核",
  item.review_status === "deterministic" ? "规则边界明确" : "请判断两条记录是否应视为不同物相/结构",
  item.review_status === "deterministic" ? "通过" : "待复核",
  "",
  "",
]);
body(identity.getRange("A4:L23"));
identity.getRange("D4:D23").format.numberFormat = "0";
identity.getRange("F4:F23").format.numberFormat = "0";
identity.getRange("I4:I23").dataValidation = { rule: { type: "list", values: ["不同物相/结构", "同一物相重复记录", "证据不足"] } };
identity.getRange("J4:J23").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据"] } };
addStatusFormatting(identity.getRange("J4:J23"));
identity.freezePanes.freezeRows(3);
identity.freezePanes.freezeColumns(2);
setWidths(identity, { A: 13, B: 18, C: 17, D: 12, E: 17, F: 12, G: 28, H: 20, I: 29, J: 15, K: 42, L: 16 });

// 文献复核：保留 5 个已有卷宗，并显式显示 3 个缺口。
title(literatureSheet, "A1:L1", "文献证据与任务可比性复核");
literatureSheet.getRange("A2:L2").merge();
literatureSheet.getRange("A2:L2").values = [["最低目标为 8 个有效卷宗。已批准卷宗不等于适用于当前任务；仍需核对材料身份、物相、测量条件和结论边界。黄色行是必须补齐的证据缺口。"]];
literatureSheet.getRange("A2:L2").format = { fill: colors.amber, wrapText: true, font: { color: colors.text } };
literatureSheet.getRange("A3:L3").values = [["序号", "文献/卷宗 ID", "标题", "卷宗路径或来源", "拟支持任务", "材料与物相匹配", "条件可比性", "复核结论", "可支持的最强结论", "缺口/冲突", "复核人", "复核日期"]];
header(literatureSheet.getRange("A3:L3"));
const literatureRows = literature.map((item, index) => [
  index + 1,
  item.document_id,
  item.title,
  item.dossier_path,
  index === 0 ? "e2e-a-wide-gap-oxides" : "待指定",
  "待核对",
  "待核对",
  "待复核",
  "",
  "已有卷宗已批准，但尚未完成本任务可比性审查",
  "",
  null,
]);
for (const [offset, candidate] of literatureCandidates.entries()) {
  literatureRows.push([
    literature.length + offset + 1,
    candidate.candidate_id,
    candidate.title,
    candidate.local_pdf_path ?? candidate.open_full_text_url,
    candidate.task_ids.join("；"),
    "待核对",
    "待核对",
    "待复核",
    "",
    `${candidate.dossier_candidate_path ? "本地 PDF 和银标档案已生成，待专业审核。" : "已登记开放全文线索，尚未生成档案。"}${candidate.screening_note}`,
    "",
    null,
  ]);
}
literatureSheet.getRange("A4:L11").values = literatureRows;
body(literatureSheet.getRange("A4:L11"));
literatureSheet.getRange("A4:A11").format.numberFormat = "0";
literatureSheet.getRange("L4:L11").format.numberFormat = "yyyy-mm-dd";
literatureSheet.getRange("E4:E11").dataValidation = { rule: { type: "list", values: ["e2e-a-wide-gap-oxides", "e2e-b-polymorphs", "e2e-c-metastable-synthesized", "待指定"] } };
literatureSheet.getRange("F4:G11").dataValidation = { rule: { type: "list", values: ["匹配", "部分匹配", "不匹配", "待核对"] } };
literatureSheet.getRange("H4:H11").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据"] } };
addStatusFormatting(literatureSheet.getRange("H4:H11"));
literatureSheet.freezePanes.freezeRows(3);
literatureSheet.freezePanes.freezeColumns(2);
setWidths(literatureSheet, { A: 9, B: 31, C: 48, D: 48, E: 31, F: 20, G: 18, H: 15, I: 46, J: 38, K: 16, L: 15 });

// 端到端任务：专业人员确认结论、边界和验证计划。
title(tasksSheet, "A1:K1", "端到端科研任务答案审查");
tasksSheet.getRange("A2:K2").merge();
tasksSheet.getRange("A2:K2").values = [["不能只看候选列表。每个任务都应同时审查：结论是否成立、哪些话不能说、证据是否足够、下一步实验/计算是否能真正消除关键不确定性。"]];
tasksSheet.getRange("A2:K2").format = { fill: colors.amber, wrapText: true, font: { color: colors.text } };
tasksSheet.getRange("A3:K3").values = [["任务", "标题", "筛选用例", "候选数", "必须检查的结论边界", "专业期望结论", "建议验证计划", "复核结论", "主要问题", "复核人", "复核日期"]];
header(tasksSheet.getRange("A3:K3"));
tasksSheet.getRange("A4:K6").values = endToEndTasks.map((item) => [
  item.task_id,
  item.title,
  item.screening_case_id,
  item.database_expected.matched_count,
  item.required_claim_checks.join("；"),
  "请填写：推荐层级、关键证据、不可下的结论",
  "请填写：优先验证对象、方法、判定标准、失败后处理",
  "待复核",
  item.literature_seed_document_ids.length === 0 ? "缺少任务特定文献" : "需完成任务可比性审查",
  "",
  null,
]);
body(tasksSheet.getRange("A4:K6"));
tasksSheet.getRange("D4:D6").format.numberFormat = "0";
tasksSheet.getRange("K4:K6").format.numberFormat = "yyyy-mm-dd";
tasksSheet.getRange("H4:H6").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据"] } };
addStatusFormatting(tasksSheet.getRange("H4:H6"));
tasksSheet.freezePanes.freezeRows(3);
tasksSheet.freezePanes.freezeColumns(2);
setWidths(tasksSheet, { A: 31, B: 31, C: 16, D: 12, E: 48, F: 48, G: 50, H: 15, I: 34, J: 16, K: 15 });

// 最终签字：不同职责不可由程序替代。
title(signoff, "A1:G1", "RA-0 关闭前签字确认");
signoff.getRange("A2:G2").merge();
signoff.getRange("A2:G2").values = [["只有下列四项全部确认，且首页自动判定为“RA-0 可申请关闭”时，才能进入下一阶段并把本基准称为科研金标。"]];
signoff.getRange("A2:G2").format = { fill: colors.amber, wrapText: true, font: { color: colors.text } };
signoff.getRange("A3:G3").values = [["签字项", "责任角色", "确认内容", "姓名", "状态", "日期", "备注"]];
header(signoff.getRange("A3:G3"));
signoff.getRange("A4:G7").values = [
  ["筛选规则复核", "独立复核人", "15 个筛选用例的约束与预期结果可重复", "", "待确认", null, ""],
  ["材料身份复核", "材料专业人员", "10 个同式异构/多晶型边界已逐项确认", "", "待确认", null, ""],
  ["文献证据复核", "材料专业人员", "至少 8 个卷宗满足身份、物相和条件可比性要求", "", "待确认", null, ""],
  ["端到端答案复核", "课题负责人", "3 个任务的结论、限制与验证计划均可作为金标", "", "待确认", null, ""],
];
body(signoff.getRange("A4:G7"));
signoff.getRange("E4:E7").dataValidation = { rule: { type: "list", values: ["已确认", "待确认", "不通过"] } };
signoff.getRange("F4:F7").format.numberFormat = "yyyy-mm-dd";
signoff.getRange("E4:E7").conditionalFormats.add("containsText", { text: "已确认", format: { fill: colors.green } });
signoff.getRange("E4:E7").conditionalFormats.add("containsText", { text: "待确认", format: { fill: colors.amber } });
signoff.getRange("E4:E7").conditionalFormats.add("containsText", { text: "不通过", format: { fill: colors.red } });
signoff.freezePanes.freezeRows(3);
setWidths(signoff, { A: 22, B: 20, C: 55, D: 18, E: 15, F: 15, G: 42 });

await fs.mkdir(outputDir, { recursive: true });

const inspections = [];
for (const [sheetName, range] of [
  ["使用说明", "A1:H16"],
  ["筛选用例", "A1:K18"],
  ["身份复核", "A1:L23"],
  ["文献复核", "A1:L11"],
  ["端到端任务", "A1:K6"],
  ["签字确认", "A1:G7"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safeName = sheetName.replaceAll("/", "-");
  await fs.writeFile(path.join(outputDir, `preview-${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
  inspections.push(
    await workbook.inspect({
      kind: "table",
      range: `${sheetName}!${range}`,
      include: "values,formulas",
      tableMaxRows: 8,
      tableMaxCols: 12,
      maxChars: 4000,
    }),
  );
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 4000,
});

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "RA0_晶态无机材料筛选_专业复核工作簿.xlsx");
await output.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  sheetsRendered: inspections.length,
  formulaErrorScan: errors.ndjson,
  summaryInspection: inspections[0].ndjson,
}, null, 2));
