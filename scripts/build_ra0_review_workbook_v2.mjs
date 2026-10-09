import fs from "node:fs/promises";
import path from "node:path";

import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const benchmarkDir = path.join(root, "tests", "fixtures", "research", "benchmark_v1");
const outputDir = path.join(root, "outputs", "ra0-crystalline-inorganic");

const readJson = async (...parts) =>
  JSON.parse(await fs.readFile(path.join(benchmarkDir, ...parts), "utf8"));
const readProjectJson = async (relativePath) =>
  JSON.parse(await fs.readFile(path.join(root, relativePath), "utf8"));

const screeningCases = await readJson("screening_cases", "cases.json");
const screeningExpected = await readJson("expected", "screening_expected.json");
const baseline = await readJson("expected", "current_baseline.json");
const identityCases = await readJson("identity_cases", "cases.json");
const literature = await readJson("literature_evidence", "index.json");
const literatureCandidates = await readJson("literature_evidence", "candidates.json");
const endToEndTasks = await readJson("end_to_end_tasks", "tasks.json");
const snapshotText = await fs.readFile(
  path.join(benchmarkDir, "database_snapshots", "materials_project_oxide_300.jsonl"),
  "utf8",
);
const snapshot = snapshotText.trim().split(/\r?\n/).map((line) => JSON.parse(line));
const materialById = new Map(snapshot.map((item) => [item.material_id, item]));
const expectedById = new Map(screeningExpected.map((item) => [item.case_id, item]));
const baselineById = new Map(baseline.cases.map((item) => [item.case_id, item]));

function matchesConstraint(record, constraints) {
  for (const [key, expected] of Object.entries(constraints)) {
    if (key === "required_elements") {
      if (!expected.every((element) => (record.elements ?? []).includes(element))) return false;
    } else if (key === "excluded_elements") {
      if (expected.some((element) => (record.elements ?? []).includes(element))) return false;
    } else if (Array.isArray(expected)) {
      const [minimum, maximum] = expected;
      const actual = record[key];
      if (actual === null || actual === undefined) return false;
      if (minimum !== null && minimum !== undefined && actual < minimum) return false;
      if (maximum !== null && maximum !== undefined && actual > maximum) return false;
    } else if (record[key] !== expected) {
      return false;
    }
  }
  return true;
}

function compareRecords(left, right, sort) {
  const leftValue = left[sort.field];
  const rightValue = right[sort.field];
  let comparison = 0;
  if (sort.direction === "target") {
    comparison = Math.abs(leftValue - sort.target) - Math.abs(rightValue - sort.target);
  } else if (typeof leftValue === "string" || typeof rightValue === "string") {
    comparison = String(leftValue).localeCompare(String(rightValue), "en");
  } else {
    comparison = Number(leftValue) - Number(rightValue);
  }
  if (sort.direction === "desc") comparison *= -1;
  return comparison || left.material_id.localeCompare(right.material_id, "en");
}

const independentAuditById = new Map();
for (const item of screeningCases) {
  const expected = expectedById.get(item.case_id);
  const recalculated = snapshot
    .filter((record) => matchesConstraint(record, item.constraints))
    .sort((left, right) => compareRecords(left, right, item.sort));
  const allIds = recalculated.map((record) => record.material_id);
  const topIds = allIds.slice(0, item.limit);
  const exact = JSON.stringify(allIds) === JSON.stringify(expected.expected_all_material_ids)
    && JSON.stringify(topIds) === JSON.stringify(expected.expected_material_ids)
    && allIds.length === expected.matched_count;
  if (!exact) throw new Error(`Independent screening audit mismatch: ${item.case_id}; count=${allIds.length}/${expected.matched_count}; top=${topIds.join(",")}; expected=${expected.expected_material_ids.join(",")}`);
  independentAuditById.set(item.case_id, { count: allIds.length, topIds });
}

const workbook = Workbook.create();
const guide = workbook.worksheets.add("审核导航");
const summary = workbook.worksheets.add("AI预审摘要");
const screening = workbook.worksheets.add("筛选用例");
const candidates = workbook.worksheets.add("候选明细");
const identity = workbook.worksheets.add("身份复核");
const literatureSheet = workbook.worksheets.add("文献复核");
const excerpts = workbook.worksheets.add("证据摘录");
const tasks = workbook.worksheets.add("端到端任务");
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
  range.format.rowHeight = 32;
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

function note(sheet, range, value) {
  sheet.getRange(range).merge();
  const cell = sheet.getRange(range);
  cell.values = [[value]];
  cell.format = {
    fill: colors.amber,
    font: { color: colors.text },
    wrapText: true,
    verticalAlignment: "center",
  };
  cell.format.rowHeight = 34;
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
    text: "待",
    format: { fill: colors.amber, font: { color: "#7F6000" } },
  });
  range.conditionalFormats.add("containsText", {
    text: "不通过",
    format: { fill: colors.red, font: { color: "#9C0006" } },
  });
}

for (const sheet of [guide, summary, screening, candidates, identity, literatureSheet, excerpts, tasks, signoff]) {
  sheet.showGridLines = false;
}

// 审核导航：明确角色、顺序和正确的完成门禁。
title(guide, "A1:H1", "RA-0 晶态无机材料筛选基准｜AI 证据预审版");
guide.getRange("A3:H8").values = [
  ["审核顺序", "1 筛选用例 → 2 身份复核 → 3 文献复核 → 4 端到端任务 → 5 签字确认", null, null, null, null, null, null],
  ["工程复核人", "复核15个筛选合同、参考命中数、Top-K顺序和系统安全停止行为；不要求材料专业背景。", null, null, null, null, null, null],
  ["材料复核人", "只重点审核10个同化学式不同空间群案例，以及文献中的物相、样品和条件可比性。", null, null, null, null, null, null],
  ["课题负责人", "审核3个端到端任务的候选分级、结论边界和最小验证方案。", null, null, null, null, null, null],
  ["填写原则", "必须查看候选明细或证据摘录后再选择结论；信息不足选“需补证据”，不得凭经验补全。", null, null, null, null, null, null],
  ["当前状态", "AI已完成第一轮证据预审，但没有替你签字；RA-0B仍待人工逐项确认，不能称为已完成金标。", null, null, null, null, null, null],
];
for (let row = 3; row <= 8; row += 1) guide.getRange(`B${row}:H${row}`).merge();
guide.getRange("A3:A8").format = {
  fill: colors.paleBlue,
  font: { bold: true, color: colors.navy },
  wrapText: true,
};
body(guide.getRange("A3:H8"));
guide.getRange("A10:B10").values = [["门禁指标", "当前值"]];
header(guide.getRange("A10:B10"));
guide.getRange("A11:A16").values = [
  ["筛选用例用户确认"],
  ["专业身份案例总数"],
  ["专业身份案例通过"],
  ["有效文献用户确认"],
  ["端到端任务通过"],
  ["签字项已确认"],
];
guide.getRange("B11:B16").formulas = [
  ["=COUNTIF('筛选用例'!$L$4:$L$18,\"通过\")"],
  ["=COUNTIF('身份复核'!$N$4:$N$23,\"需材料专业复核\")"],
  ["=COUNTIFS('身份复核'!$N$4:$N$23,\"需材料专业复核\",'身份复核'!$P$4:$P$23,\"通过\")"],
  ["=COUNTIF('文献复核'!$K$4:$K$11,\"通过\")"],
  ["=COUNTIF('端到端任务'!$J$4:$J$6,\"通过\")"],
  ["=COUNTIF('签字确认'!$E$4:$E$7,\"已确认\")"],
];
body(guide.getRange("A11:B16"));
guide.getRange("B11:B16").format.numberFormat = "0";
guide.getRange("D10:H10").merge();
guide.getRange("D10:D10").values = [["完成判定（由人工结论自动汇总）"]];
header(guide.getRange("D10:H10"));
guide.getRange("D11:H15").merge(true);
guide.getRange("D11:D15").formulas = [
  ["=IF(B11=15,\"筛选规则：15/15 已复核\",\"筛选规则：还需 \"&(15-B11)&\" 项通过\")"],
  ["=IF(B13=B12,\"专业身份：已完成\",\"专业身份：还需 \"&(B12-B13)&\" 项通过\")"],
  ["=IF(B14>=8,\"文献卷宗：达到最低数量\",\"文献卷宗：还缺 \"&(8-B14)&\" 篇有效证据\")"],
  ["=IF(B15=3,\"端到端任务：已完成\",\"端到端任务：还需 \"&(3-B15)&\" 项通过\")"],
  ["=IF(AND(B11=15,B12=10,B13=10,B14>=8,B15=3,B16=4),\"RA-0B 可申请关闭\",\"RA-0B 暂不可关闭\")"],
];
body(guide.getRange("D11:H15"));
addStatusFormatting(guide.getRange("D11:H15"));
guide.freezePanes.freezeRows(1);
setWidths(guide, { A: 22, B: 18, C: 3, D: 20, E: 18, F: 18, G: 18, H: 18 });

// AI预审摘要：先给审核人结论，再保留人工门禁。
title(summary, "A1:F1", "AI预审摘要｜你需要重点复核什么");
note(summary, "A2:F2", "这里是AI基于冻结数据库快照、现有卷宗和本地PDF做出的预审意见，不是材料专家签字。用户复核列仍全部保持“待复核”。");
summary.getRange("A4:F4").values = [["审核模块", "AI预审结果", "数量", "证据强度", "关键边界", "建议你怎么复核"]];
header(summary.getRange("A4:F4"));
summary.getRange("A5:F9").values = [
  ["筛选规则", "建议通过", "15/15", "高", "独立复算与冻结参考的完整ID顺序一致；screen-003安全拒绝未支持条件是正确行为。", "抽查screen-003、零结果screen-013和Fe2O3全量screen-015。"],
  ["材料身份", "确认记录不同；实验物相待证", "10/10", "记录层面高，实验物相层面不足", "同化学式、不同MP ID和空间群足以禁止合并，但不能自动证明已实验确认的不同物相。", "核对结构/CIF或衍射证据后，再决定是否升级为“不同实验物相”。"],
  ["任务相关文献", "仅3篇可进入当前任务复核", "3/8", "中", "其余5篇主题与三个端到端任务不匹配，不能为凑数量计入有效证据。", "逐页检查3篇候选文献的物相、样品、制备条件和结论上限。"],
  ["端到端任务", "3项均需补证据", "0/3可关闭", "中低", "数据库排序可复现，但候选与实验文献尚未完成逐一结构映射和候选特异验证。", "优先补宽禁带候选实验带隙，以及Fe2O3候选的结构匹配证据。"],
  ["总体结论", "RA-0B暂不可关闭", "—", "—", "现阶段适合作为开发/回归测试银标，不宜宣称专业金标。", "审核各页AI依据；同意时把用户复核改为“通过”，不同意则写原因。"],
];
body(summary.getRange("A5:F9"));
summary.getRange("A5:F9").format.rowHeight = 62;
addStatusFormatting(summary.getRange("B5:B9"));
summary.freezePanes.freezeRows(4);
setWidths(summary, { A: 22, B: 28, C: 15, D: 25, E: 68, F: 64 });

// 筛选用例：分开能力行为和参考答案审核，并显示完整参考信息。
title(screening, "A1:N1", "筛选用例｜AI预审与用户确认分开");
note(screening, "A2:N2", "AI已用冻结快照独立复算全部15项。J、K是AI意见，L才是你的最终确认；系统不支持某条件时，安全停止可判能力行为正确，但参考答案仍单独核对。");
screening.getRange("A3:N3").values = [[
  "用例", "科研问题", "精确约束", "排序合同", "参考命中", "参考Top-K", "当前执行能力",
  "当前实际命中", "自动证据", "AI能力预审", "AI参考答案预审", "用户复核", "用户/复核人", "AI依据与用户备注",
]];
header(screening.getRange("A3:N3"));
const screeningRows = screeningCases.map((item) => {
  const expected = expectedById.get(item.case_id);
  const current = baselineById.get(item.case_id);
  const sortLabel = `${item.sort.field} / ${item.sort.direction}${item.sort.target === undefined ? "" : ` / target=${item.sort.target}`}`;
  const supported = current.status === "evaluated" && current.filter_exact;
  return [
    item.case_id,
    item.question,
    JSON.stringify(item.constraints),
    sortLabel,
    expected.matched_count,
    expected.expected_material_ids.join("；"),
    supported ? "过滤可执行；RA-3支持指定排序" : `安全停止：不支持 ${(current.unsupported_constraints ?? []).join("、")}`,
    supported ? current.actual_match_count : null,
    supported
      ? `自动过滤与参考集合一致；仍需独立复核Top-K`
      : `未忽略不支持条件；参考集合由冻结快照独立生成（${expected.matched_count}条）`,
    "通过（AI预审）",
    "通过（AI预审）",
    "待复核",
    "",
    item.case_id === "screen-003"
      ? `AI依据：系统未忽略is_gap_direct而是安全停止；独立复算31条，Top10=${independentAuditById.get(item.case_id).topIds.join("、")}。请确认“拒绝未支持条件”是否符合产品合同。`
      : `AI依据：冻结快照300条独立复算；完整命中集合、数量${independentAuditById.get(item.case_id).count}及排序均与参考答案逐项一致。`,
  ];
});
screening.getRange("A4:N18").values = screeningRows;
body(screening.getRange("A4:N18"));
screening.getRange("A4:N18").format.rowHeight = 58;
screening.getRange("E4:E18").format.numberFormat = "0";
screening.getRange("H4:H18").format.numberFormat = "0";
screening.getRange("L4:L18").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据"] } };
addStatusFormatting(screening.getRange("J4:L18"));
screening.freezePanes.freezeRows(3);
screening.freezePanes.freezeColumns(2);
setWidths(screening, { A: 13, B: 40, C: 50, D: 32, E: 12, F: 58, G: 34, H: 14, I: 40, J: 16, K: 16, L: 14, M: 16, N: 44 });

// 候选明细：为每个筛选用例提供可筛选、可追溯的参考候选。
title(candidates, "A1:O1", "候选明细｜冻结快照参考结果");
note(candidates, "A2:O2", "每行是某一筛选用例中的一个参考候选。排序序号、Top-K标记和属性均来自冻结的Materials Project快照；审核人可按用例筛选并逐项复算。缺失值保持空白，不作猜测。");
candidates.getRange("A3:O3").values = [[
  "用例", "排序序号", "Top-K", "材料ID", "化学式", "化学体系", "带隙(eV)", "直接带隙", "非金属",
  "凸包上方能量(eV/atom)", "稳定", "密度(g/cm³)", "空间群", "晶系", "来源链接",
]];
header(candidates.getRange("A3:O3"));
const candidateRows = [];
for (const expected of screeningExpected) {
  const top = new Set(expected.expected_material_ids);
  expected.expected_all_material_ids.forEach((materialId, index) => {
    const record = materialById.get(materialId) ?? {};
    candidateRows.push([
      expected.case_id,
      index + 1,
      top.has(materialId) ? "是" : "否",
      materialId,
      record.formula_pretty ?? "",
      record.chemsys ?? "",
      record.band_gap_ev ?? null,
      record.is_gap_direct ?? null,
      record.is_metal === undefined ? null : !record.is_metal,
      record.energy_above_hull_ev_atom ?? null,
      record.is_stable ?? null,
      record.density_g_cm3 ?? null,
      record.spacegroup_number ?? null,
      record.crystal_system ?? "",
      `https://materialsproject.org/materials/${materialId}`,
    ]);
  });
}
const candidateEnd = candidateRows.length + 3;
candidates.getRange(`A4:O${candidateEnd}`).values = candidateRows;
body(candidates.getRange(`A4:O${candidateEnd}`));
candidates.getRange(`A4:O${candidateEnd}`).format.rowHeight = 21;
candidates.getRange(`B4:B${candidateEnd}`).format.numberFormat = "0";
candidates.getRange(`G4:G${candidateEnd}`).format.numberFormat = "0.000";
candidates.getRange(`J4:J${candidateEnd}`).format.numberFormat = "0.000000";
candidates.getRange(`L4:L${candidateEnd}`).format.numberFormat = "0.000";
candidates.getRange(`M4:M${candidateEnd}`).format.numberFormat = "0";
candidates.freezePanes.freezeRows(3);
candidates.freezePanes.freezeColumns(4);
setWidths(candidates, { A: 13, B: 12, C: 10, D: 16, E: 17, F: 20, G: 13, H: 13, I: 11, J: 25, K: 11, L: 17, M: 11, N: 15, O: 48 });
const candidateTable = candidates.tables.add(`A3:O${candidateEnd}`, true, "CandidateReferenceTable");
candidateTable.style = "TableStyleMedium2";

// 身份复核：只让专业人员审核十条真正需要判断的案例。
title(identity, "A1:R1", "材料身份复核｜AI建议与用户确认分开");
note(identity, "A2:R2", "AI只确认10组是不同数据库结构记录，因此不得按化学式合并；这不等于已证明不同实验物相。O列是AI建议，P列由你确认。");
identity.getRange("A3:R3").values = [[
  "用例", "化学式/体系", "左材料ID", "左空间群", "左晶系", "左带隙", "左凸包能量", "左链接",
  "右材料ID", "右空间群", "右晶系", "右带隙", "右凸包能量", "审核要求", "AI建议判断", "用户复核",
  "AI依据与用户备注", "用户/复核人",
]];
header(identity.getRange("A3:R3"));
identity.getRange("A4:R23").values = identityCases.map((item) => {
  const left = materialById.get(item.left_material_id) ?? {};
  const right = materialById.get(item.right_material_id) ?? {};
  const needsDomain = item.review_status !== "deterministic";
  return [
    item.case_id,
    item.formula ?? item.chemsys ?? left.formula_pretty ?? "同一来源记录",
    item.left_material_id,
    item.left_spacegroup ?? left.spacegroup_number ?? null,
    left.crystal_system ?? "",
    left.band_gap_ev ?? null,
    left.energy_above_hull_ev_atom ?? null,
    `https://materialsproject.org/materials/${item.left_material_id}`,
    item.right_material_id,
    item.right_spacegroup ?? right.spacegroup_number ?? null,
    right.crystal_system ?? "",
    right.band_gap_ev ?? null,
    right.energy_above_hull_ev_atom ?? null,
    needsDomain ? "需材料专业复核" : "规则确定，不计入专业门禁",
    needsDomain ? "只能确认记录不同" : item.expected_relationship,
    needsDomain ? "待复核" : "规则通过",
    needsDomain
      ? `AI依据：同一化学式，但材料ID不同（${item.left_material_id} / ${item.right_material_id}）且空间群不同（${item.left_spacegroup} / ${item.right_spacegroup}），足以判定为不同数据库结构记录并禁止合并；缺少结构匹配或实验衍射证据，暂不升级为不同实验物相。`
      : "由相同稳定ID规则确定，不进入材料专业门禁。",
    "",
  ];
});
body(identity.getRange("A4:R23"));
identity.getRange("A4:R23").format.rowHeight = 38;
identity.getRange("D4:D23").format.numberFormat = "0";
identity.getRange("F4:G23").format.numberFormat = "0.000000";
identity.getRange("J4:J23").format.numberFormat = "0";
identity.getRange("L4:M23").format.numberFormat = "0.000000";
identity.getRange("O4:O23").dataValidation = { rule: { type: "list", values: ["不同物相/结构记录", "同一物相重复记录", "只能确认记录不同", "证据不足", "待判断"] } };
identity.getRange("P4:P23").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据", "规则通过"] } };
addStatusFormatting(identity.getRange("P4:P23"));
identity.freezePanes.freezeRows(3);
identity.freezePanes.freezeColumns(2);
setWidths(identity, { A: 13, B: 17, C: 16, D: 11, E: 15, F: 12, G: 16, H: 43, I: 16, J: 11, K: 15, L: 12, M: 16, N: 25, O: 26, P: 15, Q: 48, R: 16 });

// 文献复核和逐条证据定位。
title(literatureSheet, "A1:N1", "文献复核｜AI证据审查与用户确认");
note(literatureSheet, "A2:N2", "AI已先判断任务相关性和结论上限。既有卷宗approved只表示抽取曾审核，不表示与当前任务相关；K列仍由你确认。");
literatureSheet.getRange("A3:N3").values = [[
  "序号", "文献/卷宗ID", "标题", "拟支持任务", "卷宗路径", "全文/PDF/网页", "材料范围或证据角色",
  "AI材料/物相判断", "AI条件判断", "AI定位判断", "用户复核", "AI判定的最强结论", "缺口/冲突", "用户/日期",
]];
header(literatureSheet.getRange("A3:N3"));
const seedLiteratureAudit = [
  ["部分匹配", "不匹配", "完整", "仅支持特定Nb掺杂TiO2/TCNQ界面电荷转移结论，不能验证纯TiO2候选的实验带隙、相稳定性或器件性能。", "对象含掺杂与界面体系，不能等同screen-001的纯氧化物候选。"],
  ["不匹配", "不匹配", "完整", "不支持当前三个端到端任务。", "主题为骨诱导生物陶瓷筛选，与宽禁带氧化物、Fe2O3多晶型和亚稳氧化物任务无关。"],
  ["不匹配", "不匹配", "完整", "不支持当前三个端到端任务。", "主题为生物陶瓷/结构支架，与当前任务不相关。"],
  ["不匹配", "不匹配", "完整", "不支持当前三个端到端任务。", "主题为混合阳离子钙钛矿发光太阳能电池，与当前任务不相关。"],
  ["不匹配", "不匹配", "完整", "不支持当前三个端到端任务。", "主题为生物陶瓷，与当前任务不相关。"],
];
const literatureRows = literature.map((item, index) => {
  const audit = seedLiteratureAudit[index];
  return [index + 1, item.document_id, item.title, index === 0 ? "e2e-a-wide-gap-oxides（仅背景）" : "无", item.dossier_path, item.review_path, item.benchmark_role, audit[0], audit[1], audit[2], "待复核", audit[3], audit[4], ""];
});
const candidateLiteratureAudit = {
  "lit-candidate-001": ["匹配", "部分匹配", "完整", "在喷雾干燥、SiO2限域、Fe/Si=0.4、1180°C/4h条件下获得高纯ε-Fe2O3；Mössbauer面积约92.7%。", "结论限定于特定工艺和复合颗粒；尚未把该实验相映射到具体MP记录。"],
  "lit-candidate-002": ["匹配", "部分匹配", "完整", "β-Fe2O3纳米颗粒在30 GPa以上转为单斜I2/a的ζ-Fe2O3，卸压后可保留，但样品含α+ζ混相。", "高压、纳米前驱体及混相条件不可省略；不能外推为常压块体通用合成。"],
  "lit-candidate-003": ["部分匹配", "不匹配", "完整", "跨约3万种ICSD晶态无机材料的统计表明，0 K下高于凸包并不能单独排除可合成性。", "这是群体统计边界证据，不能证明screen-004中某个具体候选已被合成。"],
};
for (const [offset, item] of literatureCandidates.entries()) {
  const audit = candidateLiteratureAudit[item.candidate_id];
  literatureRows.push([
    literature.length + offset + 1,
    item.candidate_id,
    item.title,
    item.task_ids.join("；"),
    item.dossier_candidate_path,
    item.local_pdf_path ?? item.open_full_text_url,
    `${item.material_scope}；${item.evidence_role}`,
    audit[0],
    audit[1],
    audit[2],
    "待复核",
    audit[3],
    audit[4],
    "",
  ]);
}
literatureSheet.getRange("A4:N11").values = literatureRows;
body(literatureSheet.getRange("A4:N11"));
literatureSheet.getRange("A4:N11").format.rowHeight = 68;
literatureSheet.getRange("A4:A11").format.numberFormat = "0";
literatureSheet.getRange("H4:J11").dataValidation = { rule: { type: "list", values: ["匹配", "部分匹配", "不匹配", "完整", "不完整", "待核对"] } };
literatureSheet.getRange("K4:K11").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据"] } };
addStatusFormatting(literatureSheet.getRange("K4:K11"));
literatureSheet.freezePanes.freezeRows(3);
literatureSheet.freezePanes.freezeColumns(2);
setWidths(literatureSheet, { A: 8, B: 29, C: 44, D: 30, E: 46, F: 48, G: 42, H: 18, I: 18, J: 16, K: 15, L: 52, M: 50, N: 20 });

const dossierEntries = [
  ...literature.map((item) => ({ documentId: item.document_id, path: item.dossier_path })),
  ...literatureCandidates.map((item) => ({ documentId: item.document_id, path: item.dossier_candidate_path })),
];
const excerptRows = [];
for (const entry of dossierEntries) {
  try {
    const dossier = await readProjectJson(entry.path);
    for (const item of dossier.items ?? []) {
      excerptRows.push([
        entry.documentId,
        item.category ?? "",
        item.summary ?? "",
        item.source_quote ?? "",
        item.page ?? null,
        item.page_to ?? null,
        item.chunk_id ?? "",
        item.risk_level ?? "",
        dossier.review_status ?? "",
        entry.path,
      ]);
    }
  } catch {
    excerptRows.push([entry.documentId, "", "未找到可读取的卷宗摘录", "", null, null, "", "", "missing", entry.path]);
  }
}
title(excerpts, "A1:J1", "证据摘录｜审核必须回到原文定位");
note(excerpts, "A2:J2", "摘要用于快速理解，原文摘录、页码和chunk用于定位。审核人仍应打开对应PDF核对上下文；不能只根据摘要批准科学结论。");
excerpts.getRange("A3:J3").values = [["文献ID", "类别", "结构化摘要", "原文摘录", "起始页", "结束页", "chunk", "风险", "卷宗状态", "卷宗路径"]];
header(excerpts.getRange("A3:J3"));
const excerptEnd = excerptRows.length + 3;
excerpts.getRange(`A4:J${excerptEnd}`).values = excerptRows;
body(excerpts.getRange(`A4:J${excerptEnd}`));
excerpts.getRange(`A4:J${excerptEnd}`).format.rowHeight = 58;
excerpts.getRange(`E4:F${excerptEnd}`).format.numberFormat = "0";
excerpts.freezePanes.freezeRows(3);
excerpts.freezePanes.freezeColumns(2);
setWidths(excerpts, { A: 29, B: 20, C: 54, D: 74, E: 11, F: 11, G: 38, H: 12, I: 14, J: 52 });
const excerptTable = excerpts.tables.add(`A3:J${excerptEnd}`, true, "EvidenceExcerptTable");
excerptTable.style = "TableStyleMedium2";

// 端到端任务：候选、证据和审查问题在同一行可定位。
title(tasks, "A1:M1", "端到端任务｜AI建议、证据边界与用户确认");
note(tasks, "A2:M2", "AI已写出当前证据能支持的结论和最小补证方案。它们均不等于候选已被实验验证；J列由你确认是否接受这份审查。");
tasks.getRange("A3:M3").values = [[
  "任务", "标题", "筛选用例", "候选总数", "参考shortlist", "相关文献/证据", "必须守住的结论边界",
  "AI建议结论（含不可下结论）", "AI建议最小验证计划", "用户复核", "主要问题", "用户/复核人", "复核日期",
]];
header(tasks.getRange("A3:M3"));
const taskAudit = {
  "e2e-a-wide-gap-oxides": [
    "冻结快照可复现得到83个满足计算筛选条件的候选及Top10。只能称为数据库优先级名单；不能据此断言实验带隙、绝缘强度、可制备性或器件性能。现有Nb掺杂TiO2/TCNQ文献不是候选特异验证。",
    "先选Top10中2–3个代表候选：核对结构身份与实验相；每个补至少1篇实验带隙/相稳定性来源。若做实验，采用XRD确认物相并测UV-Vis漫反射带隙；不满足则降级为计算候选。",
    "缺少Top候选逐一对应的实验相和实验带隙证据。",
  ],
  "e2e-b-polymorphs": [
    "冻结快照给出26个Fe2O3数据库结构记录，不能按化学式合并。文献证明ε相可由特定喷雾干燥/退火路线获得，ζ相可在>30 GPa由β相形成并卸压保留；但尚不能把这些实验相直接指派给某个MP ID。",
    "对拟保留的MP结构下载CIF并与论文空间群、晶格参数或模拟XRD逐一匹配；至少完成α/ε/ζ相关候选的结构映射。匹配失败时只报告“数据库结构记录”，不使用实验物相名称。",
    "已有相特异文献，但数据库ID—实验物相映射未完成。",
  ],
  "e2e-c-metastable-synthesized": [
    "冻结快照给出58个0<E_hull≤0.10 eV/atom的亚稳候选。跨材料研究仅说明正凸包能不等于不可合成；ε/ζ-Fe2O3提供特定路线实例，不能推广为58个候选均可合成。",
    "从Top10选3个候选，逐一查找同组成/同结构的实验合成证据并记录温压、气氛、尺寸和淬火条件；无候选特异证据者标为“计算亚稳候选”，不得标为“已合成”。",
    "缺少绝大多数候选的特异合成证据和稳定窗口。",
  ],
};
tasks.getRange("A4:M6").values = endToEndTasks.map((item) => {
  const audit = taskAudit[item.task_id];
  return [
    item.task_id, item.title, item.screening_case_id, item.database_expected.matched_count,
    item.database_expected.expected_material_ids.join("；"),
    [...(item.literature_seed_document_ids ?? []), ...(item.literature_candidate_ids ?? [])].join("；") || "尚无任务特定证据",
    item.required_claim_checks.join("；"), audit[0], audit[1], "待复核", audit[2], "", null,
  ];
});
body(tasks.getRange("A4:M6"));
tasks.getRange("A4:M6").format.rowHeight = 92;
tasks.getRange("D4:D6").format.numberFormat = "0";
tasks.getRange("J4:J6").dataValidation = { rule: { type: "list", values: ["通过", "待复核", "不通过", "需补证据"] } };
tasks.getRange("M4:M6").format.numberFormat = "yyyy-mm-dd";
addStatusFormatting(tasks.getRange("J4:J6"));
tasks.freezePanes.freezeRows(3);
tasks.freezePanes.freezeColumns(2);
setWidths(tasks, { A: 31, B: 31, C: 16, D: 12, E: 62, F: 42, G: 52, H: 58, I: 58, J: 15, K: 48, L: 16, M: 15 });

// 签字页：签字不能绕过各页的明细门禁。
title(signoff, "A1:G1", "RA-0B 关闭前签字确认");
note(signoff, "A2:G2", "只有审核导航显示所有明细门禁均通过，且四个责任项分别签字，才能把本基准称为专业金标。签字不能替代逐项审核。");
signoff.getRange("A3:G3").values = [["签字项", "责任角色", "确认内容", "姓名", "状态", "日期", "备注"]];
header(signoff.getRange("A3:G3"));
signoff.getRange("A4:G7").values = [
  ["筛选规则复核", "独立复核人", "15个用例的能力行为与冻结快照参考答案均已独立复核", "", "待确认", null, ""],
  ["材料身份复核", "材料专业人员", "10个需专业判断的身份案例已逐项填写依据并通过", "", "待确认", null, ""],
  ["文献证据复核", "材料专业人员", "至少8个卷宗完成物相、条件、定位和结论上限审核", "", "待确认", null, ""],
  ["端到端答案复核", "课题负责人", "3个任务均具有候选分层、证据边界和最小验证计划", "", "待确认", null, ""],
];
body(signoff.getRange("A4:G7"));
signoff.getRange("E4:E7").dataValidation = { rule: { type: "list", values: ["已确认", "待确认", "不通过"] } };
signoff.getRange("F4:F7").format.numberFormat = "yyyy-mm-dd";
addStatusFormatting(signoff.getRange("E4:E7"));
signoff.freezePanes.freezeRows(3);
setWidths(signoff, { A: 22, B: 20, C: 62, D: 18, E: 15, F: 15, G: 46 });

await fs.mkdir(outputDir, { recursive: true });

const renderTargets = [
  ["审核导航", "A1:H16"],
  ["AI预审摘要", "A1:F9"],
  ["筛选用例", "A1:N18"],
  ["候选明细", `A1:O${Math.min(candidateEnd, 35)}`],
  ["身份复核", "A1:R23"],
  ["文献复核", "A1:N11"],
  ["证据摘录", `A1:J${Math.min(excerptEnd, 30)}`],
  ["端到端任务", "A1:M6"],
  ["签字确认", "A1:G7"],
];
for (const [sheetName, range] of renderTargets) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(
    path.join(outputDir, `preview-ai-review-${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const keyInspection = await workbook.inspect({
  kind: "table",
  range: "审核导航!A10:H16",
  include: "values,formulas",
  tableMaxRows: 10,
  tableMaxCols: 8,
  maxChars: 6000,
});
const screen003Inspection = await workbook.inspect({
  kind: "table",
  range: "筛选用例!A6:N6",
  include: "values,formulas",
  tableMaxRows: 2,
  tableMaxCols: 14,
  maxChars: 6000,
});
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "AI pre-review final formula error scan",
  maxChars: 4000,
});

const output = await SpreadsheetFile.exportXlsx(workbook);
const outputPath = path.join(outputDir, "RA0_晶态无机材料筛选_AI预审工作簿.xlsx");
await output.save(outputPath);

console.log(JSON.stringify({
  outputPath,
  candidateRows: candidateRows.length,
  excerptRows: excerptRows.length,
  sheetsRendered: renderTargets.length,
  formulaErrorScan: errors.ndjson,
  keyInspection: keyInspection.ndjson,
  screen003Inspection: screen003Inspection.ndjson,
}, null, 2));
