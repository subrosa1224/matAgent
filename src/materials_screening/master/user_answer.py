"""Presentation only: retain complete audit output without leading with it."""

import html
import re


def _visible(text):
    lines = [
        line
        for line in text.splitlines()
        if not re.search(
            r"MATERIAL_QUERY_HANDOFF|查询快照|调用记录|完整查询记录|source_artifact_id"
            r"|数据集：`dataset-|计数已校验|Agent 会话与快照",
            # IDs and transport diagnostics are available in the audit layer.
            line,
        )
    ]
    text = "\n".join(lines).strip()
    labels = {
        "band_gap_ev": "计算带隙(eV)",
        "energy_above_hull_ev_atom": "凸包能(eV/atom)",
        "density_g_cm3": "密度(g/cm³)",
        "formula_pretty": "化学式",
        "crystal_system": "晶系",
        "is_stable": "计算稳定性标签",
        "is_metal": "是否金属",
        "material_id": "材料ID",
        "elements": "元素",
        "formation_energy_ev_atom": "形成能(eV/atom)",
    }
    for field, label in labels.items():
        text = text.replace(field, label)
    text = re.sub(r"(?m)^统一检索 ID：.*\n?", "", text)
    text = re.sub(r"(?m)^来源快照时间：.*\n?", "", text)
    return text


def render_user_answer(answer):
    if answer.lstrip().startswith("全文预览（仅用于选文"):
        from .fulltext_preview_display import render_fulltext_preview

        return render_fulltext_preview(answer)
    partial = "这是部分执行报告，不是完整链路成功报告" in answer
    combined = (
        "## 1. Materials Project 初筛" in answer and "## 3. 候选数据分析" in answer
    )
    if not partial and not combined:
        return answer
    main = [
        "本次任务尚未完成，已得到的结果保留；目前还不能给出最终材料推荐。"
        if partial
        else "以下是材料初筛、候选分析与文献检索的阶段结果；"
        "全文实验核验尚未完成，暂不能给出最终推荐。"
    ]
    blocks = re.split(r"(?m)^## ", answer)[1:]
    for block in blocks:
        title, _, body = block.partition("\n")
        if combined:
            titles = {
                "1. Materials Project 初筛": "材料初筛",
                "3. 候选数据分析": "候选属性分析",
                "4. 目标应用文献检索与证据缺口": "相关论文与证据缺口",
                "4. 候选池文献预检、证据重排与下载清单": "相关论文与下载清单",
                "2. 数据库候选快照预览": "候选说明与限制",
                "5. 结论边界与下一步": "结论限制与下一步",
            }
            if title in titles:
                main.append("## " + titles[title] + "\n\n" + _visible(body))
            continue
        if title.startswith("结论边界"):
            continue
        if "已返回结果" in title:
            main.append("## " + title + "\n\n" + _visible(body))
        elif "未执行" in title:
            main.append(
                "## " + title + "\n\n这一阶段尚未开始，并不表示没有相关材料或文献。"
            )
        elif "执行失败" in title:
            main.append("## " + title + "\n\n这一阶段未能完成，已有结果不受影响。")
    main.append(
        "计算属性仅用于初筛，不能替代实验性能。"
        "不同物相、掺杂与制备条件不能直接等同；缺少可比条件时不直接排名。"
        + (
            "暂时无需上传论文来绕过未完成的检索。"
            if partial
            else "下一步请根据检索结果选择相关全文，先预览，确认后再详细分析。"
        )
    )
    main.append(
        "<details>\n<summary>技术详情</summary>\n\n<pre>"
        + html.escape(answer)
        + "</pre>\n</details>"
    )
    return "\n\n".join(main)
