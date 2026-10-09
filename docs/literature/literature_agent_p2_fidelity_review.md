# LiteratureAgent P2 人工忠实度验收表

> 状态：第二轮评分通过  
> 评分对象：三份自动生成、尚未批准的盲测档案

## 评分规则

- 5 分：摘要与原文一致，类别放置正确，无需修改。
- 4 分：事实忠实，仅有轻微措辞或类别边界问题。
- 3 分：核心事实可由原文支持，但需要明显编辑。
- 2 分：存在重要外推、错配或容易误导的表述。
- 1 分：主要内容没有原文支持。

先检查高风险项，再抽查中低风险项。评分针对整篇候选档案，不要求逐条批准。P2 通过条件是
三篇平均分不低于 4/5，且任何一篇不低于 3/5。

## 查看命令

```powershell
uv run materials-screen literature dossier-pending doc-3250ce3cd68ac48d95b866ae
uv run materials-screen literature dossier-pending doc-cb4c833fc0948254786473b9
uv run materials-screen literature dossier-pending doc-6be19aada0b95fe176f931ee
```

只查看中高风险项时，在命令末尾添加 `--risky-only`。

## 评分记录

| 文档 | 领域 | 分数（1–5） | 主要问题 |
| --- | --- | ---: | --- |
| `doc-3250ce3cd68ac48d95b866ae` | Nb-TiO2/TCNQ | 4 | 沿用首轮用户评分 |
| `doc-cb4c833fc0948254786473b9` | 混合阳离子钙钛矿 | 4 | 完整度与叙述修正后的第二轮用户评分 |
| `doc-6be19aada0b95fe176f931ee` | 骨诱导 CaP 生物陶瓷 | 4 | 完整度与叙述修正后的第二轮用户评分 |

平均分：**4.00/5**。

验收结论：三篇均不低于 3/5，平均分达到 4/5；P2 人工忠实度门通过。
