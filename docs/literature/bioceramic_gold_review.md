# 2025 生物陶瓷论文金标审核单

> 文档 ID：`doc-6be19aada0b95fe176f931ee`  
> 论文：Optimal structural characteristics of osteoinductivity in bioceramics
> derived from a novel high-throughput screening plus machine learning approach  
> DOI：`10.1016/j.biomaterials.2025.123348`  
> 当前状态：已由用户批准（2026-08-20）  
> 数据来源：主论文正文；未对图中柱形高度进行人工估读，未使用补充材料中的未下载数据。

## 1. 矩阵范围

| 类型 | 数量 | 说明 |
| --- | ---: | --- |
| 实验结构组 | 24 | 4 种孔型 × 6 个孔隙率水平（50%、55%、60%、65%、70%、75%） |
| 总体分析组 | 1 | 对24种结构进行PCC和XGBoost分析 |
| 测量/正文结果 | 37 | 数值区间、相关系数、定性组别结果和结构完整性 |
| 组间比较 | 5 | 1条成骨比较、4条70%与75%结构完整性比较 |
| 摘要主张 | 1 | 最佳SSA、渗透率、孔隙率及孔型影响结论 |
| 摘要—正文证据链接 | 1 | 标记为正文支持，待人工批准 |

## 2. 实验设计金标

- 材料：DLP 3D打印磷酸钙陶瓷。
- 孔型：triangular、diamond、square、polyhedral。
- 孔隙率：每种孔型包含50%、55%、60%、65%、70%、75%，共24种组合。
- 体外平台：CaP Chip。
- 体内平台：CaP Cyl-scaffold，犬背肌植入180天。
- 重复：每项测试至少3个平行样本；动物样本 `n=3`。
- 主文报告的总体参数范围：孔隙率50%–75%，模型SSA 7.29–13.38
  mm²/mm³，渗透率1.15–5.67 × 10⁻⁹ m²。

证据位置：PDF第3、5页。

## 3. 关键数值结果

| 指标 | 正文值 | 位置 |
| --- | --- | ---: |
| 孔隙率与BV/TV的PCC | 0.541，P < 0.05 | 第7页 |
| 有利于BV/TV的SSA平台 | 9.66–11.73 mm²/mm³ | 第7页 |
| COL-I表达SSA平台 | 9.91–10.69 mm²/mm³ | 第7页 |
| OCN表达SSA平台 | 10.49–10.69 mm²/mm³ | 第7页 |
| OPN表达SSA平台 | 9.91–10.69 mm²/mm³ | 第7页 |
| Runx-2表达SSA平台 | 9.34–9.72 mm²/mm³ | 第7页 |
| ALP表达峰值SSA | 9.817 mm²/mm³ | 第7页 |
| COL-I/OPN/OCN表达峰值渗透率 | 3.74 × 10⁻⁹ m² | 第7页 |

说明：摘要和结论采用交集区间10.49–10.69 mm²/mm³作为综合最佳SSA；矩阵同时保留
各基因和BV/TV在正文中分别报告的区间，避免只留下摘要中的一个数字。

## 4. 明确报告的组别结果

| 时间/评价 | 正文报告的结构组 |
| --- | --- |
| 第7天细胞增殖较高 | diamond 60%；polyhedral 50%、55% |
| 第14天细胞增殖较同孔型其他组高 | diamond 70%、75%；polyhedral 70%、75% |
| 第7天COL-I和OPN升高 | polyhedral 65%；square 70% |
| micro-CT成骨表现优秀 | triangular 65%；polyhedral 65%；square 70% |
| micro-CT新骨量最大 | polyhedral 70% |
| 组织学新骨面积最大 | triangular 70% |
| BV/TV未计算 | 四种孔型的75%层，原因是支架解体 |
| 180天后结构完整性 | 50%–70%基本完整；75%明显降解/解体 |

证据位置：PDF第5–7、10页。

## 5. 组间比较

1. polyhedral 65% → polyhedral 70%：体内新骨形成从“优秀”上升为“最大”。
2. triangular 70% → triangular 75%：结构完整性下降。
3. diamond 70% → diamond 75%：结构完整性下降。
4. square 70% → square 75%：结构完整性下降。
5. polyhedral 70% → polyhedral 75%：结构完整性下降。

这些比较均标记为 `reported`，没有计算不存在于正文中的相对变化百分比。

## 6. 摘要与正文核验

摘要主张：骨再生主要受孔隙率和SSA影响，孔型影响较小；最佳SSA为
10.49–10.69 mm²/mm³、最佳渗透率为3.74 × 10⁻⁹ m²，对应约65%–70%孔隙率。

正文核验：结论部分再次报告polyhedral 70%新骨形成最高，PCC/ML分析确认孔隙率与
SSA的重要性，并重复最佳SSA和渗透率。因此候选核验结果为 `supported`。

## 7. 有意保留的缺失项

- Fig. 2–5中的柱形图没有在主文中给出全部24组精确数值，因此没有人工估读。
- 75%组因解体而未计算BV/TV，保留为“未分析”，没有填入0。
- 补充材料Table S2和Fig. S2–S4未包含在当前PDF中，因此未录入模型性能和补充数值。
- “孔型影响可忽略”是作者的总体结论，不被改写成不存在的组间数值差。

这些不是抽取遗漏，而是防止金标包含推测数据的证据边界。

## 8. 审核记录

用户已于2026-08-20批准该矩阵，审核理由为
`2025 bioceramic gold matrix verified`。数据库为69条矩阵实体分别写入不可变审核事件：
25组、37测量、5比较、1主张和1证据链接。

复核数据库中的已批准矩阵：

查看数据库中的完整待审矩阵：

```powershell
uv run materials-screen literature matrix-show `
  doc-6be19aada0b95fe176f931ee `
  --status approved
```
