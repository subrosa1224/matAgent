# 固定验收 v1（2026-10-06）

用途：停止“换一篇论文就临时改变验收标准”。本目录固定问题、输入指纹与来源参考；不修改生产Agent，不向Agent提供评分答案。当前只建立验收工具，并不表示系统已经稳定通过。

## 范围与题集

范围是晶体无机材料：用户问题 → Materials Project筛选 → 属性分析 → 文献检索 → 用户提供全文 → 实验数据提取/分析 → 有来源的报告。PDF上传/导入仍为现有人工步骤，不能冒称系统自动获取所有全文。实验数据库、付费全文、图线估读不在本轮范围。

普通题N01–N10共10题；边界题B01–B05共5题。开发题N01–N08、B01–B02共10题；保留题N09–N10、B03–B05共5题。15题每题3次，共45条计划记录。

主题包含紫外器件、光催化、Li-Fe-O正极、氧化铁磁性/制备和毫米波吸收。详细自然语言问题在suite.json，准备命令会生成可直接阅读的questions.md。四篇PDF均来自已有用户文件，且已做来源核对；保留的是问题，不是全新论文，不能据此声称跨论文盲测或全材料泛化。

来源参考为待二次复核的银标准，见references.json。原文有冲突的数值不能强行作为单一正确答案。专业人员最终候选排名不在当前金标准范围；本轮首先检验可复核的筛选、数值、条件、归属和结论边界。

## 两类测试分开

- offline_replay：必须使用冻结数据库切片和冻结检索输入，再执行真实Agent。prepare只生成独立期望值，不是离线Agent运行器。当前尚未新增完整的检索回放适配器；不能把已有mock测试登记为offline_agent通过。
- live：真实模型、MP与文献服务；逐次保存实际返回记录、查询限制、时间和安全配置。实际库返回可能变化，不能用冻结切片的候选数直接判错。对该次输入用独立规则重算。
- fault_simulation：B05在适配器注入检索超时/鉴权错误，不改用户密钥、不停止真实服务。当前工具不自动注入故障；未真正运行的B05保留not_run。仅证明安全故障处理，不与真实检索成功率合并。

同一轮正常题30次；主层边界题12次；故障模拟3次，分栏统计，不混成一个“成功率”。不同版本、不同主层各用独立目录与账本。

## 运行顺序

1. 先冻结代码、模型名称/参数、提示词、数据输入与评估规则。prepare记录源代码和规则哈希，但不保存密钥；操作员另存非敏感运行配置。修改任一生产代码后建立新版本，不能拼接不同版本的通过记录。
2. 每次从新主控会话开始，输入suite.json原问题。保留数据库筛选记录、属性数据集/统计结果、文献查询和工具委派轨迹。等待全文属于阶段状态，不算普通题完成。
3. 正常题提供指定PDF，沿同一会话继续；N05使用固定followup。其他题续问要求完成原问题中的全文实验分析，不附上任何参考数值/答案。
4. 使用现有导入/提取流程，记录哪些步骤是用户操作、哪些由Agent调用。若自动提取为0，不能手工抄参考数据来补成通过；若未调用实验数据分析，也不能用属性分析的成功替代。
5. 独立核对实际筛选集合、分组统计、文献相关性、实验数值/样品/条件、报告的每个关键结论。给每个检查附实际产物或审查笔记的路径、哈希、定位。模型自称正确不能充当审核证据。
6. 先跑开发题并归纳共性失败；保留题不用于修复调参。如读过保留题失败并据此调参，下一版需另换真正未参与调参的保留题，而不是继续宣称盲测。
7. 冻结后的候选版本每题执行3次。每次有唯一execution_id和新conversation_id；同次任务的前后半段允许沿同一会话继续。

现有命令入口（以下为流程示例，不是一键45次测试）：

```powershell
# 新会话：将<原问题>替换为固定题原文。
.venv/Scripts/python.exe -m materials_screening.cli master ask --llm-provider intern --materials-repository materials-project --message '<原问题>' --progress

# PDF必须位于既有允许导入目录。这个步骤显式记录为人工调用，不算主控自动导入。
.venv/Scripts/python.exe -m materials_screening.cli literature batch-analyze --pdf '<该题PDF绝对路径>' --extract-matrix

# 使用真实新会话编号，不沿用以前任务的会话。
.venv/Scripts/python.exe -m materials_screening.cli master ask --llm-provider intern --materials-repository materials-project --conversation-id '<本次编号>' --message '<固定全文续问>' --progress
```

若主控续问需要对应的用户全文证据报告，用现有literature user-report入口建立精确主题报告，并记录这一步由谁触发。不允许偷偷把这些人工预处理步骤标成Agent自动完成。

## 缓存与隔离

新会话不等于冷启动。现有论文档案、矩阵、用户报告都有缓存：如果复用旧的提取结果，只能记为warm诊断，不算3次新提取的稳定性证据。模板默认unknown，不能自动改成cold。

要登记cold，必须证明本次LLM提取、样品/条件绑定和数据分析实际重新执行；固定PDF文本/向量索引可复用，但生成的实验结果不可复用。在专用、已隔离的测试存储中执行，不删除、清空或覆盖用户已有批准数据。当前套件未自动建立隔离环境，因此尚不能宣称已具备无人值守冷启动验收。

B02尤其要隔离现有全文报告：它测试“本轮没有全文”，不能从先前缓存偷取全文结果。已有缓存可导致真实部署问题，但测试时应记录而不是隐藏。

## 通过规则

普通题必须同时满足database_selection、property_analysis、literature_search、fulltext_extraction、experimental_analysis、report_grounding六项检查。核对计数应包括完整候选集合；原问题要求化学式和晶系时，两种分组都不能省略。数值须匹配参考记录的样品、单位、测试条件和页码；重复表/正文引用不能算重复实验。每题最低数值记录见suite.json。

边界题只按各自的安全处理检查判定，不计入普通题完成率。程序runtime_status=completed完全不参与评分。

门禁需全部45条已执行且完成独立复核，普通题至少27/30通过；保留普通题也须达到90%（6次中6次通过）；主层边界12/12、故障模拟3/3全部通过；所有执行中严重事实错误为0。严重错误包括编造数值、错样品/条件、计算当实验、前人研究当本文、无依据最佳材料、虚假完成、静默放宽约束。失败或未知检查不能当作通过，未运行不能从分母删除。

缓存诊断、mock或单元测试不能构成此门禁的真实执行证据。评分工具只验证规则和审核记录一致性，无法证明人工审核意见本身正确。通过后仍标pass_provisional：限定范围工程验收，不是专家金标准。

## 使用评分工具

```powershell
# 创建新目录；拒绝覆盖已有输出。
.venv/Scripts/python.exe scripts/stable_acceptance.py prepare --version '<冻结版本>' --tier live --output-dir 'outputs/stable_acceptance/<新目录>'

# 先保留原始ledger.json，按真实执行结果创建reviewed-ledger.json。
.venv/Scripts/python.exe scripts/stable_acceptance.py score --ledger 'outputs/stable_acceptance/<新目录>/reviewed-ledger.json' --output-dir 'outputs/stable_acceptance/<另一个新目录>'

# 只读导出主控会话最新轮次，避免把控制台完成状态当成科学验收。
.venv/Scripts/python.exe scripts/stable_acceptance.py capture --conversation-id '<本次会话>' --output-dir 'outputs/stable_acceptance/<新目录>/<题号-次数>'
```

账本每项review.checks和review.critical_errors的evidence格式：

```json
{"path": "outputs/stable_acceptance/<目录>/independent-audit.md", "sha256": "<文件SHA256>", "locator": "N05第1次：实验数值与条件核对段"}
```

检查verdict为pass/fail/unknown；严重错误为absent/present/unknown；review.completed只在全部核对后置true。matched_reference_record_ids只填本次系统确实提取且核对成功的唯一记录ID，不能仅因原文存在就填入。失败行也需留下证据。报告、数据、轨迹不可凭空填写。

历史诊断只作背景，不自动计入新15题×3次验收。已知Li-Fe-O旧续跑存在0测量、无实验数据分析和参考文献误归属；因此当前没有稳定验收通过的依据。
