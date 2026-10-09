# DataAnalysisAgent 演示数据

- `quality_issues_demo.csv`：重复行、缺失值、常量列和异常值候选；用于质量检查和
  不可变清洗。
- `grouped_experiment_demo.csv`：A/B/C 三组共 30 行；用于分组描述、相关性、
  Welch t、ANOVA、箱线图、散点图和报告。
- `materials_properties_demo.csv`：含 `mp-*` ID 的材料属性；用于材料字段分析和
  DataAnalysisAgent → MaterialsDatabaseAgent 回查。
- `paired_experiment_demo.csv`：同一试样处理前后各测量一次；用于配对 t 检验与
  Wilcoxon 符号秩检验。
- `repeated_measures_demo.csv`：同一试样在 T0/T1/T2 三个时间点测量；用于 Friedman
  重复测量检验及 Holm 校正的事后比较。

推荐依次测试：

1. 上传质量数据并执行“一键 EDA”；
2. 上传分组实验数据，用表单选择 `hardness_hv`、`method` 和 `welch_t`；
3. 选择 `temperature_c`、`conductivity_s_cm` 生成散点图；
4. 上传配对或重复测量数据，确认样本 ID 与条件/时间字段后开始分析；
5. 生成科研 Word 报告并下载；
6. 上传材料属性数据，选择 `band_gap_ev` 与 `density_g_cm3` 做相关性分析。
