# MA-3 Artifact 与附件上下文验收记录

> 对应计划：`unified_multi_agent_ui_plan.md`  
> 阶段：MA-3  
> 日期：2026-08-25  
> 状态：已验收

## 1. 本阶段实现

- 新增服务器侧 `ArtifactRegistry`；
- PDF 上传生成稳定 `MultiAgentArtifact`；
- 统一页面状态只保存 Artifact ID 和安全显示名；
- 真实文件路径只在服务器内部按 Artifact ID 解析；
- 自动模式依据 `pdf_document` 的所有者交给 LiteratureAgent；
- 支持无主题上传直接预览；
- 支持“第1篇”“这些论文”“上传的PDF”等后续指代；
- 附件名称和就绪状态进入执行轨迹；
- 数据库模式仍拒绝 PDF，附件不会传给 DatabaseAgent；
- 伪 PDF、超限 PDF、未知 Artifact 和越界解析均失败关闭。

## 2. 安全边界

- 浏览器状态不保存服务器文件路径；
- Artifact metadata 不包含路径；
- 页面轨迹不显示内部路径；
- 文件复制到受控目录后才允许领域 Agent 使用；
- 新建对话只清理上下文，不删除已产生的持久化文献结果。

## 3. 验收结果

- Ruff：通过；
- mypy：通过；
- Master、Artifact、统一 UI 与文献 UI 定向测试：78 项通过；
- 自动 PDF → LiteratureAgent：通过；
- 无主题 PDF 默认预览：通过；
- “第1篇”私有路径恢复：通过；
- 公共页面状态路径为空：通过；
- 数据库模式拒绝 PDF：通过；
- 真实已分析 PDF 的 Artifact 注册和路由：通过；
- Docker Desktop 与 `matagent-pgvector` 恢复后，`literature doctor` 显示
  `pgvector: ok`、`PDF RAG ready: true`；
- 自动上传 PDF → “深度分析第1篇”：通过，生成 2564 字符结果；
- 同一会话继续“综合这些论文”：通过，生成 2517 字符主题报告；
- 两次真实联调中公共页面状态的 `paths` 均为空。

## 4. 使用方式

```powershell
uv run materials-screen master ui --port 8501
```

保持“自动判断”，上传一批 PDF 后可直接输入：

- `请预览我上传的论文`
- `深度分析第1篇`
- `综合这些论文并生成主题报告`

深度分析前需确保 Docker Desktop 和 `matagent-pgvector` 容器正在运行。
