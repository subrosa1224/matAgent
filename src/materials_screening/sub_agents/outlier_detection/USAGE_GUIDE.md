# Outlier Detection 兼容入口使用说明

此入口仅用于兼容旧调用方，不再作为独立 Agent 扩展。

## 新调用方式

通过 Master Agent 使用 `materials_database`，并在用户明确要求“离群、异常值或异常点”时调用统一离群检测能力：

```powershell
uv run materials-screen master ask --llm-provider intern `
  -m "筛选含 Li 的稳定材料，并检查带隙离群值"
```

环境变量：

```dotenv
LLM_PROVIDER=intern
INTERN_API_KEY=你的书生API_TOKEN
INTERN_BASE_URL=https://chat.intern-ai.org.cn/api/v1/
INTERN_MODEL=intern-s2-preview-35b
INTERN_THINKING_MODE=true
MP_API_KEY=你的Materials_Project_API_KEY
```

仅查询带隙不会自动触发离群检测。详细说明见 [统一使用指南](../materials_database/USAGE_GUIDE.md)。
