# Outlier Detection 兼容入口

该目录只保留旧 `outlier_detection` Agent 的兼容入口。新的离群检测能力已经统一迁移到 `materials_database` Agent，底层离群检测服务继续复用。

生产模型仅支持书生 `intern-s2-preview-35b`；`mock` 只用于离线测试。

新代码请使用：

```text
src/materials_screening/sub_agents/materials_database/
```

迁移完成并稳定后，可以删除该兼容目录。完整能力和用法见 [Materials Database Agent](../materials_database/README.md)。
