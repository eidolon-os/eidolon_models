# c10 生产切换（2026-10-09）

用户明确授权将 c10 切换生产。使用 Ops 正常 release/deploy 流程，保留身份、状态、旧release和c4制品。先prepare/dry-run，再resume/activate；不重新训练。

原release：rk3588-memory-policy-20261008-2，清单见previous-release.json。更新Models默认制品为c10 e8254243，并配套发布已验证的Agent fc9de67、SDK acf5ce6、Hub 83cf4bd、Channel ba5f8cf。Kernel/Data/Admin/Memory保持原生产提交。使用每个仓库的精确revision，未提交的其他开发内容不入包。

1秒deadline保持不变；已知长上下文偶发超时按现有LLM兜底处理。发布后核查release doctor、app-ready、模型revision/NPU后端、当前Agent配置及一次无设备副作用的生产推理。真实Korvo口述验收不能由HTTP探针替代。

回滚通过Ops的reversible切换快照恢复原release；不删除c4。先记录实际快照路径，再报告切换完成。
