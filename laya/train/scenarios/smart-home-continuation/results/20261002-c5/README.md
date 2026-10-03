# Laya follow目标切换修复：Mac候选 e8254243

c10通过预登记的开发/历史门槛、216条封存合成验收、Mac CPU/MPS FP32一致性与本地服务/并发测试。代码验证累计178个不同测试通过。生产配置未修改，OPI5 Max/NPU按用户要求延后。

**限制必须随候选保留：** 历史单句误执行仍有8条，与c4总数相同，其中有两条新增、两条旧错误修复；v3-dev误执行由3增至4。旧续接接管下降1.79个百分点，在预定3个百分点以内。并非零错误模型，也不代表所有旧行为逐条无退化。新验收为自撰合成对照，不是独立人工盲标。

- 完整当前结果：[C10-RESULTS.md](C10-RESULTS.md)
- 实际训练、恢复证据及本地验证：[REPRODUCE.md](REPRODUCE.md)
- 逐轮实验前登记：PLAN.md、C6/C7/C8/C9/C10-PLAN.md
- c5–c9失败结果全部保留在各轮RESULTS.md和原始证据包。

原始JSON/JSONL、日志、全部错误、实际训练数据与基线报告以`evidence.tar.gz`（Git LFS）保存；`evidence.sha256`校验压缩包，包内清单校验每个文件。`initialization/c9`只用于复现c10训练，本身未通过历史门槛。最终候选Torch权重位于`laya/models/laya-smart-home/e8254243`。
