# JevK5 实验索引

## companion-ip-v1 · 2026-09-28

独立项目首轮，已完成数据生成/审核/冻结、基线、seed71两个epoch最后注意力层低秩适配，以及开发评估、重载与选项换序诊断。300条训练记录（150快照），152条DEV；23次GLM调用含1次超时，未重试。无封存测试推理或发布。

epoch2的v7为35/42（原36），v8为51/72（原47），其中弃权5/24（原0）；新陪伴12/22（原13），新IP13/16（原11）。全部候选门槛未通过，selected_epoch=null。补充换序诊断由原模型12/12保持答案降至10/12；固定小TRAIN子集的弃权仍0/5。结果支持停止在同配置上无预算地续训，不构成泛化成功或基座不可训练的证明。

见 [完整结果](experiments/companion-ip-v1/RESULTS.md)、[训练前协议](experiments/companion-ip-v1/DESIGN.md)、[验证清单](experiments/companion-ip-v1/VALIDATION.json)。

## participation-decomposition-v1 · 2026-09-28

已完成328次本地前向：152条冻结DEV的动作/身份核验，以及12条双阶段逆序诊断。无训练、外部调用、封存测试或发布。显式核验在30条应弃权记录上全部失败；同名压力从47/72降到46/72，陪伴动作拆分虽从13/22到15/22，但应停误开口不变。计入提案的拆分串行中位估算5.4–7.6秒，停止此多次核验路线。

后续保持单次决策，优先补证据充分/不足的成对监督，再受控比较扩大适配范围。新训练尚未启动。见 [诊断结论](experiments/participation-decomposition-v1/RESULTS.md)、[预登记](experiments/participation-decomposition-v1/DESIGN.md)、[下一轮约束](experiments/participation-decomposition-v1/NEXT-EXPERIMENT.md)。

## product-readiness-v1 · 2026-09-28

完成Decider/JevK5各48次契约筛查，共96次。原顺序均19/24；逆序14/24、15/24；两顺序都正确13/24、15/24。两者均未过质量线。用户已明确部署云端GPU，选型不以本机/板端延迟淘汰JevK5。完整clarify/恢复/结束监督尚未训练，不能将旧简化任务微调失败推断为基座不可学习。下一方向为一次完整契约监督的可学习性实验，未启动新训练或外部生成。见[质量结论](../evals/participation/product-readiness-v1/RESULTS.md)。
