# 跨模型参与决策评测

业务范围：智能陪伴和 IP 角色团的快速参与决策，输出回应者、等待、结束或需要澄清。回复文本由其他大模型生成。

`data/` 存放本地冻结的合成数据；公开实验报告记录来源、切分、审核及 SHA256。JevK5 首轮入口见 `../../jevk5/experiments/companion-ip-v1/`。新的数据投影和训练实现不依赖 Laya Python 包；历史数据通过显式文件清单导入，原始路径保留。

共享记录为 `{id, state, questions.move, labels.move.gold, split, meta}`；state 和 question 是唯一模型输入，gold 是完整可接受选项集合，meta 存放 family、snapshot、task 和来源。所有快照、改名和换序变体按 family 同组切分。模型依赖、权重和适配器分别属于各模型子项目。

开发集反复用于选型后不能作为独立泛化证据。必须分别报告回应正确/漏回应、停止正确/误发言、歧义弃权、任务与家族稳定性、输入长度和端到端延迟。缺失动作组标为未测，不能隐式通过门槛。语义与速度验收均通过前不得自动接入 Host。

当前推进主线见 [产品可行性路线](product-readiness-v1/ROUTE.md)：先修正clarify/abstained契约覆盖与接续验收，再按部署约束筛选单次决策候选。旧实验不能直接代表产品验收。现有数据覆盖统计见 [coverage.json](product-readiness-v1/coverage.json)。
