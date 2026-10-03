# c8：离线教师约束，训练前登记

从c7权重6b3a2f0e开始。冻结c4（7b695ba8）为训练期教师，只对intent/device/action/pick添加中心化logit均方误差，权重0.1；follow不蒸馏。教师与学生读取相同的当前打乱选项后的输入、mask和markers。扣除各题有效选项上的公共logit偏移，忽略padding。教师eval/no_grad且不参与优化器；运行时仍是一个普通模型，无模型分流或新策略。

沿用c5训练、验证、校准文件，字节不变。原监督金标和损失保留。单轮，encoder学习率5e-6、head2e-5，batch8/累积2，全部encoder层训练，Brier0.5，warmup0.06，weight_decay0.01，seed7，pad64。按原val NLL选择checkpoint，原calib拟合温度。不做蒸馏权重搜索。

保持continuation-laya-v1：显式换目标须重新理解。沿用C7全部开发/历史门槛及严格C15仅一条剔除口径，保留原645条和644条结果。阈值0.95/0.5/0.8不改。历史集仅是回归集，不宣称独立泛化结果。冻结的新216条验收在开发和历史门槛全部满足后才能运行一次；否则不验收、不导出、不发布。

训练前检查：24项pytest通过，Ruff通过；真实c7/c4架构、tokenizer、内存隔离和冻结通过；4条训练记录/6题项的小批次真实优化通过，hook核对教师输入逐张量一致、eval/no_grad，教师参数逐位未变。smoke代码与输出保留在c8目录。默认未配置distill时训练路径不变。
