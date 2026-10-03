# c6：在 c4 上保留既有能力的修复训练

2026-10-03，c6 训练启动前登记。c5 及其失败结果保留，不覆盖。

c5 从 r14 重训后，目标一致性开发集668条全部正确，六条 mined regression 全部选择重新理解；但旧 c-dev 接管率85.37%→78.51%，P03切片退化，历史单句集0.8误执行8→10且配对检验失败。因此不运行新验收，不导出、不发布c5。

## 下一轮变化

- 初始化改为现有已验证 c4 / 7b695ba8，避免从 r14 重新学习 c4 已有能力。
- 原 c5 数据、拆分、采样权重逐字节复用，不增删失败样本，也不使用新验收结果。
- 做一轮小步修复：encoder学习率5e-6、head学习率2e-5，均为c5的1/5；其余优化与模型设置不变。全参数仍可训练。
- 使用原混合calib重新拟合温度，不提高运行时阈值，不加规则或修改Agent/SDK。

## 判断标准

沿用 PLAN.md 的六条 mined、target-dev、单句覆盖/误执行和配对检验、新验收及后端验证门槛。新216条验收继续封存，只有有效开发回归通过才运行。

旧续接数据同时按项目已有 LABELING.md C15 和未修改的 tools/cp_audit.py 报告两种口径。该条款在本次实验前就已登记：冻结 c-dev/c-test 不修改，报告原登记指标及剔除已知缺陷对照副本后的指标。c-dev共有7条被该既有审计标记，c5的“老人房灯吧”→“床边夜灯”就在其中；选项说明明确是“老人房·灯”。不能为了符合错误金标训练模型拒绝合法房间指代。

原登记口径的失败始终保留并明确标记。能力保持比较在相同的638条有效c-dev上检查错误执行不增加、接管率下降不超过3个百分点，同时报告原645条。c5即使按此审计仍因接管率和历史单句退化而失败；没有借此放行c5。C15不影响新目标开发集、历史单句集或新验收集。

## 复跑

从仓库根目录复制 c5 的 data、dataset、baseline-new 至新的 run 目录，校验每个文件 SHA-256 与来源一致；随后：

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 laya/.venv/bin/eidolon-laya-train train \
  --config laya/train/scenarios/smart-home-continuation/train-c6.yaml \
  --dataset laya/train/runs/c6/dataset --out laya/train/runs/c6/checkpoint --device mps
```

训练完成后，在 laya/ 下运行同一 finish_target_run.sh（参数 train/runs/c6 mps）。最终权重、校准哈希和结果另存，不引用c5的revision作为c6。

## 运行后补记

c6未通过，见C6-RESULTS.md。逐条检查进一步发现cp_audit匹配的7条中仅1条明确标错，其他6条不能排除。C7-PLAN.md在生成c7前登记了更严格的644条语义复核口径；原645条与机械638条结果都继续保留。本补记不改变c6训练前的实验设置或失败结论。
