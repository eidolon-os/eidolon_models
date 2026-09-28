# IP Team 小规模决策模型试验（预登记，2026-09-28）

## 问题和边界

本次只测 Laya 在 IP Team 公开文本快照上预测下一步 respond / clarify / wait / finish / abstain，以及从当次动态候选中选出一位成员。它不生成 instruction，也不决定权限、取消、会话状态或实际说话执行。现有 SDK DecisionRequest/DecisionResult 仍由 Agent 适配器构造、校验和映射；候选键 M0… 仅为模型输入中的局部位置，不能当作真实成员 ID。

## 数据和比较

- 训练：人工审核的模板决策家族，名字、主题、候选顺序变化；反事实对改变最近的公开发言或约束。数据是合成的，没有真实 PTT/ASR 转写。
- 同一训练来源按动作类别做确定性的家族分组切分，家族及其所有变体只在 train、val、calib 之一；每个动作类别都须出现在 val 和 calib。温度仅在 calib 上拟合。初次通用哈希切分使 val 缺少 wait/clarify、calib 缺少 wait/abstain，因此在任何模型训练前改为下面的分层规则。
- 选型看独立的 comparison 家族；holdout 仅在方案固定后一次报告。两组的家族 ID、名字和主题与训练来源隔离；仍属合成诊断，不能称为线上验收。
- 对照：r14 原样；IP-only 从 r14 初始化、只训 IP；joint 从相同 r14 初始化、加入 r17 训练切分中按稳定哈希抽取约 2.5% 的家居回放，且只加到 train，不改变两模型的 IP val/calib。r14 原权重、既有 release 和历史训练产物均不修改。
- 家居回归以历史 v1/v2-dev/v3-dev 报告作描述性参考。这些集合已参与之前选型，不是新的独立验收，更不能用它们反复调本次超参数。历史 v2-test 同理只作背景。

## 预设选择规则

1. 首先要求 comparison 上动作、成员、完整一步分别报告；对 respond/clarify 的成员正确率单独计算，wait/finish/abstain 必须正确选择 NONE。同时报告分组家族正确率、反事实成对正确率、各切片、置信覆盖与错误例数。
2. 如果 IP-only 和 joint 的 comparison 完整一步准确率差不超过 **2 个百分点**，优先选择 IP-only，避免与家居权重耦合；joint 只有超过 IP-only **2 个百分点**、且历史家居开发集每个主要问题准确率相对 r14 下降不超过 **3 个百分点**时才考虑共享。
3. 任一模型若 comparison 完整一步低于 **90%**、任一动作类别低于 **80%**，或反事实成对正确率低于 **80%**，仅保留研究产物，不作为集成候选。即使通过，真实自由对话、ASR 错字、动态授权和 instruction 仍需独立验证，不能自动执行。
4. 不以 holdout 错例重新改数据/配方；若试验失败，报告失败与需要采集的真实样本类别。任何后续训练另立版本与新冻结集。

## 运行约束

Mac 训练必须确认 MPS 可用并显式使用 --device mps。先跑 IP-only，再跑 joint；避免两轮并发争用内存。模型实验目录在 train/runs/；不更新 ops/component.toml、活动 release 目录或 Host 服务。

## 结果（2026-09-28；按上述规则判定）

MPS 检查为 PyTorch 2.14.0、is_built=True、is_available=True。两轮均从 r14 checkpoint 初始化，训练三轮，按 val NLL 保存第一轮。独立模型训练约 301 秒；共享模型约 351 秒。两轮都在同一批 120 条 IP 校准记录（240 道题）上分别拟温度：独立 T=1.5848，共享 T=1.6237。

| 数据 | IP-only | joint |
|---|---:|---:|
| IP train / val / calib 记录 | 520 / 120 / 120 | 520 / 120 / 120 |
| 额外 r17 家居 train 回放 | 0 | 343 |
| IP 验证题最佳准确率 | 45.83% | 42.92% |
| IP 验证题最佳 NLL | 1.4043 | 1.4949 |
| checkpoint SHA256 前 16 位 | 9d09903cedd9562c | 2577c873a88e4f7b |

开发对照集 140 条、13 个全新家族，均为合成文本。表中“完整一步”要求动作和成员同时正确。反事实成对正确要求同一例子的两个分支都正确，且测试集中有 10 对。

| comparison 指标 | r14 零样本 | IP-only | joint |
|---|---:|---:|---:|
| 动作准确率 | 35.71% | 71.43% | 71.43% |
| 成员准确率 | 28.57% | 40.71% | 46.43% |
| 需要说话时的成员准确率 | 21.00% | 34.00% | 43.00% |
| **完整一步准确率** | **15.00%** | **31.43%** | **37.86%** |
| 反事实成对正确 | 0/10 | 1/10 | 2/10 |
| 澄清 / 等待 / 弃权完整一步 | 0/10、0/10、1/20 | 0/10、0/10、0/20 | 0/10、0/10、0/20 |
| 置信度两题均 ≥0.7 时 | 1 条且错误 | 0 条 | 4 条且全错 |

因此，虽然两个微调模型都比 r14 零样本强，**均未达到预设 90% 完整一步、80% 各动作类别和 80% 反事实对门槛**。joint 比 IP-only 的完整一步高 6.43 个百分点，但仍远低于候选标准。不能把任一模型交给 IP Team 运行时作自动决策；尤其不能通过置信阈值补救当前错误分布。

共享模型还在已使用过的家居开发集上做了描述性回归，不将这些集合重新称为独立验收：

| 家居历史集 | r14 全题 | joint 全题 | 主要回归 |
|---|---:|---:|---|
| v1 locked-182 | 99.35% | 99.35% | 预设 gate PASS |
| v2-dev | 96.45% | 94.92% | device 93.94% → 89.39%，预设 gate FAIL |
| v3-dev | 94.06% | 93.56% | 预设 gate PASS；并非新安全证明 |

**决策：维持 r14 为现有家居发布权重；IP Team 独立模型与共享模型都只保留为实验产物。** 当前结果也支持暂不让同一权重承担两种场景：共享模型没有解决 IP 准确率问题，并触发了家居 v2-dev 的设备回归门槛。IP 的 holdout.jsonl 未进行模型评估；没有因 comparison 错例修改数据或继续调参。

局限：训练/对照都是人工模板变化，没有真实 PTT/ASR、自然自由对话或用户确认数据；动作类别的独立家族数很少，不能以变体记录数替代独立样本量。Laya 决策头无法生成 SDK 在 clarify 时必需的 instruction 文本。后续可先在文本输入的影子链路中采集经同意的真实决策快照与人工标签（包括用户原话、已播出的公开发言、动态候选、约束和唯一可接受下一步），再冻结独立家族/会话测试集；若只有合成文本，继续扩大模板训练也不能证明真实场景可用。

## 可复现命令

从仓库根目录执行。报告脚本读取普通 Laya eval 的逐题行，另算完整一步和反事实对。

```bash
laya/.venv/bin/eidolon-laya-train gen --scenario laya/train/scenarios/ip-team --config laya/train/scenarios/ip-team/gen.yaml --out laya/train/runs/ip-pilot-ip-only/ip-team.jsonl
laya/.venv/bin/eidolon-laya-train gen --scenario laya/train/scenarios/ip-team --config laya/train/scenarios/ip-team/gen-comparison.yaml --out laya/train/scenarios/ip-team/eval/comparison.jsonl
laya/.venv/bin/eidolon-laya-train gen --scenario laya/train/scenarios/ip-team --config laya/train/scenarios/ip-team/gen-holdout.yaml --out laya/train/scenarios/ip-team/eval/holdout.jsonl
laya/.venv/bin/python laya/train/scenarios/ip-team/prepare.py --ip-source laya/train/runs/ip-pilot-ip-only/ip-team.jsonl --home-source laya/train/runs/r17/dataset/train.jsonl --eval-set laya/train/scenarios/ip-team/eval/comparison.jsonl --eval-set laya/train/scenarios/ip-team/eval/holdout.jsonl --ip-out laya/train/runs/ip-pilot-ip-only --joint-out laya/train/runs/ip-pilot-joint
# 以下训练、校准、评估需在 MPS 可用的 Mac 上顺序执行。当前 checkpoint 已存在时勿覆盖。
laya/.venv/bin/eidolon-laya-train train --config laya/train/scenarios/ip-team/train-pilot.yaml --dataset laya/train/runs/ip-pilot-ip-only/dataset --out laya/train/runs/ip-pilot-ip-only/checkpoint --device mps
laya/.venv/bin/eidolon-laya-train train --config laya/train/scenarios/ip-team/train-pilot.yaml --dataset laya/train/runs/ip-pilot-joint/dataset --out laya/train/runs/ip-pilot-joint/checkpoint --device mps
```

两模型各自对同目录 dataset/calib.jsonl 运行 calibrate，再对 comparison.jsonl 运行 eval；逐步结果、门槛及原始行保存在各自 run 的 eval/ 目录。holdout.jsonl 的 SHA256 为 176abd946d4c7ae58fae0bca60266467e899d8c5fd394d246464eb11ed5ebb60，尚未用于模型评估。
