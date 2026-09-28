# Ensemble 小脑：第一批原创对话试验（预登记，2026-09-28）

## 目标与边界

按 Ensemble 页面六种体验中的团队对话部分，预测一次公开文本快照的下一步：`respond:<当前候选>`、`clarify:<当前候选>`、`wait`、`finish`、`abstain`，以及发言任务 `answer/propose/build/challenge/compare/synthesize/comfort/clarify/none`。联合的 move 选项保证不会出现“选了沉默却又选了成员”这类跨题不一致；task 仍需在服务端按动作校验。候选 M0 等只是每次请求内随机排列的位置。每次最多一位成员发言，具体台词由角色模型生成。Laya 不负责生成 SDK 所需的自由文本 instruction，也不控制取消、权限、会话生命周期或音视频。

训练材料只来自本目录的原创对话片段：点名、开放讨论的内容接力、同一成员续说、用户插话改目标、安静陪伴、澄清、等待和收束。`heard` 仅代表已经公开完成的发言；中断后的旧发言不能作为新的完成事实。[Ensemble 页面](https://eidolon.aimanthor.com/ensemble/)与本地 `eidolon-official-site` 页面源码是产品语义依据。[CPED](https://github.com/scutcyr/CPED)、[MPDD](https://github.com/ntunlplab/Dialogue-MPDD)、[AMI](https://groups.inf.ed.ac.uk/ami/corpus/) 等外部语料仅用于研究语言和话语行为类别；本试验没有复制这些语料，因为没有“当前候选中谁应当下一步说话”的金标，影视对白也另有授权问题。

## 数据、配方、对照

数据在模型评估前冻结：`train.yaml` 32 个完整对话家族，65 个独立步骤，4 次姓名与候选顺序置换，共 260 条；`val.yaml` 8 家族 13 步 ×3 =39 条；`calib.yaml` 8 家族 13 步 ×3 =39 条；`comparison.yaml` 12 家族 22 步 ×3 =66 条。另封存 `holdout.yaml` 8 家族、11 步 ×3 =33 条，本次不用于训练、校准或首轮模型选择；生成文件 SHA256 为 `66897bfe4856d05a51c6be723ec7b24e2d424e8222c4fac41d35f0730709232d`。各分组的家族、具体文本和姓名池隔离，且同一对话所有连续步骤在同组；“点名”“等待”等语义类别刻意跨分组出现，因此不能称为主题完全隔离。训练只是小批量诊断：独立样本数按家族或步骤计，不能把变体行数当作 260 个真实对话。比较集仍是合成文本，不是独立用户验收。

从 r14 家居发布 checkpoint **复制初始化**，只训 Ensemble 数据；3 轮、batch 16、末 8 层解冻、encoder 1e-5/head 5e-5、seed 41，按 val NLL 取最优轮。校准只用 calib，温度对 comparison 没有调参权限。家居 r14 发布权重和历史 IP-only/joint 权重原样保留。对比同一个新 comparison 的 r14 零样本、旧 IP-only，以及本次新模型；旧模型没有训练过 task，因此还需分别报告 move 与完整 move+task，不把新增题型导致的旧模型劣势伪装为同口径改进。

## 冻结的判断口径

- 对 66 条 comparison 报告 move、task、完整一步的准确率和计数，逐动作类别、逐场景模式报告，额外报告按家族全部步骤正确的比例。`respond/clarify` 的成员准确率单列。
- **仅把小试验视作有改进**：move 准确率相对旧 IP-only 至少提高 10 个百分点，且完整一步准确率至少 60%；两项只做合成诊断结论，不据此上线。若未达到，客观报告失败，不用 comparison 错例回写本批训练数据。
- 所有需要 `wait/finish/abstain` 的静默场景均单列错误发言数；任何关键场景误发言都会阻止进入实际自动调度测试。即使全对也不足以证明真实自由对话安全。
- 不沿用已经用于 r14/r15/r16 选型的家居验收集调参，也不将本次 synthetic comparison 称为用户金标。下一步是否值得集成，需另行收集经同意的真实文本对话、候选和人工审核的下一步标注；没有 ASR 也可以先从键入文本和影子模式开始。

## 数据质量检查（训练前）

四组 ID 无重复，家族互斥；每条都有 move 与 task 金标，且两题均可完整编码，没有选项因 1024/384 长度窗口丢失。最长编码 574 tokens。动作分布（train / comparison）：respond 184/42、clarify 12/3、wait 24/6、finish 32/12、abstain 8/3。澄清和弃权各只有少数独立家族，其切片指标波动会很大。`abstain` 的本批样本仅测试没有可用动作时的出口，不能代表语义不确定时的弃权能力。

## 结果

已完成，checkpoint 采用第 2 轮（val NLL 1.7101、逐题准确率 27/78=34.62%），三轮累计训练 172.7 秒。校准用 39 条独立记录、78 题，choice 温度 1.4135；整体 calib NLL 从 1.8452 到 1.8137。checkpoint `model.safetensors` 约 1.2 GB，SHA256 `98b67434c16a02c16a69ea41906e07828f32b42bed5446be94c156b08b152925`。在 Mac MPS 上，对一个 3 成员、407/321 tokens 的两题状态，加载 2.51 秒，热启动 30 次重复推理中位数 75.5 ms、p95 84.0 ms；这不是 RK3588 延迟或端到端语音延迟。

同一个冻结 comparison（66 条，12 个家族）结果如下。完整一步要求 move 和 task 都正确；旧 IP-only 未见过新的 task 题型，所以完整一步只作端到端参照，move 是可比的主要指标。

| 指标 | r14 零样本 | 旧 IP-only | 本次 v2 |
|---|---:|---:|---:|
| move（动作+成员） | 16/66，24.24% | 24/66，36.36% | **25/66，37.88%** |
| task | 16/66，24.24% | 18/66，27.27% | **23/66，34.85%** |
| 完整一步 | 8/66，12.12% | 6/66，9.09% | **8/66，12.12%** |
| 需发言时选对成员 | 4/45，8.89% | 16/45，35.56% | **18/45，40.00%** |
| 应静默时错误要求发言 | 1/21 | 6/21 | **13/21** |
| 整个对话变体所有步骤正确 | 2/36 | 1/36 | **3/36** |

本次 v2 的完整一步逐动作：respond 3/42、clarify 0/3、wait 0/6、finish 5/12、abstain 0/3。`abstain` 的 move 本身在 3/3 条只有一个合法选项，完整一步仍因 task 错误变成 0/3；不能据此认定模型会在不确定时弃权。错误分布显示 6 个 wait 全被预测成 respond，12 个 finish 中 7 个被预测成 respond，3 个 clarify 全被预测成 respond。训练数据自身的逐题准确率只有 160/520=30.77%，表明此设置连作者数据都未充分拟合；不能把问题只归咎于新对话泛化。

**预定门槛失败**：move 比旧 IP-only 只高 1.52 个百分点，低于 10 个百分点；完整一步 12.12%，低于 60%，且静默误发言 13/21。此 checkpoint 只保留为研究诊断，不打包为发布权重、不进入 IP Team 自动调度、不修改 r14 家居 release。未打开封存 holdout 做模型评估，也不根据 comparison 错例回写本批训练数据。

下一轮需要重建数据任务：独立作者审核每个状态的唯一可接受动作、可接受成员集合和发言任务，明确“只等用户”和“结束当前讨论”的边界；增加真正不同的会话家族，尤其是澄清、安静、已足够、目标打断和同一成员续说，控制 respond 与 propose 的比例。应先在 train/val 上证明可学习，再冻结新的家族测试集。当前的 32 个训练家族、65 个步骤太少，且任务类别多、对话标注可能主观；简单增加姓名置换不会增加语义覆盖。若用户没有 ASR，可先用真实打字对话或同意录入的手工场景做盲标；只靠合成语料仍不能证明实际自由对话准确性。

## 复现与产物

训练配方为本目录 `train-pilot.yaml`，四组生成配置为 `gen-*.yaml`。从仓库根目录依次执行 `laya/.venv/bin/eidolon-laya-train gen --scenario laya/train/scenarios/ip-team-v2 --config laya/train/scenarios/ip-team-v2/gen-<split>.yaml --out <对应 jsonl>`；具体输出路径在 `train/runs/ip-ensemble-v2/dataset/`（train/val/calib）以及本目录 `eval/`（comparison/holdout）。随后 `train --config laya/train/scenarios/ip-team-v2/train-pilot.yaml --dataset laya/train/runs/ip-ensemble-v2/dataset --out laya/train/runs/ip-ensemble-v2/checkpoint --device mps`，再对独立 calib 执行 `calibrate`，对 comparison 执行 `eval`。已存在 checkpoint 时应改新 run-id，不能覆盖。

模型报告在 `laya/train/runs/ip-ensemble-v2/eval/comparison.json`，同目录 `r14-baseline/`、`old-ip-baseline/` 保存对照原始逐题结果，`summary.json` 由本目录 `report.py` 计算。训练 checkpoint 与 eval 报告保留在被 `.gitignore` 忽略的 run 目录，本目录的 authored YAML、生成器、生成配置、比较集和封存集保留为源文件。冻结 comparison JSONL SHA256 `46c3a509e8d108ad21397286a25df752d457420d0d5572551be1b29f8a8f3608`。
