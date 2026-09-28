# Laya 真实使用数据与独立验收方案（2026-09-27）

当前 r14–r16 的训练和评测主要来自撰写、增强及外部文本。仓库中未发现可直接用于本验收的真实家庭语音转写。下列方案用于取得能回答“是否可以免确认执行”的证据；现有合成验收集已参与选型，不再充当新模型的独立最终验收。

## 数据很少时先做试点

不需要先攒齐正式样本才开始。先用 r14 作**确认后执行**的历史候选，在实际 ASR 链路中只记录预测、不下发设备动作。连续收集手头能取得的几十条最终转写，包含无关话语、查询和控制；即使只有一个家庭，也可以用于发现错字、设备清单不匹配、误把陈述当命令等错误。记录采样时段和纳入规则，不能只挑模型答错或“像命令”的句子。

无需预先准备转写服务：本仓库的 `eidolon-asr` 已包含流式、离线修订和标点模型，在 Mac 上可本地运行。`./scripts/eidolon-asr doctor` 检查模型；`./scripts/eidolon-asr infer <16kHz 单声道 PCM16 WAV>` 可从自愿录制的短音频得到最终文本。2026-09-27 本机 `doctor` 与仓库示例 WAV 推理均通过。**示例 WAV 是链路冒烟样例，不能算真实家庭语料**。如果连自愿录音也暂时没有，就先做文本场景开发和确认策略，真实语音验收保持未完成。

```bash
cd laya
uv run --extra train eidolon-laya-train real-prepare \
  --events train/private/pilot-events.jsonl \
  --scenario train/scenarios/smart-home \
  --out-dir train/private/pilot-001 --pilot-only
```

`--pilot-only` 将全部记录放在 `dev.jsonl`，允许单一家庭，不生成 `test.jsonl`，也不能运行 `real-gate`。小样本可用于人工错误分析和开发集阈值探索；不能当独立验收。真实录音经实际 ASR 重放可查转写鲁棒性，但朗读脚本或 TTS 音频不能替代自然背景谈话的误触发率。

为什么仍需后续积累？若观察到 **0 次**错误，50 条无关话语的 95% 单侧精确上界仍约 **5.82%**，100 条约 **2.95%**；50 条候选执行全部语义正确，正确率下界仍仅约 **94.18%**。这是二项模型下的界，家庭内相关性还会降低有效样本量。试点能帮我们修问题，不能证明原定的 2%／99% 安全标准；正式验收仍按下文预登记的 400／300 门槛与跨家庭切分执行。精确二项界方法见 [NIST 统计手册](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/exacbici.htm)。

## 数据单位与采样

- 以一次完整的说话事件为单位，记录 ASR 最终文本、匿名家庭及会话 ID、当时的设备清单快照、ASR 版本与时间段。音频如需用于错字分析，应单独保管；训练仓库只接收经许可和脱敏的文本与必要上下文。
- 从固定时段连续抽取事件，保留背景谈话、对家人或宠物说话、设备播报、转述和取消指令，避免只收“像命令的句子”。另建困难负例挑战集，但与自然抽样的发生率报告分开。
- 覆盖不同户型、设备数量、说话人和 ASR 条件。按家庭与会话分开发集、最终验收集；同一事件的转写修订、增强版本和近似副本不能跨集合。
- 第一批至少准备 400 条真实无关事件和 300 条实际满足候选自动执行条件的事件。随后按预先约定的精确二项置信区间扩充样本：无关直接执行率的 95% 上界须 ≤ 2%，自动执行正确率的 95% 下界须 ≥ 99%；样本不足时明确报告“证据不足”，不以点估计宣布达标。

## 标注

- 每条记录标注意图（控制／查询／无关）、是否明确对助手发话、设备与动作可接受集合，以及“含糊、需要确认”标记。控制句的设备和动作若无法可靠标注，记录为未验证，不能算自动执行正确。
- 对所有候选免确认执行事件及随机抽取的背景事件进行双人独立人工标注，分歧由第三人裁定。特别审阅“我要洗澡了”“宝宝醒了”等隐含意图：其自动化许可属于产品政策，应与语义金标分开记录。
- 记录标注规范版本、标注者及裁定状态。只有完成裁定的最终验收记录进入安全指标分母。

## 冻结与评测

1. 在查看最终验收模型输出之前冻结记录 ID、分组、标签、阈值和执行政策，并保存数据与配置哈希。
2. 只在开发集选择 `p(控制)`、设备和动作置信度阈值；最终验收集只运行一次。模型走实际 `ask_if`/推测并行路径，以干跑方式记录会否下发设备动作，不实际控制设备。
3. 分别报告无关→控制分类误触发、**无关直接执行率**、自动执行正确率与覆盖率、误执行动作、需要确认比例、漏掉真命令比例；附置信区间及按家庭、ASR 条件、隐含意图和高后果设备的切片。
4. 同一候选在 RK3588 上与 PyTorch 对拍，并报告端到端延迟及与 TTS 并发时的影响。模型和阈值都通过后再考虑扩大免确认范围。

在满足上述条件前，真实用户请求采取“模型建议、确认后执行”；含糊陈述和高后果动作继续要求确认。

## 可执行的数据导入入口

仓库现有 ASR 与 Laya 是分开的模型服务；采集必须在实际串联 ASR、家庭设备状态和执行策略的应用侧完成。应用侧保持设备执行关闭（shadow），连续导出**最终 ASR 文本**和同时刻设备选项。导入工具不会读取音频、调用设备或启动训练。原始导出、标签和冻结结果放在 `laya/train/private/`（Git 已忽略）或受控外部目录；不要把真实家庭数据写入仓库可提交路径。`approved_for_evaluation` 是应用侧许可记录，工具只检查其值，无法代替许可或脱敏审查。

事件 JSONL 每行示例（这里的数据是虚构的）：

```json
{"event_id":"evt-001","household_id":"anon-home-01","session_id":"session-09","captured_at":"2026-09-27T10:00:00+08:00","asr_model":"asr-version","asr_text":"把客厅灯打开","devices":{"客厅灯":"客厅·灯"},"approved_for_evaluation":true}
```

- `household_id`、`session_id` 和 `event_id` 必须是稳定的匿名标识；`event_id` 不能重复。`devices` 要使用**该时刻**实际呈现给 Laya 的名称和说明，不可事后按模型预测修改。
- 连续采样时保留无关话语、查询和控制话语，记下采样起止与纳入规则。导出前人工检查姓名、地址、电话等敏感内容；自动替换可能改变语义，替换规则须记录并在两个集合一致使用。
- 转写修订不能当成新事件。不同家庭不能共用一个 `household_id`；工具按家庭切分，并拒绝只有一个家庭的导出。

人工裁定标签单独放在 JSONL，每行示例：

```json
{"event_id":"evt-001","status":"adjudicated","guideline_version":"v1","addressed_to_assistant":true,"needs_confirmation":false,"gold":{"intent":"控制","device":"客厅灯","action":"打开或启动"}}
```

`gold` 的设备和动作必须属于当时选项；控制句应填满三题。无关句只填意图。实际标注表还需在受控目录保存两位标注者的独立原始答案、分歧和裁定人记录；此导入文件只接收裁定结果。`needs_confirmation` 是政策判断，应与三题语义金标区分。

```bash
cd laya
uv run --extra train eidolon-laya-train real-prepare \
  --events train/private/events.jsonl \
  --labels train/private/adjudicated.jsonl \
  --scenario train/scenarios/smart-home \
  --reference train/runs/r14/dataset/train.jsonl \
  --reference train/runs/r17/dataset/train.jsonl \
  --reference evals/smart-home-v2 \
  --reference evals/smart-home-accept \
  --out-dir train/private/freeze-001 --require-complete
```

输出 `dev.jsonl`、`test.jsonl` 和 `manifest.json`。目录必须事先不存在，清单记录输入／输出 SHA256、家庭分组、数量和指定参考集的**去空白精确语句重合**；它不能发现近义复述。常见命令的重合属实，应报告，不能将全部事件宣传成模型未见语句。实际审计需把所有相关训练版本与历史评测文件都列入 `--reference`。固定整批事件、标签及分组后，把验收集封存；开发集可用于选择阈值，验收集只在模型和阈值冻结后评测一次。若仅先检查未标注数据，可省略 `--labels` 和 `--require-complete`；这时不能据报告宣布安全达标。最终评测仍需检查近似重合、原始采样覆盖、置信区间、设备执行干跑和平台对拍。

验收时先在开发集固定模型和阈值，再对 `test.jsonl` 运行一次 `eval`，最后运行 `real-gate`。下例的阈值仅是**命令占位示例**，实际值必须来自开发集并事先冻结：

```bash
uv run --extra torch --extra train eidolon-laya-train eval \
  --checkpoint train/runs/r14/checkpoint \
  --eval-set train/private/freeze-001/test.jsonl \
  --out train/private/freeze-001/reports
uv run --extra train eidolon-laya-train real-gate \
  --freeze-dir train/private/freeze-001 \
  --report train/private/freeze-001/reports/test.json \
  --tau-intent 0.95 --tau-slots 0.95 \
  --out train/private/freeze-001/safety-result.json
```

`real-gate` 用 95% 单侧精确二项界检查：无关事件直接执行率上界 ≤ 2%、候选免确认执行的**三题语义正确率**下界 ≥ 99%，同时要求至少 400 条无关事件和 300 条实际自动执行候选。需要确认或并非对助手发话的事件即使三题语义正确，也不能算免确认执行正确。缺失金标、模型／校准配置哈希、报告与冻结集不符或样本不足时不会给出通过结论。即使返回 `MODEL_DECISION_GATES_PASS`，也只说明模型语义和该确认策略的统计条件通过；Laya 的动作选项仍是粗粒度类别，尚未验证实际工具调用参数、设备响应及物理后果。上线免确认执行仍需人工审查切片、产品策略和设备侧干跑／实测。
