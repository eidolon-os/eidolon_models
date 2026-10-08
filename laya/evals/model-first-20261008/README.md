# Laya 优先解释：Mac 验证报告（2026-10-08）

本次完成代码、契约、文档和自动化回归；未训练、未更换权重、未在 OPI5 Max 测试或部署。板端当前版本为 c4 `7b695ba8`；c10 `e8254243` 是最后训练的候选，并非板端已验收版本。本次分别使用两者在 Mac CPU / Torch FP32 推理，不据此建议切换板端权重。

## 交付行为

- Laya 模式下移除 RulesInterpreter 前置语义筛选，所有获授权输入先进入 Laya。设备、权限、能力、参数、幂等与执行回执校验继续保留。
- 默认保留最近 3 轮已完成对话，可配置 1–5 轮，保留 30 秒上下文有效期。当前句问题与上下文问题分别构造输入，在一次请求中批量推理；不把完整 history 塞入所有问题。
- Laya 无法可靠完成时，把候选、得分、模型版本、升级原因和有界上下文交给 LLM；LLM 可修正或拒绝候选。旧服务不支持逐问题 state 或输入截断时不直接执行。
- 相对风速改用 Provider 基于当前状态执行 delta；重复命令继续增减，重试保持幂等。取消已完成操作明确说明未撤销。连续完整输入按会话 FIFO 执行，并分别记录排队、解释、Agent 返回和结果投递耗时。

架构与上线契约见 [Agent 文档](../../../../eidolon_agent/docs/smart-home-model-first.md)、[SDK 契约](../../../../eidolon_sdk/contracts/smarthome/v1/model-context.md)、[Hub ADR](../../../../eidolon_hub/docs/adr/20261008-relative-fan-speed.md)、[Channel 耗时口径](../../../../eidolon_channel/docs/smarthome-turn-timing.md)。

## 真实模型回放

`cases.json` 包含原 20 轮 ASR 文本顺序回放、36 条独立输入回归，以及其中 27 条上下文负例各在焦点/待选设备两种状态下的 54 次检查。测试案例参与了开发调试，属于回归集，不是未见过的泛化评测集。

| 固定权重 | 20 轮状态/命令回放 | Laya 直接完成 | 独立输入检查 | 上下文负例检查 | 模型耗时中位数 / 最大值 |
|---|---:|---:|---:|---:|---:|
| c4 `7b695ba8` | 20/20 | 6/20 | 36/36 | 54/54 | 433.46 / 577.77 ms |
| c10 `e8254243` | 20/20 | 9/20 | 36/36 | 54/54 | 425.75 / 575.86 ms |

每版均真实调用模型 110 次。20 轮输入均先经过 Laya；其余分别 14/11 轮进入兜底。**兜底是按预期返回的测试替身，执行器也是内存替身**：20/20 证明模型与应用路由、状态、命令交接符合这些用例，不能当作真实 LLM 或设备端到端准确率。独立输入检查允许模型拒答，验证已接受命令正确、负例不误执行，不代表 36 条均被模型独立解决。

相对风速回放使用合成初始值 30，检查连续变化 `30 → 40 → 50 → 60 → 50`，不是恢复板端历史真实初始状态。

c4 额外比较 1/3/5 轮历史：均为 20/20、6 次直接完成。1 轮模型中位数 434.98 ms，5 轮 580.77 ms；运行条件与样本规模不足以推出稳定性能排序或证明 3 轮最优。原板端审计见 [20 轮记录](../../research/korvo-1-review-20261008/report.html)，旧路径仅 6 轮调用 Laya、3 轮直接完成；新旧不是同硬件受控延迟对比。

完整输出： [c4](c4.json)、[c10](c10.json)、[c4 一轮](c4-window1.json)、[c4 五轮](c4-window5.json)。

## 自动化测试

在实际工作目录运行，共 **634 项通过**：Agent 419、SDK 85、Hub 85、Channel 15、Models 30。覆盖默认模型入口、有界上下文、过期与截断、矛盾候选、证据交接、FIFO、取消反馈、相对风速边界、持久状态及重试幂等。Hub 的真实 Home Assistant bench 测试需要在线实例，本轮明确排除，未计入通过数。

在各仓库使用其 `.venv/bin/python -m pytest -q --tb=short -p no:cacheprovider`，并将当前仓库及同级 SDK 加入 `PYTHONPATH`，测试路径如下：

- Agent：`eidolon_agent/domain/smarthome/tests eidolon_agent/infra/smarthome/tests eidolon_agent/infra/interpretation/tests eidolon_agent/app/smarthome/tests eidolon_agent/domain/interpretation/tests`
- SDK：`tests/interpretation tests/smarthome`
- Hub：`tests/smarthome --ignore=tests/smarthome/homeassistant/test_bench.py`
- Channel：`eidolon/livekit/tests/agent/test_smarthome_transport.py eidolon/livekit/tests/agent/test_smarthome_idle.py eidolon/livekit/tests/agent/test_smarthome_startup.py`
- Models（laya 目录）：`tests/test_engine.py tests/test_service.py`

真实回放：在本地分别启动指定 revision 的 Laya Torch CPU 服务，再用 Agent 环境运行 `replay.py --url http://127.0.0.1:<port> --revision <revision> --output <result.json>`；上下文比较加 `--context-turns 1` 或 `5` 和 `--conversation-only`。脚本验证服务模型版本。离线模型 deadline 为 10 秒，生产默认仍为 1 秒。

## 发现与剩余缺口

直接把全部历史送给所有问题会扰动原有单句分类，因此采用逐问题 state。移除词法见证后，c4/c10 都暴露过高置信度引用/假设命令误判；未经训练的新“是否授权”问题也不可靠，没有采用。最终明确设置保守接纳下限：直接结果 0.99，上下文 pick/follow 0.95，follow 还要求当前控制意图 0.99。分数不是校准后的正确概率，这些阈值降低直接覆盖率，也不能保证泛化安全。

本轮不继续训练。当前仍需后续独立验证：真实 LLM 是否有效利用交接证据、不同家庭/设备目录和更广的否定/引用/跨设备场景、真实 ASR 与 VAD 提交延迟、NPU 多问题批处理耗时及 1 秒 deadline。Mac 数据不能证明 NPU 或端到端体验已达标。Home Assistant 相对调整依赖读后写，对外部控制器并发修改不具备原子性。

上述板端工作留待另行决定；本次没有执行。跨仓库代码上线需配套 SDK、Agent、Hub、Channel 与支持 `features.question_state` 的模型服务。

## 配套提交

SDK `acf5ce6`，Agent `fc9de67`，Hub `83cf4bd`，Channel `ba5f8cf`。Models 的服务契约、回放脚本、原始结果与本报告一起提交。仅本地提交，未推送。
