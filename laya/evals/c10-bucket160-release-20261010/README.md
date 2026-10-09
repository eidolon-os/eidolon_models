# c10 160 档发布验收（2026-10-10）

目标：将固定权重 c10 `e8254243` 的 160-token FP16 档位投入生产，补齐推理记录和已观察到的会话语义问题。不训练、不降低阈值、不改变三轮上下文/一秒解释预算。

## 发布前证据

- 独立 HTTP 服务，端口 18774；生产 8771 保持原配置。验收约 578.8 秒。
- 800 次串行：800 成功，P50 677.3ms、P95 782.8ms、最大 864.6ms；无超过 1 秒。
- 30 组两客户端并发：30 成功、30 快速 503，成功请求最大 851.7ms。max_pending=1 的拒绝策略保持不变。
- 与 participation 服务同时进行合成分类：两边各 30 次全部成功，家居侧最大 868.7ms，participation 最大 521.2ms。这里只验证两个 NPU 服务共存，未调用设备执行接口或真实 participation 业务决策。
- RSS 预热后 1,821,156 kB（约 1.74 GiB），第 50 到第 800 次增加 264 kB；可用内存始终超过约 7 GB，SwapFree 没有下降。这是约十分钟验收，不是数日稳定性或完整内存泄漏证明。
- 候选 Agent + 真实 NPU + scripted oracle fallback：110/110 通过，包括 20 会话、36 独立、54 上下文安全；20 会话中 9 次直接完成，模型耗时中位数 617.37ms、最大 814.81ms，0 超时/错误。执行器在内存中。
- 单独测试真实配置 LLM（强制 primary abstain）：20 条会话 × 3 次，60/60 指令目标/动作/参数匹配；6 条否定、转述、未来请求及取消 × 3 次，18/18 未执行设备动作。
- 真实 Laya 160 档 + 真实配置 LLM + 内存执行器：同一套 20 条会话 × 3 次，60/60 通过；6 条安全负例 × 3 次，18/18 不执行。见 `behavior-chain.json`、`behavior-chain-negative.json`。
- 初版提示词测试曾在“关了吧”出现 2/60 取消误判；最终补充设备关闭与取消的区别后，完整重跑 60/60 通过。原洗衣机“关了吧？”语气意图不明确，合成测试改为明确“停止洗衣机”，不声称复原原始录音语义。
- Agent 相关单元测试 347 项与配置合约 4 项通过；Models 配置/引擎/NPU 46 项和启动器合约 6 项通过。

具体证据见 `http-validation.json`、`agent-replay.json`、`behavior-candidate.json`、`behavior-negative.json`、`concurrency.json` 和 `coexist.json`。前期 326 输入/778 分类一致性、概率最大漂移 0.0114 见相邻 `c10-efficiency-20261009`。

## 变更

1. 新制品根 `/var/lib/eidolon/models/laya-smart-home-c10-b160-rknn-e8254243`，包含 128/160/256/384/512 档。原 c10 制品根不覆盖；新图 SHA256 固定为 `591f6cf439e673f6c94c2630b65fd5b5cbc3e591f8747ad6420051cc6480de45`。
2. 核 1:128/160/256/384/512，核 2:128/160/256；核 0 仍归 participation。不采用上一轮穷举调度实验。
3. Agent 开启本地 JSONL 解释记录（含实际上下文、top3、模型/策略、提案与耗时），0600 权限，8 MiB × 4 文件的大小轮转。LLM 完整提案、实际下发参数与 Provider 回执关联 turn ID 记录。
4. pending 状态下，只有本句完整命令达到原直接阈值、且 pick 高置信度要求重新理解时，允许处理新的完整命令。冲突、取消或不确定仍兜底。
5. LLM 提示词明确绝对值/增量、唯一近期设备焦点、重复关闭和取消语义；不增加语义规则入口。

## 复跑与边界

`serve.py` 启动临时 HTTP 候选；`validate.py` 运行持续/并发/共存测试并采样资源。`behavior.py` 使用真实配置 LLM、合成家居和 `home_fakes.py` 内存执行器；加 `--laya-url` 可把真实 Laya 纳入链路。`replay.py` 保留固定 110 条回归，fallback 为 oracle，不把其结果冒充真实 LLM 准确率。

这里没有测试 Korvo 录音识别、端侧播报 ACK 或真实家电物理效果。提前启动 LLM 尚未实现：本轮先落地已验证的档位和可观测性，后续需根据实际生产记录确定可提前交接的证据边界。

## 生产发布结果

2026-10-10 00:24（Asia/Shanghai）正式激活 `rk3588-c10-b160-20261010-1`，Ops 返回 `activated/applied`，doctor 为 `healthy`，App readiness 的全部检查通过。

- Agent 提交 `3808351fa7c80a221fba30598ae1653cd05ffc1a`；Models 运行时提交 `29797b8716c0be697f299b9a8f15ebeb50c98c39`。完整固定版本组合见 `matrix.json`，未将其他未提交工作混入发布。
- 生产 8771 `/v1/info` 确认固定 c10 `e8254243`、RK3588 NPU、FP16、128/160/256/384/512 档及核 1/2 分配正确。
- 以生产 `eidolon` 身份执行 `smoke_recording.py`：诊断解释成功，模型版本正确，记录写入 `/var/lib/eidolon/agent/interpretations/smarthome.jsonl`，0600 权限；未调用设备执行器。
- 发布及模型、就绪检查的最小证据见 `production-summary.json`；发布过程中生成代码与 Host 配置快照。
- 本轮未提前启动 LLM。下一步先观察真实语音链路各段耗时，再判断是否值得引入并发推测或提前交接，避免增加成本却不改善整体响应。

发布工具清理了上一版代码目录后，已通过正式 Ops prepare 重新准备 `rk3588-peer-admin-20261009-2` 的固定原版本，未激活。见 `rollback-prepare-summary.json`。旧 c10 制品根也保留；本轮未实际执行回滚演练。
