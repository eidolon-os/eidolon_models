# Laya 参与决策 p4：部署版本与评测交付（2026-09-29）

范围：`eidolon_models` 的 Laya 模型服务、部署重构与模型评测。家居 `/v1/systemone`、参与决策 `/v1/participation/decide`
接口与 SDK 契约（participation v2、interpretation v1）均未改动。Agent 侧（LLM 兜底、组合入口、预算 / 取消 / 状态校验、家居上下文接入）不在本交付内。

## 1. 部署对象、状态与 NPU 数据

**状态：已在 opi5max 上按部署形态验证，尚未经 Ops 正式发布。** 验证用的是板上临时实例：同一启动脚本
`scripts/eidolon-laya-participation`、同一工件字节（`laya-participation-p4-rknn-ae6718a4`）、同一代码，
端口 8773、NPU 核 0，环境里故意带着家居服务的 `EIDOLON_LAYA_*`（启动脚本全部不继承）。正式发布会重启 eidolond
与家居 laya，需先协调窗口（见文末）。

| 项 | 值 |
|---|---|
| 模型 | `laya-participation` revision `ae6718a4`（release p4，`laya/train/scenarios/participation/EXPERIMENTS.md`） |
| Profile | `participation.json` sha256 `85063d9d0f369a8d1d0539ea729c8597a16ae7c6f995e52e5e313dca7f69cf7c`，schema 2，state_format `participation-laya-v1` |
| policy_version | `participation-laya-v1/ae6718a4/min0.95` |
| 阈值 | `min_confidence = 0.95`（动作题校准后置信度；在 p-dev 上选定，p-test 未参与） |
| 其他约束 | 候选 ≤ 5；本轮须完整落在最近 16 条公开记录内；输入 ≤ 640 token；超出一律 abstain |
| NPU 形态 | 核 0，档 384 / 512 / 640（256 并入 384），分阶段（先动作，回应 / 澄清再问人选），`MAX_PENDING=1`，OpenBLAS 1 线程 |
| 工件 | `ops/component.toml` 的 `laya-participation-p4-rknn-ae6718a4`（15 个文件逐个 sha256）；`hidden_l384.rknn` `5d6f10c3…`、`hidden_l512.rknn` `8f691f6f…`、`hidden_l640.rknn` `b775cf5f…`，`librknnrt.so` 2.3.2 `d31fc19c…`；无 NPU 主机用 `…-onnx-ae6718a4` |

**源码提交**（相对板上正在运行的 `eidolon_models 7e71d6a`，可部署代码只多出以下提交，全部属于本工作）：

| 仓库 | 提交 | 内容 |
|---|---|---|
| eidolon_models | `1fe1f4d` `5cb6288` | 三题参与决策适配器（participation-laya-v1）；本轮须落在训练窗口内、clarify 任务可在 profile 关闭 |
| eidolon_models | `0378bbb` | 两个独立服务：参与决策 unit / 启动脚本 / 端口 8773 / 能力 `local_laya_participation` / 工件；NPU 分核（家居 1+2、参与决策 0）；同档跨核共享权重；OpenBLAS 1 线程 |
| eidolon_models | `d3421b7` | 服务日志记录弃权原因（关联 ID、版本、动作、原因、置信度、耗时，不记对话） |
| eidolon_models | `15cb218` `25f925a` | NPU 基准工具（`laya/deploy/rk3588/laya_npu.py`，不部署） |
| eidolon_ops | `fa7f351` | 能力 / unit / 端口 / 就绪探针；`participation.url` 指家居端口被拒；rknpu2 主机的服务账号入 `video` |
| eidolon_kernel | `e3e4a35` | 发布契约、打包、eidolond 服务清单、就绪检查（`/participation/readyz`） |

**NPU 就绪**（板上实测）：

```
GET /v1/participation/readyz → {"status": "ready", "task": "ip_team.participation", "schema_version": 2,
  "policy_version": "participation-laya-v1/ae6718a4/min0.95", "model_version": "ae6718a4"}
```

NPU 映射 894 MiB（3 个图，核 0）；每个决策进程 CPU 70 ms、3 个线程；与常驻 NPU 的家居模型同时加载无地址空间报错。

**延迟**（p-dev 623 个决策点逐个请求，服务端计时，`board-scratch-20260929/summary.json`）：

| 全部 | 等待 | 结束 | 回应 | 澄清 | 弃权 |
|---|---|---|---|---|---|
| p50 696 / p95 1,244 ms | 450 | 624 | 881 | 1,277 | 673 |

同一发布对家居服务的影响：NPU 从三核改为核 1+2，样例请求 p50 337–356 → 454 ms；每请求 CPU 1.9 s → 67 ms（OpenBLAS 空转修复）；同档权重共享。

## 2. p-dev 逐条结果与真实拒答集合

文件（本目录 `board-scratch-20260929/`）：

| 文件 | 内容 | sha256 |
|---|---|---|
| `items.jsonl` | 623 行：id、金标、部署结果（status / action / participants / instruction / 原因 / 置信度 / 服务端耗时）、PyTorch 参照、正确性 | `2fb3264c04e351eac45c1cb7ecdfea34d1f7fd5deecfa73c0ee67476436a4bf7` |
| `responses.jsonl` | 服务的原始 HTTP 响应 | `d592a5138655ae19c3fd6cafce5a75024ce22a2de8bc6a9311cda8e4f0d302f1` |
| `service.log` | 服务日志（每个决策一行原因） | — |
| `summary.json` | 汇总、拒答 ID、差异、哈希 | — |

输入：`evals/participation-laya/dev/dev-companion.jsonl` `9b5a32ab6e017775fcc9b0f692101d45e8aacde32e4d9d844dff60d2423beebd`、
`dev-team.jsonl` `2990a14fa2db755f61cf3c78d491816b3587445dbd37e85ddfa1bf2b7a23a0cd`；每个 episode 的 variant-0 候选顺序（与离线评测同一状态）。
复现：`laya/train/scenarios/participation/deploy_report.py`。

| 指标 | 值 |
|---|---|
| 决策点 | 623 |
| 决定 / 弃权 | 558（89.57%）/ 65（10.43%） |
| 决定中动作正确 | 556 / 558（99.64%） |
| 决定中端到端正确（动作 + 人选 + 澄清任务） | 551 / 558（98.75%） |
| 决定中"应停却发言" | 0 |

**拒答集合**：65 个，原因**全部是 `low_confidence`**（置信度 0.42–0.949，中位 0.90；没有上下文不支持、截断或非法提案）。
金标分布：结束 33、回应 23、澄清 6、等待 3；团队 42、陪伴 23。ID：

```
DC-0005#3 DC-0006#4 DC-0009#0 DC-0014#3 DC-0017#1 DC-0017#3 DC-0019#5 DC-0020#1 DC-0022#2 DC-0023#2 DC-0027#4
DC-0029#3 DC-0030#2 DC-0030#3 DC-0031#1 DC-0036#3 DC-0040#2 DC-0041#3 DC-0041#5 DC-0042#3 DC-0045#1 DC-0048#6
DC-0051#1 DT-0004#2 DT-0005#2 DT-0006#2 DT-0006#3 DT-0007#1 DT-0008#3 DT-0011#1 DT-0011#4 DT-0012#0 DT-0014#1
DT-0014#3 DT-0018#2 DT-0020#3 DT-0025#2 DT-0026#2 DT-0028#3 DT-0029#2 DT-0029#3 DT-0030#3 DT-0031#3 DT-0032#3
DT-0033#0 DT-0038#0 DT-0039#1 DT-0040#0 DT-0040#6 DT-0041#2 DT-0043#4 DT-0044#5 DT-0051#3 DT-0051#4 DT-0053#3
DT-0054#3 DT-0054#4 DT-0055#2 DT-0055#3 DT-0056#2 DT-0056#4 DT-0056#5 DT-0058#1 DT-0063#1 DT-0063#3
```

**决定了但错的 7 个**（置信度都 ≥ 0.97，阈值拦不住）：动作错 2——`DC-0048#5` 金标回应、判结束；`DT-0003#0` 金标回应、判澄清（指代不明）；
人选错 5——`DT-0009#1`、`DT-0024#2`、`DT-0040#2`（同名角色认错）、`DT-0059#2`、`DT-0064#4`。

**PyTorch 与 NPU 的差异**：同一请求 614 / 623 决策完全相同。9 处不同：

| id | NPU（部署） | PyTorch |
|---|---|---|
| DC-0007#1 | 结束，0.9501 | 弃权，0.9473 |
| DC-0007#2 | 回应 ailin，0.9501 | 弃权，0.9496 |
| DC-0027#1 | 结束，0.9505 | 弃权，0.9487 |
| DC-0030#2 | 弃权，0.9483 | 回应 haibo，0.9522 |
| DC-0031#0 | 澄清 sun，0.9588 | 弃权，0.9486 |
| DT-0011#1 | 弃权，0.9475 | 回应 xi，0.9533 |
| DT-0012#0 | 弃权，0.9493 | 回应 man，0.9538 |
| DT-0024#0 | 回应 lv，0.9519 | 弃权，0.9470 |
| DT-0051#0 | 回应 yaoyao，0.9767 | 回应 yuhui，0.9768（两人都在金标里） |

8 处是 fp16 与 fp32 的置信度差（≤ 0.011）跨过 0.95；1 处是开放问题里另一位同样合格的人选。离线逐题（两种候选顺序）NPU 与 PyTorch 同选项：
p-dev 99.79%、p-test 99.89%，配对门槛（α 0.05）通过。**阈值两侧的决策会因后端不同而不同；需要逐位一致时，比较须在同一后端上做。**

## 3. 规范版本、澄清原因与有界任务说明的来源

| 对象 | 版本 |
|---|---|
| 标注规范 `laya/train/scenarios/participation/LABELING.md` | 提交 `6fef9cd`（p4 训练数据所依据的最后版本），blob `fb5b66fa9500`，sha256 `045368ac9182b1c7…` |
| 题目 `scenario.yaml`（action / speaker / clarify_about，逐字冻结进 profile 的 `questions`） | 提交 `0bdeddb`，sha256 `7f68491b2bc777e5…` |
| 训练数据修正 `laya/train/data/participation/fixes.jsonl` | sha256 `0faae9cc1341f626…`（p4 `inputs.json` 记录的同一份） |
| 状态格式 `participation-laya-v1` | `laya/src/eidolon_models_laya/participation.py` `LayaParticipationPredictor`：场景目标、候选（M0…按请求顺序）、本轮用户请求、最近 16 条公开记录、最新一条 |

**澄清原因**（`clarify_about` 的三个选项，定义见 `scenario.yaml` 与 `LABELING.md` §4）：

| 原因 | 定义 |
|---|---|
| 指代不明 | 不确定用户说的是哪一位或哪一个 |
| 要求不明 | 不确定用户要做什么 |
| 对象不在场 | 用户点名的角色不在候选里 |

**有界任务说明**（profile 的 `clarify_instructions`，随 `policy_version` 固定）：

| 原因 | 任务说明 |
|---|---|
| 指代不明 | 用户说的对象不明确（不确定是哪一位或哪一个）。用一句话请用户说清楚指的是谁或哪一个，不要替用户猜。 |
| 要求不明 | 不确定用户想让大家做什么。用一句话请用户说清楚具体想要什么。 |
| 对象不在场 | 用户点到的角色不在场。用一句话告诉用户这位不在，并问用户想让在场的哪一位来回应。 |

来源：`laya/train/scenarios/participation/pin_profile.py` 的 `CLARIFY_INSTRUCTIONS`，提交 `1fe1f4d` 引入、之后未改；由 Claude 依据 LABELING §4 的定义撰写，
形式对齐 Agent `role_reply.py` 对 clarify 的处理（Agent 另加"只提出一个澄清问题，等待用户回答"）。**未经 IP 团队主线评审。**
`pin_profile.py --no-clarify-tasks` 生成不带任务说明的 profile，此时所有 clarify 弃权（与 ip-team-v3 一致）。

## 4. 家居上下文输入能力

**已部署的家居模型**（r14，`laya-smart-home-r14-rknn-45f3dedb`，`/v1/systemone`）：

- Agent 适配器（`eidolon_agent/infra/interpretation/adapters/laya.py`）发送的状态只有 `{"utterance"}`；题目为 intent / device / action，
  设备选项 `{名称: 房间·类型}` 加"多个设备或整屋""没有对应的设备"两个出口。**不带任何对话或待选上下文。**
- SDK `InterpretationRequest` 的字段：`utterance`、`origin`（device_ref、area_id）、`candidates`（id、名称、类型、别名、area_id）、`areas`、
  `allowed_intents`、`timeout_ms`——同样没有会话上下文字段。Laya 服务不保存任何会话状态。
- 已验证的是**单句**任务：locked-182 99.35%、v2-dev 96.45%、v3-dev 94.06%，0.9 自动执行覆盖 72.1 / 64.9 / 40.7%（同一 evaluate 代码重跑）。

**续接（pick / follow）**：由另一个训练工作流负责，契约 `continuation-laya-v1`（`laya/train/scenarios/smart-home-continuation/LABELING.md`）：

- 状态 `{"utterance", "context": {"上一句", "Agent", "待执行" | "设备"}}`，由 Agent 从它自己的 `HomeContext`（30 秒、带版本）现场组装；
  Laya 仍无状态，不新增与 Agent 重复的会话状态。`pick` 只在"待选设备"、`follow` 只在"刚成功操作过的设备"时调用。
- 进度（同目录 `PLAN.md` §7）：c1 续接在 c-dev 过门槛（argmax 95.5%，τ_exec 0.97 下接管 69.3%、错误执行 0），但单句回归 v2-dev device 不过 → 不是候选；
  c2 不过；c3（补多设备 / 整屋单句数据）已登记。**没有候选模型、没有 RKNN / NPU 验证、没有作为服务发布、未接入 Agent。**
- 仍需训练：一个同时过续接门槛与单句回归的候选，c-test 只跑一次；之后才做 RKNN 导出、NPU 一致性与延迟验证。

## 待协调：正式发布

发布内容：`eidolon_models`（含上表提交）、`eidolon_kernel e3e4a35`，其余组件钉在板上当前版本
（`rk3588-http-pools-20260929-1`：data `8a873297`、hub `c5269396`、admin `60760629`、agent `9eee07b4`、channel `5551aa9b`、memory `5333cb0a`、sdk `c747e2f6`）。
影响：eidolond 与家居 laya 重启（家居 NPU 分核生效），参与决策服务起在 8773；Agent 的 `participation.url` 仍指 8772 fixture，行为不变。
发布后在部署实例上重跑本报告的回放，确认与本节结果逐条一致。
