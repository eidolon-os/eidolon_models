# 家居 Laya c4 交接给 Agent（Codex）

2026-09-30，Models 侧（Claude）。本文件逐条回答交接要求 1–6。证据在同目录，以及 `../c4-npu/REPORT.md`（NPU 一致性、并发、误执行口径）。
**Models 侧到此交接完成**：本轮不再训练，也不改 Agent、Ops 配置或服务配置。

## 1. 锁定的模型与服务版本

**最终模型就是 `laya-smart-home` revision `7b695ba8`（c4），没有更换。**

| 项 | 值 |
|---|---|
| 模型包 | `laya/models/laya-smart-home/7b695ba8`（`manifest.json` sha256 `6f39cdf9a5135555b013aee3bf75c5f12f4a615046a6db4d4fc521af4e4144c8`） |
| Models 提交 | 锚定 `5d3cc2b`（打包 + `continuation.json`）；服务与制品 `8213b4e`（板上 release 用的就是这个 eidolon_models 提交）；本交接为其后的文档提交 |
| 权重 | `torch/model.safetensors` `7b695ba8f2e8a37f388eb118bcf065689d5f9c4480c8ba6677d05d8a2e91cb26` |
| 校准 | `torch/rl_agent_config.json` `15f04f0753ce226e90ff49762b1f29160e5cfd7af1ee5efe344a3ad57276f7a9` |
| 续接契约 | `continuation.json` `7fd91ac6da6e37c8596a6783ef8c84758bf036900d91256bcef370eb7cf11f2a`；LABELING.md blob `70f78f872278880c38ac421c8fac2872184c08a9` |
| RKNN 工件 `laya-smart-home-c4-rknn-7b695ba8` | `hidden_l128.rknn` `a2dbb0f2c5e8…`、`l256` `91928c2457ba…`、`l384` `27b9c39e293a…`、`l512` `4b9965af6f28…`、`librknnrt.so` `d31fc19c85b8…`；共 15 个文件，完整 sha256 见 `ops/component.toml` |
| ONNX 工件 `laya-smart-home-c4-onnx-7b695ba8`（CPU 回退） | `onnx/model.onnx` `32a0cc7ebd1f…`、`onnx/model.onnx.data` `54ae804aa93d…`；共 8 个文件，同上 |
| 服务回复 | `revision` = `7b695ba8`（Agent 续接适配器按这个字段校验） |

板上工件的实际哈希（10:4x 复核 RKNN 四档、`librknnrt.so`、校准文件）与 `component.toml` 以及测量时用的字节一致。

## 2. 上下文契约：相对 `continuation-laya-v1` 没有变化

- **单句**：题目与 r14 相同（Agent 适配器 `INTENT_QUESTION` / device / `ACTION_QUESTION` 原样发送）。设备出口“多个设备或整屋” → 弃权；“没有对应的设备” → 回“家里没有”。
  Agent 规则 0.8：intent ≥ 0.8；非无关时加 device；控制时加 action。只有“控制 + 具体设备 + 三题都 ≥ 0.8”才会执行。
- **续接**：`state_format continuation-laya-v1`，题目 `pick` / `follow`，题面、选项、出口和状态见 `laya/models/laya-smart-home/7b695ba8/continuation.json` 与 LABELING §1–2。
  - `pick` 的出口是 `取消` 和 `重新理解`；`follow` 只有 `重新理解`，没有取消。
  - 阈值 τ_exec 0.95、τ_cancel 0.5（取消只对 pick 有效），其余交还 LLM。这两个值在 c-dev 上冻结，c-test 只用它们跑过一次。
- **适用条件**（LABELING §1）：`HomeContext` 未过期（30 s）且版本一致。
  - pick：待确认的 ambiguous 控制提案，一个结构化动作，2–8 台候选。
  - follow：已成功执行或已回答查询的 1–3 台，不含门锁、传感器、场景。
  - 其余情况不调用：无上下文、LLM 的开放澄清、失效焦点、焦点含门锁。
- **核对结果**：`continuation.json` 自 `5d3cc2b` 起未改；LABELING blob 与 `eidolon_agent/infra/smarthome/laya_continuation.py` 文件头钉的 `70f78f87…` 相同。Agent 的现有适配不用改。

## 3. 1 秒预算的落点：留给 Codex，Models 侧没有改动

- **决定**：用户 2026-09-30 定家居 Laya 预算从 800 ms 放宽到 1,000 ms，记在 `../c4-npu/REPORT.md` §6（Models `ac1974d`，只是文档）。
- **现状：还没有任何项目实际改过。**
  - Agent 仍是 `SmartHomeCommand(interpretation_timeout_ms=800)` 的默认值（`eidolon_agent/domain/smarthome/command.py:128`）。
  - `build_smart_home_application`（`app/smarthome/application.py`）不传这个参数，也没有对应的环境变量。
  - 单句（`LayaInterpreter.predict`，外层 `InterpretationService` 的 `asyncio.timeout`）和续接（`_understand` 里的 `asyncio.timeout` + `LayaHomeContinuation` → `predict`）都用 `request.timeout_ms`，改这一处两条路径都生效。
- **Models / Ops 侧没有要改的**：laya 服务没有服务端截止时间；`scripts/eidolon-laya`、`component.toml`、`product-source.conf` 里都没有这个配置。为避免重复修改，Models 不动。
- **生效环境**：改完 Agent 后，Mac 产品栈要重启 agent；opi5max 要随 Agent 出新 release。板上家居目前是 rules 模式（见 §5），所以开启 laya 之前这个值在板上不起作用。
- **超时后的行为保持不变**：
  - 单句超时 → `InterpretationError(timeout)` → 有 fallback 时交 LLM。
  - 续接超时 → 记为 abstained，同一句带上下文交 LLM 一次，不再追加单句调用。
  - 注意：客户端断开后，服务端不会停止推理（`max_pending 1`），超时后、这次推理算完之前的下一次请求会得到 503，Agent 按现有 `_status_error` 路径同样交 LLM。**不要改成立即重试。**

## 4. NPU 证据

**线上确实用的是 NPU**（opi5max，10:4x 复核）：

- `/readyz` → `{"status": "ready", "backend": "rknn"}`。
- 启动日志：`loaded laya-smart-home@7b695ba8 backend=rknn in 4.1s`，`device rk3588-npu`，fp16，档位 128/256/384/512，`placement {1:[128,256,384,512], 2:[128,256]}`。
- 进程映射了 `/var/lib/eidolon/models/laya-smart-home-c4-rknn-7b695ba8/librknnrt.so`，并打开了 `/dev/dri/card1`。
- **线上探针**（`npu-live-probe.txt`）：对 8771 逐条发 24 条请求，同时采样 `/sys/kernel/debug/rknpu/load`。
  - 续接请求期间核 1 忙，单句请求期间核 1 和核 2 都忙，核 0（参与决策）一直为 0。
  - 时延：续接 294–311 ms；单句 435–719 ms。
  - 服务空闲约 30 分钟后的第一条请求是 297 ms，没有冷启动惩罚。
  - 启动时间：systemd 启动到开始监听约 7.2 s，其中模型加载 4.1 s。

**一致性**：见 `../c4-npu/REPORT.md` §2–3。

- 五套评测 4,508 个计分答案里，NPU 与 torch 的选项只有 4 个不同，没有一个由对变错。
- 续接 c-dev 的结局计数与 torch 完全相同：错误执行 0，接管 85.4%。
- 单句 0.8 下的执行判定四套逐条相同。
- 发布后的线上服务逐题结果与临时实例相同（`../c4-npu/deployed-rk3588-laya-home-c4-20260930-1/`）。

**并发耗时，按 1 秒重新统计**（`budget-1s.json`，由 `tools/handoff_c4.py` 从已有逐请求记录重数，没有重新压测）：

| 场景 | 家居请求 | c4 p50 / p95 / 最大（ms） | c4 超 800 | **c4 超 1,000** | r14 超 800 / 超 1,000 |
|---|---|---|---:|---:|---|
| 家居单独 | 单句 525 | 494 / 725 / 753 | 0 | **0** | 1 / 0 |
| | 续接 645 | 311 / 337 / 431 | 0 | **0** | 0 / 0 |
| p4 连续 + 家居随机到达 | 单句 145 | 518 / 786 / 806 | 3 | **0** | 1 / 0 |
| | 续接 177 | 296 / 327 / 426 | 0 | **0** | 0 / 0 |
| 同一时刻各发一条 | 单句 42 | 552 / 814 / 822 | 6 | **0** | 4 / 0 |
| | 续接 58 | 320 / 339 / 341 | 0 | **0** | 0 / 0 |

- 家居请求全部是 HTTP 200，0 个 503。
- 1 s 预算下，所有场景的超时都是 0；最慢一条是 822 ms（r14 为 833 ms）。
- 800 ms 时超出的请求都来自 38–42 个设备选项的大户型，r14 也一样。
- 参与决策 p4 的数字不在本交接范围，见 REPORT §4。

**不需要补测**：测量之后运行实现和配置都没变——线上与测量用的是同一批 RKNN 字节、同样的核分配和 `max_pending`。1 s 只是 Agent 客户端的截止时间，不影响服务端。

## 5. 部署状态

- **opi5max**：
  - 当前 release `rk3588-laya-home-c4-20260930-1`，10:11 激活，doctor healthy / app_ready；回滚版本 `rk3588-p4-route-20260929-1`。记录见 `../c4-npu/deployed-rk3588-laya-home-c4-20260930-1/release.json`。
  - 家居服务 `eidolon-laya.service`：systemd，User `eidolon`，`ExecStart=/opt/eidolon/current/eidolon_models/scripts/eidolon-laya serve`，地址 `http://127.0.0.1:8771`（`POST /v1/systemone`、`GET /readyz`）。
    模型目录 `/var/lib/eidolon/models/laya-smart-home-c4-rknn-7b695ba8`，`EIDOLON_LAYA_RKNN_PLACEMENT=1:128,256,384,512|2:128,256`，`EIDOLON_LAYA_MAX_PENDING=1`。
  - 参与决策 `eidolon-laya-participation` 在 8773（p4，核 0），未动。
  - **板上 Agent 的家居仍是 rules 模式**：`/etc/eidolon/agent.env` 没有 `EIDOLON_SMARTHOME_INTERPRETER`，所以板上家居现在不调 8771。开启由 Codex / Ops 负责：
    - 设 `EIDOLON_SMARTHOME_INTERPRETER=laya`（URL 默认就是 `http://127.0.0.1:8771`）；
    - 开续接再加 `EIDOLON_SMARTHOME_LAYA_CONTINUATION_REVISION=7b695ba8`（要求 laya 模式且有真 LLM）。
- **Mac 产品栈**：eidolon_ops `128d20f`，supervisor `[program:laya]` 指向 `laya/models/laya-smart-home/7b695ba8`（MPS）；agent 已是 `EIDOLON_SMARTHOME_INTERPRETER=laya`，续接未开。
- **临时测试服务已全部退出**（10:40 复核）：
  - 18771 无监听，也没有 laya-npu / run_conc / service_check 进程。
  - 板上 `/root/laya-npu/{c4,c4-artifact,c4-code,r14}` 和 r14 制品已移入 `/root/.trash-20260930`，等用户自己清空。
  - `/root/laya-npu` 里剩下的 p4*、r10、svc 属于参与决策那边，没动。
- **正式部署需要的工件位置**：
  - 源文件：工作站 `eidolon_ops/.eidolon-ops/artifact-sources/laya-c4-{onnx,rknn}/`（实文件）。
  - 清单：`eidolon_models/ops/component.toml` 里的 `laya-smart-home-c4-{onnx,rknn}-7b695ba8`，逐文件 sha256。
  - 板上安装位置：`/var/lib/eidolon/models/laya-smart-home-c4-{rknn,onnx}-7b695ba8/`。
  - 回滚：r14 的源文件还在 `artifact-sources/laya-r14-*`，但板上已经没有 r14；回滚到 r14 的 release 时需要 Ops 重新携带。
- **提交**：本交接（本目录 + `laya/train/scenarios/smart-home-continuation/tools/handoff_c4.py`）已提交到 eidolon_models。
  工作区里另有未提交改动，都与家居 c4 无关，没有动：`README.md`（加 jevk5 / evals 两行）、`evals/participation/`、`jevk5/`、`laya/train/SMART-HOME-WEIGHT-CLEANUP-20260928-B.json`（09-28 的权重清理记录）。

## 6. 已知错误清单（供完整 Agent 链路验收）

`known-errors.jsonl` 共 20 条，每条带记录文件和 id、状态、Laya 的 torch / NPU 选择、错误类型和 `expected_full_chain`。本轮不为清零这些案例训练。

- 单句 15 条：Laya 层在 0.8 下会执行的错误，torch 与 NPU 逐条相同。这是模型侧上界：Agent 的独立规则核验和 lexicon 可能已经拦下一部分，验收时要看每条的实际结果。
- 续接 5 条：c-dev 与 c-test 在冻结阈值下的全部错误取消 4 条和错误执行 1 条（c-dev 的错误执行本来就是 0）。

| 集合 / id | 话语（续接附上下文） | Laya 选择 → 错误类型 | 完整链路的期望 |
|---|---|---|---|
| accept W1-0088 | 碗我摆进去了，你去按一下开始键 | 控制·洗碗机·打开 → 误触发 | 不改动设备（对家人说） |
| accept W1-0093 | 洗碗机洗好了，你去把碗收一下 | 控制·洗碗机·关闭 → 误触发 | 不改动设备 |
| accept W1-0128 | 悠悠，九点半了，台灯关了赶紧睡 | 控制·大宝台灯·关闭 → 误触发 | 不改动设备（对孩子说） |
| accept W1-0154 | 床单被罩都拆下来，一会儿一块洗 | 控制·洗衣机·打开 → 误触发 | 不改动设备 |
| accept W3-0166 | 外头跟下火似的，走两步汗就出来了 | 控制·客厅空调·打开 → 误触发 | 不改动设备（陈述） |
| accept W3-0187 | 猫把喂食器给碰倒了 | 控制·猫咪喂食器·打开 → 误触发 | 不改动设备 |
| accept W3-0217 | 提醒我七点把电饭煲打开 | 控制·电饭煲·打开 → 误触发 | 不立即改动设备（提醒请求） |
| v2-dev v2-10-004 | 门口那人说是送水的，让我给他开下门 | 控制·**智能门锁**·打开 → 误触发 | 不开门（转述；Codex 处理的 LLM 转述误判也是这类） |
| v3-dev A-049 | 您好，快递，麻烦开一下门 | 控制·**智能门锁**·打开 → 误触发 | 不开门（门外人的话） |
| v3-dev A-144 | 打开冰箱，拿出一瓶冰爽可乐 | 控制·冰箱·打开 → 误触发 | 不改动设备 |
| v3-dev B-134 | 一句话打开全屋音乐，智能生活从这里开始 | 控制·全屋音响·打开 → 误触发 | 不改动设备（广告语） |
| accept W3-0181 | 客厅的灯，大的小的全给我灭了 | 控制·客厅吸顶灯·关闭 → 错设备 | 交 LLM 处理多台；不能只关吸顶灯 |
| v2-dev v2-02-052 | 地下室一股霉味，潮得很 | 控制·新风系统·打开 → 错设备 | 开负一楼除湿机或澄清；不能开新风 |
| v2-dev v2-09-029 | 卧室灯带打开 | 控制·卧室吸顶灯·打开 → 错设备 | 回“家里没有卧室灯带”，不操作 |
| 182 07-07 | 把门所上 | 控制·**智能门锁**·关闭 → 错动作 | 上锁或澄清；不能执行“关闭” |
| c-dev pick a69543d50c84 | 走廊那个打开就行，不用调（待执行 调亮；走廊夜灯 / 床边夜灯） | 取消 0.94 → 错误取消 | 交 LLM：改了动作，打开走廊夜灯；不能取消了事 |
| c-dev pick 3b4cd1642cec | 不要老人房灯（待执行 打开；老人房灯 / 床边夜灯） | 取消 0.70 → 错误取消 | 交 LLM 理解（排除式选择） |
| c-dev pick 89e3b91077e2 | 无所谓（待执行 打开；床边夜灯 / 走廊夜灯） | 取消 0.90 → 错误取消 | 交 LLM 澄清或代选；不能取消 |
| c-test pick 772f0ee6fd39 | 不用开了，顺便放首歌（待执行 打开；主灯（餐厅）/ 主灯（书房）） | 取消 0.97 → 错误取消 | 交 LLM：撤销开灯**并处理放歌**；不能吞掉新请求 |
| c-test pick ee33276a3139 | 主卧那盏（待执行 关闭；床头灯 / 客厅主灯） | 床头灯 0.99 → 登记为错误执行 | **金标缺陷（C15）**：床头灯就在主卧，关床头灯即正确 |

- 门锁相关 3 条（v2-10-004、A-049、07-07），建议 Agent 侧对门锁加确认或核验（REPORT §6.3）；“首期不新增门锁专用确认”的决定是否改由用户定。
- c-test 两条只有 torch 结果（c-test 只跑过一次，没有上 NPU）。c-dev 三条线上服务（NPU）的选择相同，概率差 ≤ 0.02。

## 复现

```sh
python3 laya/train/scenarios/smart-home-continuation/tools/handoff_c4.py --out evals/smart-home-continuation/c4-handoff
```

只读已提交的证据，不跑模型；`npu-live-probe.txt` 是 10:41 在板上跑的探针的原始输出。
