# Mac 两角色实机测试：2026-10-29

> 历史记录。2026-10-30 已将协议投影、结果验证和可选审计迁入正式 `laya/src/eidolon_models_laya/remote_participation.py` 与 `participation_audit.py`，测试迁入 `laya/tests/test_remote_participation.py`。临时 gateway HTTP 入口已退役；历史结论和审计分析器保留。部署请使用 Models 原有 participation 服务与 Ops 生命周期，不再按下文历史步骤启动网关。

目的：验证真实 PTT 输入 → ASR → d3 决策 → 独立回复大模型 → 指定设备播报。
决策模型只决定回应、澄清、等待、结束以及谁回应，不生成角色回复。

## 测试边界

- 保留已有 Owner、设备绑定、手机数据，不擦除 NVS。
- 固件使用各板官方 `flash --app-only`，按板顺序构建和刷写。
- Host 使用 Ops 管理的现有 pinned source runtime；不切换到未验收源码。
- d3 使用原始策略。先前追加提示词没有证实泛化收益，不作为默认方案。
- gateway.py 是临时本机 HIL 适配器，不是生产服务。只支持完整公开消息上下文、1–6 个候选；含 summary/pending_requirements 等不支持输入时明确拒绝，不静默丢失。
- 用户已明确授权本次实机转写文本、角色描述和会话上下文发送至 HTTP d3；不发送原始音频或密钥。先前自动审批拒绝后曾撤回配置，获得明确授权后才启动服务并重新配置。

## 硬件与接线

| 用途 | 板卡 | USB 端口 | 序列号 |
|---|---|---|---|
| 唯一 PTT 输入（待验证在线） | Waveshare ESP32-S3 Touch AMOLED 2.06 | /dev/cu.usbmodem1401 | 1C:DB:D4:7A:EF:0C |
| 成员输出 A | ESP-BOX-3 | /dev/cu.usbmodem1201 | 24:EC:4A:52:F3:54 |
| 成员输出 B | M5Stack StackChan | /dev/cu.usbmodem1301 | 68:EE:8F:D8:4A:04 |
| 配置/控制 | Android PHK110 | ADB b3b61f91 | — |

USB 提供刷写/日志/供电；会话音频依赖设备和 Mac 的网络连接。两个输出成员需绑定不同 Companion；输入设备独立于输出。

## 当前证据

- Mobile：官方脚本安装成功（保留数据），已安全连接 Mac 192.168.3.40。
- Mac：Ops status 15/15 正常，正式 app-ready 检查通过（仅 Host 侧 LAN 契约）；角色团 participation.url 原本为空。首次 restart 被配置校验拒绝，未完成重启；现有服务继续运行。
- Ops：参与决策必须通过 `[model_endpoints].laya_participation` 配置，loopback 端口固定 8773，不能 settings.overlay 覆写。
- Waveshare：应用刷写、分区回读、ota_0 启动选择、运行 buildstamp 校验全部通过；owner_trust=ready。
- BOX-3：应用刷写、分区回读、ota_0 启动选择和运行 buildstamp 校验通过；owner_trust=ready。
- StackChan：应用刷写、分区回读、ota_0 启动选择和运行 buildstamp 校验通过；owner_trust=ready。
- 手机驾驶舱当前显示 0/6 身体在线，活动/轮次/后台任务部分读取未知。服务探活不代表硬件就绪。
- 手机已进入 IP 角色团队准备页；未启动团队、未录音。
- Waveshare 官方串口监控确认 WiFi 扫描反复 `No AP found`；当前尚未连上保存的无线网络。USB 连接不等于网络就绪，需要正常配网或恢复原 WiFi。
- 适配器单元测试 2 项通过；loopback 8773 → 外部 d3 合成点名请求返回 HTTP 200、正确 respond/real-id，单次端到端约 351 ms（不是延迟分布）。尚无实机端到端成功证据。
- 已通过 Ops model_endpoints 配置；生成 agent.yaml 已确认 participation.url=127.0.0.1:8773/v1/participation/decide，Host 重启进行中。
- 用户选择通过 Mobile 重新配网；助手不再操作手机界面。

## 后续验收顺序

1. 完成剩余固件的读回/运行版本校验。
2. 查明设备联网与 Mac Owner 绑定状态，通过 Mobile 正常流程连接；不直接写数据库或擅自解除绑定。
3. 获得实机 d3 外发授权后启动 loopback 8773，做合成契约请求验证，再通过 Ops 配置和重启。
4. 确认输入 PTT、两位不同 Companion 和两个输出设备可用；在 Mobile 开始团队。
5. 用户按住 PTT：点名 A、点名 B、要求 A 后 B、同角色追问、歧义澄清、暂停/结束、播报中打断。
6. 分别记录转写、决策动作/角色、运行时执行、实际播报和各阶段时延；不得把 ASR 或播报故障记作模型分类错误。
7. 用未提前调提示词的自然表达复测；短场测只给接入结论，不宣称泛化准确率。

## 本地日志

- /tmp/eidolon-team-wave-20261029.log
- /tmp/eidolon-team-box-20261029.log
- /tmp/eidolon-team-mobile-20261029.log
- /tmp/eidolon-team-host-restart-20261029.log
- /tmp/eidolon-team-d3-gateway-20261029.log

原 Ops profile 备份：/tmp/eidolon-mac-profile-before-team-20261029.toml。

## 完整审计已启用（2026-10-29，上一轮实测之后）

上一轮缺失的模型请求/回复正文无法追补。以下仅覆盖启用后的新会话。

- Agent 运行源码：独立 `codex/participation-audit-hil`，提交 `ac205fcfb48128024883b4d1254d04907d04b692`。Ops 仅更新 Agent source override；旧 pinned Agent 源码保留。
- 原主 Agent 工作区保留同样的未提交改动，便于后续正式评审合入。
- Agent 与 gateway 均设置 `EIDOLON_PARTICIPATION_AUDIT_DIR=/private/tmp/eidolon-role-audit-20261029`；无此变量时关闭正文审计。
- 本地目录权限 0700，JSONL 文件 0600，按来源/UTC 日期/PID 分文件。仅记录公开角色团内容与执行元数据；不记录原始音频、HTTP 认证头、环境密钥或私人记忆。
- 写入失败不会改变决策/播放流程，会输出失败计数告警；gateway `/health` 显示审计开关和写入失败数。
- 模型、提示词、概率选择及轮次预算逻辑未改。记录同步写入本地文件，有少量 I/O 开销。

### 可核查的事实

|层|事件与内容|
|---|---|
|场景|session_id、角色名字/说明、Companion 与设备映射、目标和回复预算|
|Agent 请求|decision_id、epoch、用户原请求、触发消息、完整公开上下文、候选与约束|
|Gateway|实际投影后的 d3 payload、M0 等槽位映射、HTTP 状态、完整解析输出与概率、映射后结果、总耗时、失败/取消|
|Agent 采纳|返回结果、契约验证通过、未采纳及取消原因类型、当时 epoch|
|回复|相同 decision_id → turn_id → Companion/设备；累计生成正文、生成是否完成、播放是否确认、取消/错误终态|
|控制|PTT press/release、设备停止回执、speaking、reply_played、waiting、closed|

`generated_text` 不是用户实际听到的逐字稿。打断时它可能包含尚未播出的文本；`text_is_audible_transcript=false` 明确标记。只有匹配的完成回执可标 `playback_completed=true`，缺少终态事件一律未知。审计不改变仅把完整播放回复加入讨论历史的既有逻辑。

### 分析入口

```sh
python3 evals/participation/mac-hil-20261029/analyze_audit.py \
  /private/tmp/eidolon-role-audit-20261029 \
  --session '<真实团队 session_id>' \
  --output /private/tmp/eidolon-role-audit-20261029/report
```

输出 `joined.json`（完整请求、结果、回复与事件）和 `INDEX.md`（逐次索引）。以 `(session_id, decision_id)` 精确关联；不会用返回顺序猜测。残缺 JSON 行会列为解析缺口，缺少 Agent 记录不等于“未执行”。不自动计算语义准确率。

### 已验证与限制

- Agent 协调/回复/审计测试 31 项通过；gateway 与分析器测试 4 项通过。
- 覆盖取消时部分正文留存、生成与播放区分、日志写入失败不影响流程、按 ID 关联和残缺行处理。
- 真 d3 合成请求 `audit-smoke-20261029` 返回成功；5 条 gateway 审计事件完整，失败计数 0，汇总成功。该请求属于 `audit-smoke-not-user-session`，不能计入真实测试。
- Ops 重启成功，重启后 `app-ready` 通过；实际 Agent 进程 cwd 已确认指向独立审计源码。
- 尚需新的一轮实机对话确认 Agent 与 gateway 的完整联合记录；未把合成验证称为实机通过。
- Host 重启会结束旧团队，需要在 Mobile 重新开始团队。
- gateway 仍是当前会话启动的临时进程，并非系统开机自启服务。
- 测试完成后先归档所需审计文件，再删除不需要的正文记录；/private/tmp 不作长期存储。
- 回退参考 `/tmp/eidolon-mac-before-audit-20261029.toml` 的 Agent source override，移除 Agent env 中本次新增的审计变量并通过 Ops 重启；不要覆盖其他后来改动的 Host 配置。
