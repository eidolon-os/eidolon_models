# IP 角色团训练重新审查与推进：Astra 交接

本文件是历史证据和待审问题清单，不是要求继承的正确方案。用户 2026-09-28 明确认为之前可能误解了场景和数据，要求在一个全新 Astra 会话里重新 review、优化数据生成脚本和设计文档，调整方向后继续训练。此前“必须先有真实数据才能继续”的结论也要重新评估；用户已授权用 GLM 生成合成数据做开发，真实数据缺失不应阻止可做的实验。

## 用户真实目标与约束

- 用 Laya 做快速、准确的小型决策模型，服务 IP 故事角色团的多种自由对话协同与调度。人物是故事中的角色本人，不是编剧、运营、专家顾问等制作团队。谁回答、谁接话、何时旁听或收束需从体验与语义理解出发。
- 体验依据 https://eidolon.aimanthor.com/ensemble/ 。本地源码可直接读 `/Users/manson/ai/eidolon/eidolon-official-site/app/ensemble/{page.tsx,EnsembleStage.tsx,cast.ts,casts.ts}`。六个演示模式：单独聊、点名回应、一起聊、放手讨论、随时打断、安静陪伴。必须亲自阅读理解，不只依靠此处概括。
- 当前输入明确是 text，不引入 ASR。上下文容量、候选数量、角色关系和自由接话应按真实场景估算并实验。不是自动生成角色的完整回复；与角色生成模型、Agent 编排和设备硬约束有边界。
- 用户曾提出 OS 内部 gRPC、项目边界清晰、解耦、调用性能、复用既有机制、不造轮子不打补丁。检查当前真实契约；不能仅凭用户当时疑问断言现状全部走 gRPC。
- 用户明确不让聊天助手直接批量编写训练数据，要求由脚本调用 GLM。最终指定 **glm-5.3-flash**（不是昂贵的 glm-5.3）。API Key 已填入 ignored `.env`，可继续按小规模有界计划调用；不要回显或传播密钥。
- 数据按版本/切片分目录，原始响应、拒稿理由、审定数据、切分、配置、结果均可追溯。先小批量验证学习与泛化，再适量扩充，避免单纯扩数量。用户授权根据问题调整方向、跑几轮训练。
- 发布权重独立于训练权重，不能覆盖正在使用的 release。磁盘紧张，训练串行运行，控制 checkpoint 数量。

## 必须优先重新审查的假设

1. 当前 `move` 单步分类（respond/clarify 加一个成员、wait、finish、abstain）是否足够表达六种体验？此前“一次最多一个成员”与 `wait/finish` 细分是旧方案，需与产品、SDK、Agent 的时序核对，不能当用户已确认需求。自由对话是否允许多个合理下一步？强制唯一金标是否正在制造伪错误？
2. 目前 GLM 根据切片填平面 JSON，本地 `materialize_flat` **直接赋动作标签**。生成内容不一定蕴含这个标签。`named_finish` 的“先别接话”可能只是临时等待，`finish_after_peer` 曾要求字面固定短语“答完本轮就结束”。需要修复监督定义和审稿流程，不能仅增加同模板样本。此前所谓“人工审稿”实际由本会话 AI 逐条审阅，不是独立的人类标注员。
3. 早期 GLM 轮次误生成为“IP 内容制作团队”；后来新增人物性格数据仍叠加旧 v3 的 580 条训练记录，验证集也沿用旧 13 家族。必须审计继承数据与目标场景是否一致，不能只把新增提示词改成人物就认定全量已对齐。
4. 是否过度围绕停止发言和显式命令，遗漏内容驱动接话、人物关系、讨论动力、用户旁听、重复观点、同一成员续讲、同名、多人合理应答、恢复话题、安静陪伴等主要体验？需从官网和架构重新建立场景矩阵。
5. 成员 ID/名字/候选顺序绑定，标签与选项对应，shuffle、冻结层、初始化（为何用家居 r14 而不是基座）、目标分布/损失、NLL 选 checkpoint 与决策质量的关系，都需要有受控实验，避免沿用历史配方。
6. 57 条验证实际上只有 13 家族，置换不独立；已被多轮调参。两份新 GLM 小检查集也已用于诊断。不要宣称其为新的独立验收。现有冻结 comparison/holdout 在 v3/v4 阶段仅做分组/长度审计，未做推理；也要先检查它们是否语义适用。
7. 当前模型会给 clarify 成员，但具体 clarification instruction 如何生成，以及 `DecisionResult` 所需字段是否齐全，要与集成契约核对。不能只过一个动作准确率就宣布可集成上线。
8. 旧文档可能存在手填哈希错误（例如过长的 SHA256），以原始文件重新计算为准。`episodes.generate` 的来源字段可能仍统一标为 authored，GLM 真正来源需看 provenance；修复未来版本的来源标记。

## 代码和证据入口（相对 eidolon_models）

- `laya/train/scenarios/ip-team-v3/EXPERIMENT.md`：旧 v3 设计、门槛、上下文和速度实测，作为待 review 的历史方案。
- 同目录 `IP-V4-GLM-EXPERIMENT.md`：场景偏离的早期 GLM 两轮；`IP-V4-CHARACTERS-EXPERIMENT.md`：后来四轮及其问题。
- `generate_glm.py`、`generation-plan*.yaml`：生成请求、配额、提示词、本地金标适配、结构检查；`GLM-DATA-GENERATION.md`：调用说明。
- `curate_glm.py`、`review*.yaml`：逐条审定/拒稿、来源哈希、不可覆盖写入；`prepare_glm_pilot.py`：向 train 追加，val/calib 字节不变。
- `episodes.py`、`scenario.yaml`、`gen*.yaml`：状态/选项序列化、家族展开、名字和顺序置换；`audit.py`：分组/选项/上下文检查；`eval_dev.py`、`report.py`：开发门槛和分层统计。
- `context_probe.py/json`、`latency_probe.py/json`、`paired_context_eval.py`：上下文和性能诊断。
- `laya/src/eidolon_laya_train/{train.py,model.py,generators.py,evaluate.py,records.py}`：公共训练和评估框架；复用并审查，避免在场景脚本堆补丁。
- `laya/train/data/authored/{ip-team,ip-team-v2,ip-team-v3}/`：历史自编数据，重点检查场景/标签继承是否合理。
- `laya/train/data/generated/ip-team-v4/`：按版本切片的审定 GLM YAML 与 provenance。包括 `ensemble-reviewed-1`（18 家族）、`ensemble-safety-reviewed-2`（19）、`ensemble-exclusive-reviewed-3`（13）；另有两个开发检查目录。
- `laya/train/private/`：ignored 原始请求元信息/响应/草稿。数据不是用户隐私，但不要误传 `.env`。存在旧失败/拒稿记录，不能直接拿所有草稿训练。
- `laya/train/runs/ip-ensemble-v4-characters-r{1,2,3,4}/`：每轮 dataset、lineage、eval、checkpoint 配置与 summary。部分权重已清理，见下文。
- 家居历史：`laya/train/EXPERIMENTS.md`、`R1-R15-SUMMARY.md`；**家居 r14 不是 IP 模型已训练十四轮**，它只是之前 IP 实验初始化来源。

## GLM 与 Mac 环境

Key 文件：`laya/train/scenarios/ip-team-v3/.env`（ignored，权限 600）。变量：EIDOLON_IP_DATA_API_KEY、EIDOLON_IP_DATA_BASE_URL、EIDOLON_IP_DATA_MODEL。正确普通 API base 是 `https://open.bigmodel.cn/api/paas/v4`，model `glm-5.3-flash`；用户提供官方文档 https://docs.bigmodel.cn/cn/guide/develop/http/introduction 。旧 Z.ai 地址与嵌套 JSON 请求曾失败，已改为平面 JSON，最近 14 次调用均正常得到结构有效草稿。脚本每次 `--count` 为 1–8，失败也计配额，无自动补发，pending 必须查清再继续。

Python/CLI：`laya/.venv/bin/python`、`laya/.venv/bin/eidolon-laya-train`。Mac MPS 必须先 `torch.backends.mps.is_available()`；沙箱可能误判 MPS 不可用，已授权的训练可通过 require_escalated 使用真实 MPS，不要静默退 CPU。不要并行训练。现无活跃训练进程，旧 exec ID 不再有效。单份权重约 1.2 GiB。

## 已观察结果（历史，不等于方向正确）

| 轮次 | 家族/训练变体 | 初始化 | 训练时长 | 选中 epoch | train move | 旧 val move | 旧 val 误发言 |
|---|---:|---|---:|---:|---:|---:|---:|
| characters-r1 |98/676|家居 r14|771.3秒|3|591/676|42/57|2/15|
| characters-r2 |117/788|家居 r14|865.8秒|2|607/788|27/57|12/15|
| characters-r3 |117/788|characters-r1|431.0秒|1|729/788|43/57|3/15|
| characters-r4 |130/884|characters-r3|477.6秒|2|835/884|41/57|5/15|

r1/r2：8 epochs，encoder/head LR 5e-5/2e-4；r3/r4：4 epochs，1e-5/5e-5。均 batch4、累积2、解冻最后8层、max_len2048/head256、seed53、val NLL 选权。旧门槛 train≥85%、val≥65%、15静默题误发言≤1；四轮全部 FAIL，未校准/打包/发布。r2 后期虽44/57却 NLL 更高，未保存对应权重。r3 的3条静默错来自同一个家族，r4的5条来自两个家族。停止条件是旧会话自己拟定，用户现已授权重新审查后继续推进。

额外检查：9 家族/33 变体产品小集，r1 28/33、旧 v3 17/33，r1 静默错3/6；7 家族/30 变体安全小集，r2 22/30、r1 10/30，静默错2/21与15/21。两集均已消费作开发诊断，不得假装新测试。没有证据证明真实使用安全；这不意味着不能继续合成数据开发。

上下文历史测量：4成员9条公开输入约1756 tokens，Mac前向中位213.4ms/p95 228.2ms；短512 tokens约46.8ms。只是旧投影和单机合成测量，不是端到端时延承诺。长讨论2875 tokens、64条线协议上限约9479，2048不足；应重新按真实场景估算、实验投影及窗口。

## 集成边界与并行工作

已有独立会话「Laya 决策模型跨项目集成与运行时管理」，ID `01a0e380-a58b-7160-b08a-dbc2e5ef3164`。可只读查状态；本次授权是新训练会话，不自动授权给其他会话发消息或切 Host。
阅读 `/Users/manson/ai/eidolon/docs/IP团队/决策模型集成边界审查-20260928.md` 和当前 `laya/src/eidolon_models_laya/participation.py`、SDK/Agent 契约。集成会话报告已提交默认关闭的 participation v2 适配与专用就绪门槛，家居 r14 不会当角色团权重启用。具体线上状态要另核，训练任务不要激活 Host。工作区有很多既有未提交修改，不能 reset/覆盖。

## 2026-09-28 磁盘清理与保留权重

按用户明确请求，仅删除以下四个 run 的 `checkpoint/model.safetensors`：`ip-ensemble-v4-glm-r1`、`ip-ensemble-v4-glm-r2`、`ip-ensemble-v4-characters-r2`、`ip-ensemble-v4-characters-r4`。数据、配置、tokenizer、summary、评估全部保留，权重 SHA 与删除清单在 `laya/train/WEIGHT-CLEANUP-20260928.json`。释放 5,150,614,880 字节，空闲由约5.1增至9.9 GiB。这些权重需要重训才能恢复，不要误当可加载 checkpoint。
保留 characters-r1/r3、ip-ensemble-v3-b4/v2、早期 pilot、家居 r14/r15/r16/r17/r18及基座；所有 `laya/models/` 发布包未改动。后续只保留必要基线与候选，避免占满磁盘。

## 新会话应实际完成的工作

先独立阅读官网源码、产品/集成文档、SDK/Agent契约及数据样本，写出目标、决策边界、场景矩阵和旧方案根因审查，明确哪些是用户需求、哪些是你的假设。随后实际优化脚本/数据规范/设计文档，修复发现的问题，验证标签与语义、身份绑定、上下文与泄漏；再用 GLM 生成一版适量新数据，执行有假设的小规模训练与同题基线比较。对歧义可考虑多可接受动作、分布标签或合并运行时等价动作，但应先以真实契约为依据，不强行套旧表示。不要只审查后停下，也不要盲目继承旧的唯一金标与门槛追分。若真有无法从代码/产品推断的关键产品语义，再给用户具体选项；其他已授权工作继续推进。报告可复核的改善、仍失败之处和是否满足文本影子集成条件。
