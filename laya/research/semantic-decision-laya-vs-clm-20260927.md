# Eidolon 语义决策模型：Laya 与 CLM 研究方案

日期：2026-09-27。性质：源码与官方资料核对、实验设计；**没有在本报告中训练、下载 8B 权重、部署或刷机**。本报告讨论单聊与 Team 的回应、等待、澄清和候选指代，之后扩展家居。架构边界遵循[模式隔离评估](../../../docs/IP团队/Eidolon模式隔离与公共能力架构评估-20260927.md) §14：各场景 Agent 拥有状态、授权与执行，模型仅给出提议。

## 结论摘要

1. **当前可执行主线是 Laya 多语底座＋场景数据微调＋保守出口。** 它已有中文判别接口、本项目训练/校准/ONNX/RKNN 路径和 RK3588 实测。现有家居 r14 仍未达到自动执行安全门槛，不能把家居评测分数外推到单聊或 Team。
2. **`Contrastive-LM/deepswe-clm-heads-8k` 不是完整的通用决策模型。** 官方卡称它为 DeepSWE Bo4 轨迹验证用的单个投影头；必须配套 Qwen3-8B 的特定末 token 池化编码器。卡中自报的 31/38（81.579%）是 38 个留出软件工程任务的候选轨迹选择结果，不是中文对话或设备控制准确率，也不是本项目独立复现。[专用头模型卡](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k)、[CLM 代码](https://github.com/Contrastive-LM/CLM)。
3. **CLM 通用头可列为云端研究对照，DeepSWE 头只作负迁移/初始化消融。** 双编码器有复用候选 embedding 的潜在优势，但 Team 的角色说明、会话版本、指代目标会变；候选变化时须重新编码或严格按版本失效缓存。Opi5max 上尚无 Qwen3-8B pooling＋该头的延迟、内存、并发实测，不能承诺端侧可用。
4. **模型共享与执行隔离是两层决策。** 首选一个共享多任务底座/服务接口，按场景分别维护题目、校准、拒答阈值和评测；只有跨场景负迁移或资源争用的测量结果支持时，再拆权重、adapter 或实例。单聊、Team、家居 Agent 仍各自独立。

## 1. 核对范围与版本

| 对象 | 本次可核对版本/证据 | 限制 |
|---|---|---|
| Laya 运行时 | 本仓 vendored `laya` v0.3.20；[服务说明](../README.md)、`src/eidolon_models_laya/` | 本地实现事实；不是上游最新版承诺 |
| Laya 多语底座 | 本仓 manifest 锁定 `convaiinnovations/laya` 子目录 `multilingual`，revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`；[官方多语模型卡](https://huggingface.co/convaiinnovations/laya-multilingual) | 官方独立模型仓的主分支卡可能已更新；实验固定本仓 revision |
| 家居候选 | r14 本仓 `laya-smart-home/45f3dedb`，模型 SHA-256 `45f3dedbb91d108a90821c8c1c1488baae6b567c4b7769f4e63d460d13178246`；[r1–r15 汇总](../train/R1-R15-SUMMARY.md)、[后续实验](../train/EXPERIMENTS.md) | r17/r18 分组切分复训已完成但开发门槛失败，未取代 r14；r14–r16 旧验证/校准切分有同源交叉 |
| CLM 通用头 | [官方 `CLM-v0.1-8B` 卡](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)、[CLM 仓库](https://github.com/Contrastive-LM/CLM)，查阅于本报告日期 | 本次未固定完整 git SHA；正式跑实验前须记录 commit、权重哈希和 Qwen revision |
| DeepSWE 专用头 | [官方卡](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k)、Hub 文件树显示 revision 前缀 `c60876f`，卡中 checkpoint SHA-256 `554989fe88635606cb978dc45a1ce083be1990c4a51e551ea3b6055ead1a029a` | 前缀不是完整 revision；运行前用 Hub 文件与本地哈希再次核对 |

许可核对：本仓 Laya 多语底座为 Apache-2.0；CLM 通用头与其代码标注 Apache-2.0；DeepSWE 专用头 Hub 标注 MIT；Qwen3-8B 官方模型卡标注 Apache-2.0。正式分发前仍需逐一核对锁定 revision 的许可文件，不能只继承头的许可。[CLM 基础卡](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B)、[DeepSWE 卡](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k)、[Qwen3-8B 卡](https://huggingface.co/Qwen/Qwen3-8B)。

官方模型卡的声明和本项目实测分开陈述。`deepswe-clm-train-embeddings-8k` 的[数据卡](https://huggingface.co/datasets/Contrastive-LM/deepswe-clm-train-embeddings-8k)目前把标题/命令写成 PRM、旧用户名；不能据此断定它是本 DeepSWE CLM 头的完整可复现训练配方。应以专用头卡、CLM 仓库的 `--task clm` 路径及固定数据 revision 交叉核验。

## 2. 任务边界：先定义决策，再比较模型

统一提议结构建议为 `scene, scene_revision, turn_id, action={respond|wait|clarify}, candidate_id?, referent_id?, confidence, reason_code`。候选由拥有该场景的 Agent 给出，稳定 ID 与显示名/角色描述分离。`wait` 不等于模型可无限搁置：应用层给出最大等待时间、下一触发条件；`clarify` 只建议追问目标，问题正文由对应执行器生成/审核。单聊的 `respond` 候选通常是当前 Companion；Team 是本场已授权成员；家居另用意图/设备/动作提议，不把“控制”混进团队发言候选。

每轮输入必须有当前话语、说话人、最近公开轮次、场次目标、候选成员和角色说明、候选版本、上轮已批准的语义状态。点名、别名、代词、轮换、打断、否定、同时点两人和“谁都不用答”要能从同一契约表达。候选变更后旧 `candidate_id` 和旧 `scene_revision` 的结果被 Agent 拒绝；模型文本不得生成 DeviceRef/Owner/权限。响应前仍由 Agent 核对成员绑定、许可、取消状态与回复预算。家居的设备执行需独立的权限与结果核验，歧义/高后果动作保留确认。

现有[模式隔离评估](../../../docs/IP团队/Eidolon模式隔离与公共能力架构评估-20260927.md)记录：Team 当前是固定 mock 裁决，1–16 输出成员，最近 16 条公开消息；角色名单每轮单独注入；语义选择尚未实现。**这些是即将接入的应用约束，不是 Laya 或 CLM 的已验证能力。**

## 3. 基础架构、训练目标与适配比较

| 维度 | Laya 多语＋本仓微调 | CLM 通用头 | DeepSWE 8K 专用头 |
|---|---|---|---|
| 表示与输出 | mmBERT-base 双向编码器 22 层、768 维，约 322M；一题的状态、指令与动态选项同序列，取各 `[MASK]` 标记，经两层决策头输出 logits；`choice/noul/score` | 冻结 Qwen3-8B 的状态/动作编码，分别经约 20M 参数投影头；内积对候选排序，选项 embedding 可缓存 | 同 CLM 架构的一个任务微调头，不含 8B 编码器；卡称 75/38 任务划分，训练成功轨迹任务 66（59 训练、7 验证），Bo4 验证 |
| 已公开训练目标 | 上游模型卡称 RLCD/proper scoring；本仓微调以软标签交叉熵为主，可加 Brier、proper score、策略梯度；`r14` 全部 22 层解冻 | 官方三阶段：约 60M Q&A 对、约 30M 合成难负例、约 1M agent 轨迹；状态—动作双向 InfoNCE，微调脚本冻结 8B、只训双投影头并按同一 step 分组遮蔽假负例 | 从 CLM-v0.1-8B 头初始化，`train/finetune.py --task clm`；以软件工程轨迹状态/动作为配对，验证按候选轨迹末 12 个可用 step 分数均值选 Bo4 |
| 动态候选 | 原生请求内选项；选项间可相互注意，但固定 `head_max_len` 预算随候选数分摊；官方建议约 20 项内 | 候选单独编码，缓存适合复用动作；候选文本变更需重新编码，跨候选交互只在归一化/排序层 | 机制同左，但训练分布是 DeepSWE 轨迹；“可排名任意文本”不证明对话语义迁移 |
| 中文与指代 | mmBERT 预训练支持中文；官方多语卡提供跨语 benchmark，但模型卡同时承认零样本分类弱、未校准；本项目家居中文微调有局部实测 | Qwen3-8B 能处理中文是底座能力；CLM 官方通用卡主要给英文/agentic 基准，缺本任务中文、ASR 与指代评测 | 官方 DeepSWE 结果不能支持中文单聊/Team/家居结论 |
| 校准与拒答 | 上游多语卡称出厂温度全为 1 且过度自信；本仓按题型/选项数留出拟温度，出口选项和阈值由场景策略验证 | softmax 是候选相对分数，API `confidence` 定义为最高概率与其它平均的差；不能视为已校准正确概率。需单独拟温度、出口训练和选择性风险评测 | 同左；Bo4 的 81.6% 也不能变成 `respond` 正确率或拒答可靠性 |
| 部署 | 已有本仓 torch/ONNX/RKNN 导出与对拍 | 官方示例为 vLLM Qwen3-8B pooling 服务＋CPU/GPU 投影头服务；候选缓存改变热路径成本 | 75.6 MB 只是头文件，不代表整栈内存和延迟 |

来源：[Laya 多语卡](https://huggingface.co/convaiinnovations/laya-multilingual)、[本仓训练说明](../train/README.md)、[CLM README](https://github.com/Contrastive-LM/CLM)、[CLM 微调实现](https://github.com/Contrastive-LM/CLM/blob/main/train/finetune.py)、[DeepSWE 专用头卡](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k)。上游 Laya “100+ 语言”指覆盖声明，模型卡列的 MASSIVE 指标覆盖 51 语言；不能把覆盖数当作中文任务质量。

## 4. 本项目已有证据与风险

**家居。** 零样本 Laya 的 182 条早期集控制端到端约 41%，低于规则基线约 60%；说明微调必要。[早期评测](../evals/smart-home/conclusions.md)。r14 用 15,374 行训练，v2-test 总准确率 96.8%、设备 95.2%、控制端到端 91.1%。但 1,141 条合成验收集上自动执行正确率 96.3%（404 例，目标 ≥99%），无关误触发 5.3%（450 中 24，目标 ≤2%）。r15 换种子后 v3-dev 意图退步；r16 提高无关类损失也未通过预定门槛。更关键的是 r14–r16 的训练、开发、验收语句都由 Claude 撰写，旧训练/验证/校准之间有同源交叉，真实 ASR 泛化和安全概率未证实。[轮次记录](../train/EXPERIMENTS.md)、[真实数据方案](../train/REAL-WORLD-EVAL-PLAN.md)。现有家居结果只证明该家居数据上的候选能力，不证明 Team 裁决。

**Opi5max。** 板为 RK3588、16 GB 共享内存，NPU 与 TTS/ASR 并行资源紧张；[硬件档案](../../HOST-RK3588.md)。Laya 64/128 token 的编码器 NPU 曾量到约 59/105 ms；r14 的实际单题长度档位为 126/265/423/638 ms，Mac 到板的家居服务请求命令/查询约 314–346 ms、无关约 176–192 ms，均应以对应输入长度和并发条件解释，不能拿 59 ms 当完整 Team 请求 SLA。[r14 记录](../train/EXPERIMENTS.md)。[Qwen 官方卡](https://huggingface.co/Qwen/Qwen3-8B)列 8.2B 参数，CLM 所需编码器仅按 fp16 参数粗算就约 16.4 GB 十进制字节，**还未计激活、KV/缓存、运行时与其它服务**，因此与现有 16 GB 共存显著不现实；量化后的实际内存/算子支持和速度均未在本板验证，不能由头的 75.6 MB 或官方 H100 延迟推算。若研究 CLM，先在独立 GPU/云上做精度对照，再决定是否值得端侧可行性 PoC。

## 5. 共享多任务模型还是分场景权重

建议先采用 **共享 Laya 初始化权重和推理接口，场景专属数据/题目/阈值**。用三任务混合训练（单聊、Team、家居），每批次按场景平衡，并对已成熟的家居数据做回放；每场景设独立校准组和拒答政策。先比较 `shared` 与三个同底座场景微调权重：两者保持相同输入、数据量、训练计算预算和出口定义。共享路线只有在最差切片及误触发不退步、总体部署成本明显较低时才胜出。

若共享全量微调伤害某一场景，再试“共享冻结编码器＋场景小头/LoRA”，但 **Laya 当前服务与 RKNN 导出没有现成热切换适配层契约**，须把导出、校准和端侧成本算入；不能把“理论少参数”当已可部署。分场景完整权重会增加常驻内存、加载/版本矩阵与回归成本，只有显著质量收益或资源隔离需求时才采用。CLM 路线同理：通用头先试，按场景头作为消融；DeepSWE 头的专业化可能产生负迁移，不能用它替代通用头基线。

## 6. 可复现实验设计

### 6.1 数据与冻结切分

统一事件格式：`event_id, family_id, household_id, session_id, scene, scene_revision, utterance, asr_version, public_context, candidates[{stable_id, display_name, role_description, authorized}], gold_action, acceptable_candidate_ids, referent_ids, ambiguity, policy_label, tags, provenance`。身份和授权字段仅用于构造合法候选、测策略拒绝，不交给模型自由生成。保存脱敏前后映射于受限系统，训练数据只留脱敏文本。

第一轮训练建议各场景 1,000–2,000 个独立事件家族，覆盖 `respond/wait/clarify`，再用 3–5 倍受控增强补候选顺序、别名、指代和 ASR 错字；这是**预算假设，不是已验证足够样本量**。先收至少每场景 300 个真实或人工独立撰写事件作冻结测试，Team 固定成员、动态增删/改角、单聊以及家居各自有开发、校准、最终集。按家庭/人物设定/会话/原句家族分组，变体与同状态副本不得跨集合；另留完全未见过的角色名、设备类型和候选数量。用真实连续语音转写构建自然分布集，合成难例集单独报告。最终集冻结后只跑一次；重复调参时另建新最终集。

人工双人独立标注并裁决：分别标“语义上可接受”和“产品政策允许”。对隐含家居请求、“不用回答”、反问/转述、多人点名、跨轮代词、否定、改口、被打断和候选已失效设置标签。多可接受答案记录集合；判断 `clarify` 是否正确时要求说明缺失的槽位，不把总是澄清当高准确率。

### 6.2 同一输入下的臂与训练

| 臂 | 目的 | 固定条件 |
|---|---|
| R0：规则/mock | 对照实际收益与确定性成本 | Team 现有 mock；单聊固定回应/超时规则；家居现有规则，不拿不适配任务互比 |
| L0：Laya 零样本多语 | 量迁移起点 | 本仓固定 `1c5edc17...`；所有出口和选项描述一致 |
| L1：Laya 场景微调 | 主线 | 软 CE＋Brier、选项重排，先冻结末 8 层，再做全层消融；独立校准组、至少两训练种子 |
| L2：Laya 共享多任务 | 测负迁移/收益 | 与 L1 同总事件家族和训练预算，场景平衡＋回放；每场景独立阈值 |
| C0：CLM-v0.1-8B 通用头 | 合理的 CLM 零样本基线 | 官方 Qwen3-8B pooling 配方、固定 max token、缓存开/关各测；不使用 DeepSWE 专头替代 |
| C1：CLM 场景头 | 测状态—动作迁移 | 冻结相同 Qwen 编码器，预先生成相同训练事件 embedding；按官方 `--task clm/choice` 中与目标匹配的路径改造，记录代码 diff；同样两种子、同划分 |
| C2：DeepSWE 专头 | 只检验领域迁移 | 相同编码器/候选与门槛，作为消融，不以其 81.6% 先验认定优劣 |

CLM 的训练单位必须对应任务：单轮合法行动 `{respond(member), wait, clarify(slot)}` 与状态配对；不能把“成功软件轨迹的最后 12 step 平均分”直接当单轮交互标签。`wait/clarify` 要是真实候选文本并进入训练，不能只用 softmax 阈值凭空制造。Laya 则把三种行动和候选 ID 映射为稳定选项，另训练指代与目标题；大候选集要测试分两级（先行动，再候选）和一次多选项的差异，避免单一 256/512 token 头预算导致列表截断。所有训练与推理记录 token 长度、截断比例、选项数。

### 6.3 指标、决策门槛和统计

按场景分别报：行动混淆矩阵、回应对象 top-1/可接受集合命中、指代解析、`wait` 延误、澄清必要性与有效率、点名/动态候选/长上下文/ASR 切片；候选重排稳定性；NLL、Brier、ECE、可靠性图；覆盖率—错误率曲线。安全指标用 **错误直接回应/错误设备执行的事件率**，并报告假回应、漏回应、澄清负担、端到端成功率。每项给分子/分母与成组 bootstrap 或精确二项 95% 区间，不用单点分数宣称达标。按家庭和会话聚类 bootstrap，减少同一场景模板重复的虚假精度。

预注册最低门槛：Team 的“向无授权/旧版本候选发言”必须为 **0 次应用层放行**（模型可犯错，Agent 必须拦截）；语义模型相对 mock 在人工终局任务成功率有可测增益且最差切片不降；单聊错误等待/错误澄清不得超过产品设定上限。家居沿用现有直接执行正确率 ≥99%、无关直接执行率 95% 上界 ≤2% 的门槛，并以新真实数据最终集核验；未满足时仅建议、确认后执行。阈值和对应覆盖率必须在开发/校准集冻结，最终集不能反复选阈值。[现有真实数据方案](../train/REAL-WORLD-EVAL-PLAN.md)。

运行时记录模型版本、候选与场景 revision、耗时拆分（序列化/编码/头/网络/策略）、CPU/NPU/GPU 峰值内存、p50/p95/p99、并发 1/2/4、取消到停止时间、TTS/ASR 并行影响。端到端延迟从 ASR 最终文本到 Agent 提议通过，另记到开始播放；同硬件、同输入长度与候选数比较。候选缓存要测冷/热及成员或角色变更后的正确失效。

### 6.4 成本预算与停止规则

| 阶段 | 粗预算（仅规划） | 进入下一阶段的条件 |
|---|---|---|
| 数据契约与 3 场景小集 | 3,000–6,000 个独立事件家族，双人标注的冻结集每场景 ≥300；合成增强单独计数 | 标签一致性、分组无泄漏；先不承诺模型指标 |
| Laya 两种子与消融 | 现有 r14 全层训练约每 epoch 45–49 分钟 Mac MPS；三 epoch 约 2.3–2.5 小时/种子，仅作同配置量级估算。新任务数据和混合训练可能改变时间 | 开发集上收益和最差切片达到预注册门槛；先停在离线 |
| CLM C0/C1 云端研究 | Qwen3-8B 编码、训练 embedding 存储与 GPU 小时需先做 100/1,000 事件计时样本，再按总 token 外推；不在没有硬件/供应商报价时写虚假价格。头训练成本小于编码 8B 的成本是架构推断，须测 | 若无显著精度/覆盖优势，停止端侧移植 |
| Opi5max PoC | 只对入选方案测 100/1,000 事件、冷/热缓存、语音并发；复用本仓 RKNN 对拍流程 | 峰值内存留足 ASR/TTS、p95 及取消达标且无质量回退，才另行评审部署 |

最小记录：代码 commit、模型/数据 SHA-256、tokenizer 和 pooling 版本、随机种子、切分组哈希、训练命令、温度与阈值、设备驱动/RKNN/vLLM 版本、每项分母及失败样本 ID（脱敏）。Laya 的 `eval --logits` 可复用同一题目和金标比较 PyTorch、ONNX、RKNN；CLM 需要独立适配器把相同事件转为状态—候选，并输出统一的逐事件 JSONL。报告任何不可比的候选文本或截断差异。

CLM embedding 的**存储下限估算**：官方 DeepSWE 数据卡列 4,096 维 float16，单向量约 8 KiB；若一个事件有一条状态和四个完全不复用的行动，6,000 个事件约 `6,000×5×8 KiB≈234 MiB` 原始向量，还未计索引、元数据、训练副本与验证集。实际 GPU 时间不能由这个存储数或 75.6 MB 头文件推出，按上表先做定长样本计时。Laya 的训练时间也只是 r14 同机器同配方量级，不能直接当新三场景项目工期。

## 7. 分阶段推荐

1. **先锁场景契约和评测集。** Team 只允许已授权固定成员中的 `respond/wait/clarify`；单聊用同结构但独立政策；动态增删/改角先作为离线测试，待版本与取消屏障在 Agent 契约落地后再启用。
2. **以 Laya 多语固定底座跑零样本和微调基线。** 在 Team/单聊新数据上建立至少两个种子、独立校准与出口；保留 mock 和规则基线。家居继续修复分组切分、真实 ASR 验收及执行风险，不把 r14 自动执行。
3. **只在有公平数据后研究 CLM。** 先通用头，再场景头；DeepSWE 头作为专业化负迁移消融。若 CLM 在真实场景上没有明确质量或候选规模优势，以 Laya 为主；若云端明显胜出，先评估端云协同及延迟/隐私约束，再考虑 Opi5max。
4. **最后决定共享或分场景权重。** 用相同最终集比较共享 Laya、场景 Laya、可选 adapter 的质量和资源账。运行模式 Agent、执行权限和审计始终分离；推理服务是否拆实例由实测争用和故障隔离需求决定。

## 8. 硬件性能估计补充（2026-09-27）

以下比较**常驻、预热、单请求、短文本**的决策延迟，不含 ASR、回复 LLM、TTS、网络跨机往返。Mac 指项目中实测的 **M3 Pro 18 核 GPU**，不是所有 Mac；3090/4090 指各 24 GB 的桌面卡。Team 主要是一道 `respond/wait/clarify+成员` 题，家居可能有意图、设备、动作三题；输入长度和头部预算决定延迟。CLM 的一个新状态与 3–5 个已缓存的短候选才与单题 Laya 粗略相近。**不同公开/本地基准的输入不完全相同，表中区间用于容量规划，不是同条件排行榜。**

| 平台 | Laya 多语：单题约 128–256 token | Laya：3 题/较长候选 | CLM 通用/DeepSWE 头＋Qwen3-8B：新状态、候选已缓存 | 依据与部署判断 |
|---|---|---|---|---|
| Mac M3 Pro | **约 20–70 ms，估计**；项目另有 2 题 337 token 38 ms 实测 | **71–93 ms 实测**（早期 3 题、约 478 token）；头部预算扩大时 p95 约 143 ms | **约 0.3–2 s，低可信规划区间**；需 ≥36 GB 内存和经验证的 Apple Silicon pooling 实现，18 GB 的 fp16 方案不合适 | Laya 用本仓 PyTorch/MPS 记录；CLM 没有本机实测，不能套官方 vLLM/CUDA 命令 |
| RK3588 / Opi5max | **126 ms@128、265 ms@256 实测**，RKNN 单题 p50；64 token 图 59 ms 只是更短输入 | **约 314–346 ms 实测**普通家居命令/查询端到端服务；40 设备户型约 720 ms；单题 512 档 638 ms | **现有 16 GB 板不可用 fp16；量化移植延迟未知**，不填假精确数值 | Laya 与 TTS 同时占三核曾造成 TTS 断音，必须测核分配与并发 |
| RTX 3090 | **约 15–45 ms，推算** | **约 25–80 ms，推算** | **约 40–100 ms，推算**（短新状态、热候选）；候选首次编码可能约 0.1–0.4 s，需实测 | 24 GB 显存可研究短上下文 fp16 CLM，但 8K、vLLM 预留、并发可能触顶；从 4090 锚点与硬件规格外推，非已测 |
| RTX 4090 | **约 10–35 ms，推算** | **约 20–65 ms，推算** | **约 28 ms 官方自报实测**（固定候选、新状态、服务端 p50；3 或 50 个候选都约 28 ms）；本任务 128–512 token 粗规划 **约 30–120 ms**，冷候选另加编码时间 | 24 GB 显存；官方短请求结果不能代表 8K DeepSWE 轨迹或网络端到端 |

依据：[本仓 Mac 原始性能表](../evals/smart-home/REPORT.md)、[RK3588 档位及 TTS 并发实测](../train/EXPERIMENTS.md)、[CLM 官方 4090 缓存基准](https://github.com/Contrastive-LM/CLM#the-vector-cache)、[Laya 官方 T4 基准](https://huggingface.co/convaiinnovations/laya-multilingual)、[NVIDIA 3090 24 GB](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)、[NVIDIA 3090 架构白皮书 936 GB/s](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf)、[NVIDIA 4090 架构白皮书 24 GB / 1008 GB/s](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)、[Apple M3 Pro 150 GB/s](https://support.apple.com/en-ie/117737)。

估算方法与误差：Laya GPU 区间锚定本仓 M3 Pro 和上游 T4 的 1 题 32.8 ms / 5 题 40.1 ms，再考虑 3090/4090 的更高算力、token 长度、Python/tokenizer 和 kernel 启动固定开销；**没有按 TFLOPS 简单等比例缩短**。3090 CLM 以同为 24 GB、但架构与算力低于 4090 的事实，从官方 4090 约 28 ms 扩成宽区间；它的 936 与 4090 的 1008 GB/s 峰值带宽接近，仍不足以预测真实 pooling 耗时。Mac CLM 的 fp16 Qwen3-8B 权重约 16.4 GB、M3 Pro 官方带宽 150 GB/s，单纯读一遍权重理论下界约 0.11 s；实际还要计算、激活和框架开销，0.3–2 s 仅是保守容量规划，**不作为性能承诺**。RK3588 上连 0.8B 对照模型曾约 4 s/请求，但不同架构/量化，不能据此给 8B 线性数字。[本仓端侧对照](../evals/smart-home/COMPARISON.md)。

DeepSWE 头与 CLM 通用头的**算子成本**相近，质量和输入长度却不同。8K 状态、动态角色描述首次编码、50 个全新候选、多轮缓存失效，都会显著偏离表中短热路径；尤其不能把官方 4090 的 28 ms 写成 DeepSWE 8K 延迟。下一次真正比较时固定三档：`S`（128 token/4 候选）、`M`（512 token/8 候选）、`L`（2,048 与 8,192 token/动态候选），分别测冷候选与热候选、p50/p95、吞吐、峰值显存及正确率。Laya 同时记录 `head_max_len` 是否截断选项；CLM 记录 embedding cache 命中率。最终按目标设备与语音负载实测后更新上述估计。
