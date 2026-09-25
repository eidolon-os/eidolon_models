# Typed decision（Jev / laya / 同类模型）落地全景

快照日期：2026-09-25（Jev 09-15 发布，laya 09-18 发布，一切都在十天之内）。
来源：GitHub 代码搜索（去重 2,067 个仓库、laya 全部 372 个 issue/PR）、HF API（40 个 Space、117 个下游模型、
9 个 org）、厂商与第三方博客、HN / 媒体报道、中文社区。**证据等级：A = 组织 / 在线服务自称生产使用；
B = 可安装的库，或已合入有用户的产品主线（线上是否启用未必能核实）；C = demo / 实验 / 评测。**
下载量、星数都是当天的快照；所有“倍速”“省钱”数字除注明外都是单一来源。

## 0. 一句话结论

**“把 if-else 交给一个一次前向、给概率的决策模型”这件事已经大规模落地——但落地的主体是闭源托管的 Jev，
开源的 laya 还停在移植、shadow 接入和 demo。** Jev 一周内进了 LiteLLM、deer-flow、Chatwoot、inbox-zero
等大项目的主线和十几个网关 / 框架；有名有姓的生产案例有 Unblocked（A）、Vercel（A，媒体转述）；
而没有任何组织声明在生产里用 laya；国内同样没有公司级生产声明，机构动作集中在端侧芯片（爱芯元智 NPU、阿加犀）、
自托管（长亭 Decis）和兼容 API（博查）。同一路线的开源模型远不止 laya：Qwen 小模型路线（decider、Kev、Tev1、
OpenSparX 车载）已经比 laya 更常被下载，另有免训练（AnyJev）和极小专用编码器（CUA-S1、带弃权槽的 GLiClass）两条路。

## 1. 这一类模型是什么，有哪几条路线

共同接口：给一个 state 和若干“类型化问题”（`choice` 单选 / `score` 有序打分 / `noul` 是否），
一次调用返回每个选项的概率，不生成文本，结构一定合法。TypeSafe 称之为 System One（快思考），用来替代
“让大模型输出 JSON 再解析”的判断步骤。

| 路线 | 代表 | 底座 | 特点 |
|---|---|---|---|
| 闭源托管 | [Jev](https://typesafe.ai)（TypeSafe） | 未公开 | $0.042/M 输入 token、输出免费；70–500 ms；最多 255 个选项 |
| 编码器（选项放进输入） | [laya](https://github.com/NandhaKishorM/laya)（Convai） | ModernBERT-large 421M / mmBERT 322M | 开源 Apache-2.0；33 ms（T4）；中文 zero-shot 弱 |
| 小 LLM 打分 | [decider](https://huggingface.co/Mapika/decider-2b)（Qwen3.5-2B，30 天 13 万下载，开源仿品里最多）、[Kev](https://github.com/jaredpalmer/kev)（Qwen3.5 + LoRA + pointer head，6.7k★）、[Tev1-4B](https://huggingface.co/togethercomputer/Tev1-4B-experimental)（Together）、[Bespoke-Nimble-9B](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B)、[APUS-OpenJev](https://huggingface.co/apus-ailab/APUS-OpenJev-v1)、[OpenSparX 座舱](https://huggingface.co/OpenSparX/OAK-Decision-cabin-qwen3.5-0.8b)（0.8B，12 类车控意图） | Qwen3/3.5 0.6B–35B | 精度更高（Kev-27B 0.848 vs Jev 0.857），但更大、更慢 |
| 免训练 | [AnyJev](https://github.com/nokia-applied-research/AnyJev)（Nokia 研究院，484★） | 任意 LLM | 包一层接口，typed-decisions 上 0.80 |
| 极小专用编码器 | [CUA-S1-forms](https://huggingface.co/cua-ai/cua-s1-forms)（706K 参数，填表 99.7%）、[OpenJev Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m)（GLiClass-ModernBERT 151M，**带弃权槽**，1.9 万下载） | 小编码器 | 单任务、极快 |
| 闭源竞品 | [meraGPT Decider 1](https://meragpt.com/models/state-decider-1) | 未公开 | typed-decisions 榜 0.768（但该榜衡量的是与 4B 老师的一致度） |

## 2. 落地分级

### A 级：组织声明生产使用（全部是 Jev，没有 laya）

| 组织 | 用在哪里 | 数字 | 来源 |
|---|---|---|---|
| Unblocked | 代码问答 agent 里挑要注入 prompt 的记忆，替换 cross-encoder | precision 34.8→46.4%，recall 63.8→77.6%，p50 0.20→0.35 s | [官方博客](https://getunblocked.com/blog/jev-in-production-vs-cross-encoder/) |
| Vercel | 命令安全审查分类器，替换原模型 | 快 5–18 倍 | [TechCrunch 转述](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) |
| Dot（getdot.ai） | 原来跑在大模型上的判断负载 | 便宜 10 倍、快 2 倍 | [HN 创始人自述](https://news.ycombinator.com/item?id=49765916) |
| CodeAlive | Mastra 输入审核：blocking（noul）+ category 一次请求，P≥0.7 拦截 | 58 例：恶意 9/9 拦截、正常 0/49 误拦，0.39–0.44 s | [mastra-jev-moderation](https://github.com/CodeAlive-AI/mastra-jev-moderation) |
| 匿名 | 实时语音断句（turn-taking） | — | [HN](https://news.ycombinator.com/item?id=49814753)（弱） |
| PostHog | 生产用**自托管的 JevK5**（Kev 系，沿用 Jev 协议）；TypeSafe 只准用于实验 | — | [posthog egress README](https://github.com/PostHog/posthog/blob/HEAD/posthog/egress/typesafe/README.md) |

厂商口径的规模：Vercel 称上线 24 小时内 AI Gateway 约 13% 的付费团队在用；Almeida 称日用量超过万亿 token（[Latent Space](https://www.latent.space/p/jev)）；$40M 种子轮（媒体）。

### B 级：进了产品 / 框架主线（节选）

- **平台与网关**：Vercel AI Gateway、Cloudflare Workers AI、OpenRouter（`/api/alpha/decisions`，p50 218 ms）、Netlify、
  opencode Zen、LocalAI（Kev 兼容的 `/v1/systemone`）。
- **SDK / 框架 provider**：官方 `@typesafe-ai/sdk`（周下载 29.8 万）、LangChain、Vercel AI SDK、Pydantic AI、Spring AI、
  Airflow、BAML、ruby_llm、laravel/ai、Effect、rig、goose、AgentScope、Composio、TanStack AI、deepeval、mlflow、pipecat、
  AutoGPT、n8n、Node-RED、Haskell `hs-jev`，以及几十个 MCP 插件。
- **可观测 / 评测**：Langfuse、Braintrust、Opik、latitude（shadow 预分类）、neuronpedia（给 SAE 特征解释打分）。
- **数据库里的判断函数**：MotherDuck `prompt_jev()`（GA）、GreptimeDB `ai_match / ai_choose / ai_score`。
- **产品主线代码**（线上是否默认开启无法核实，多数需要自带 key）：见下一节按场景的表。

### laya 自己的生态

- PyPI 周下载约 7 万；HF 下游 117 个仓库里 **约 57% 是格式转换**：GGUF、ONNX、CoreML、MLX（laya-mlx 6.2k★，M3 Max p50 7–13 ms）、
  Android LiteRT（Google `litert-community`）、AXERA NPU、昇腾 NPU（比 CPU 快 34–71 倍）、浏览器 WebGPU，
  以及 Rust / Swift / Elixir / Julia / npm 的库。
- 已核实的 org：TextCortex（Raya，三档模型路由）、FluidInference（CoreML + Swift `LayaManager`，Mac 端侧填表）、
  litert-community、AXERA-TECH、telepatia-ai（葡 / 西语）、InfinimindCreations、dnagpt（华中科大，生物序列）、receptron、ollaya-dev。
  **没有一家在卡片里写生产部署。**
- 分发：1Panel 应用商店的 laya-server、superlinked/sie、vllm.cpp 适配；官方 `laya[serve]`（Jev 协议）、`laya[mcp]`、LangChain 集成。

## 3. 按场景看落地（和 eidolon 相关的加粗）

| 场景 | 项目（模型，证据） | 怎么接的 |
|---|---|---|
| 模型 / 意图路由 | [LiteLLM Auto Router](https://github.com/BerriAI/litellm/blob/HEAD/litellm/router_strategy/complexity_router/jev_classifier.py)（Jev，B，59.6k★）；coder/xum（Jev，B）；[TextCortex Raya](https://huggingface.co/TextCortex/raya)（laya 微调，B）；[openclaw.net](https://github.com/clawdotnet/openclaw.net/blob/HEAD/docs/laya-routing.md)（laya，B） | 一次 choice 选难度档位；LiteLLM 报告档位命中 95% vs Haiku 74%、中位 127 ms vs 688 ms、成本 −96%；openclaw 先 shadow、不确定就回退基线 |
| **语音对话路由** | [**lemma-platform call router**](https://github.com/lemma-work/lemma-platform/blob/HEAD/lemma-frontend/src/call/jev-router.ts)（Jev，B） | **delivery / action / target 三个 choice，confidence < 0.65 降级为 “context”**——和我们的陪伴路由几乎同构 |
| **智能家居 / IoT** | [**fulloch**](https://github.com/liampetti/fulloch/blob/HEAD/core/laya.py)（laya CPU，C）；[**HA-Jev**](https://github.com/AboveColin/HA-Jev)（Home Assistant 集成，B）；[halo-laya](https://huggingface.co/spaces/AlenJoby/halo-laya)（Android TV 氛围灯，C）；[OpenSparX 座舱](https://huggingface.co/OpenSparX/OAK-Decision-cabin-qwen3.5-0.8b)（车控 12 类，原型） | **fulloch：Home Assistant 语音命令只能从给定 schema 里选，置信度阈值 0.90** |
| 工具调用护栏 / agent 门控 | [deer-flow](https://github.com/bytedance/deer-flow/blob/HEAD/backend/packages/harness/deerflow/guardrails/typesafe.py)（Jev，B，82.9k★）；claude-code-templates；GoPlus agentguard；[laya-cli-gate](https://github.com/uzuw/laya-cli-gate)（laya 微调，C）；Vercel eve 自动审批工具调用 | deer-flow：工具执行前一个 noul，≥阈值拒绝，**默认 fail-closed**，启用前要求评测达标（漏放危险调用 0、误拦 ≤5%、p95 ≤1 s） |
| 客服 / 工单 / 邮件分诊 | [Chatwoot](https://github.com/chatwoot/chatwoot/blob/HEAD/app/services/captain/conversation_classifier_service.rb)（Jev，B\*，37.2k★）；[inbox-zero](https://github.com/elie222/inbox-zero/blob/HEAD/apps/web/utils/decision-model/choose-rule.ts)（Jev，B\*）；SmartMom 工厂笔记（laya，C）；学校邮件 / Make.com 分诊 Space | Chatwoot：**每个候选标签一个 noul 放在同一请求里（多标签），≥0.5 取前 3 作 UI 建议**；inbox-zero：choice 概率 < 0.3 或出错就回退 LLM |
| 检索重排 / 记忆 / 上下文 | OpenViking（火山引擎，Jev rerank，B）；hindsight；Unblocked（A）；fast-jev-compaction；LambChat 记忆写入门控（off / shadow / gate） | 每篇文档 / 每条记忆 / 每个 tool result 一个 noul 的“保留概率” |
| agent 动作选择 | [browser-use jev-ultrafast](https://github.com/browser-use/jev-ultrafast)（19.7k★）；cklxx/laya-browser（laya 微调：元素 top-1 0.10→0.66）；droidrun | 每步 operation + target 两个 choice；**返回值不在选项里就不执行** |
| 记账 / 金融 / 新闻 | we-promise/sure（交易分类，含 `__uncategorized__` 选项）；TradingAgents（帖子离题 noul + 立场 choice）；A 股快讯数据集 jevm-news | 置信度不够就不计入 |
| 设备 / 助手后端 | BasedHardware/omi（Jev，B\*） | 任何失败返回 None，保留原来的安全默认 |
| 游戏 | 狼人杀、俄罗斯方块、贪吃蛇、国际象棋、RimWorld、Doom | 每帧一个 choice |

## 4. 反复出现的工程模式（我们照搬）

1. **置信度分流，而不是只信模型**：回退 LLM（inbox-zero <0.3、stuntd、omp-laya-judge ≥0.6 才接受）、转人工（SmartMom）、
   降级到保守动作（lemma <0.65、omi 返回 None）、默认拒绝（deer-flow fail-closed）。
2. **先 shadow 再切换**：latitude、LambChat、openclaw、LumiCore、alpha-pilot；laya 仓库 [#367](https://github.com/NandhaKishorM/laya/issues/367) 也推荐这条路。
3. **启用前设评测门槛**：deer-flow 的三条门槛（漏放 0、误拦 ≤5%、p95 ≤1 s）是现成模板。
4. **钉版本，阈值不跨模型搬**：Airflow / orchestkit 要求钉 `jev-1.13.0`；laya 的 confidence 定义（1 − 归一化熵）和 Jev 不同，
   同一批答案在 0.9 阈值下 Jev 公式放行 58%、熵公式只放行 30%（[#302](https://github.com/NandhaKishorM/laya/issues/302)）。
5. **一个 state、多道题、一次请求**；多标签用“每个候选一个 noul”（Chatwoot）。
6. **选项多先粗筛**（shortlist top-k 再 choice）；**返回值严格校验**（不在选项里当错误）。
7. **给出口**：`__uncategorized__`（sure）、abstain 槽（OpenJev Verdict）、“context” 降级（lemma）——和我们智能家居评测里“多个 / 没有”两个出口 0% 的问题是同一件事。

## 5. 负面结果与质疑（别被发布热度带偏）

- **校准**：公平硬币 Jev 说正面 0.92；公平骰子平均给 83% 概率、实际只中 19%（[alexmolas](https://www.alexmolas.com/2026/09/23/jev-cant-be-calibrated.html)）；nibzard 测得 Jev ECE 0.246，最差。
- **安全**：越狱检测在整理过的数据集 AUC 0.937，真实攻击在 5% 误报下只拦 19.9%（ProtectAI v2 58.8%）；prompt injection 能把拦截概率从 0.76 压到 0.48（VentureBeat）。
- **能力边界**（TypeSafe 自己承认）：计数、算术、多步推理弱，无关上下文会干扰（[model jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)）。
- **放弃案例**：hermes-agent 评测后“Do not adopt Jev”（上下文压缩）；no-mistakes 撤掉 Jev 审阅简报。
- **laya 中文**：飞书职场 20/64（Jev 64/64）；中文语音指令 10/20（Jev 20/20，15 行 regex 19/20）；JevBench laya 第 36、Jev 第 1；
  封闭测试集 30.8% vs 公开集 58.4%（[#252](https://github.com/NandhaKishorM/laya/issues/252)）；官方 Docker 镜像曾经每个请求 500 而 `/health` 正常（[#365](https://github.com/NandhaKishorM/laya/issues/365)）。
- **HN 的主要批评**：“just BERT with more data”；laya 的对比拿微调过的自己比 zero-shot 的 Jev。

## 6. 中文生态

**没有一家国内公司公开说在生产业务里用 laya 或 Jev。** 机构层面的动作集中在端侧芯片、平台上架和兼容 API；
应用层是聊天副驾、机器人插件和各种路由 / 分诊项目；中文实测普遍说明 laya 零样本不可用、微调后才可用。
（本节的 A 指机构正式发布的产品或在线服务，不是“某业务在生产里用”的声明。）

| 场景 | 项目 | 做什么 · 模型 | 效果 / 规模 | 证据 |
|---|---|---|---|---|
| 端侧芯片 | 爱芯元智 [AXERA-TECH/Laya](https://huggingface.co/AXERA-TECH/Laya)、[laya.axera](https://github.com/AXERA-TECH/laya.axera) | 三套检查点编成 AX8850 NPU 模型，**用中文客服数据做量化校准**，中文推荐 multilingual | 约 28–31 ms/问 | A |
| 端侧平台 | 阿加犀 Model Farm（[模型广场](https://aiot.aidlux.com/zh/models)） | 09-24 上架 laya，定位工单、邮件分流 | — | A |
| 自托管 API | 长亭 [chaitin/Decis](https://github.com/chaitin/Decis) | Jev 兼容 `/v1/systemone`，默认 laya-multilingual | Docker 拉取 2306 | A/B |
| 国内 Jev 兼容 API | 博查 [jev.bocha.cn](https://jev.bocha.cn)（bocha-jev-v1） | 协议与 TypeSafe 一致，限时免费；是否自研未核实 | 已被 jev-chat 设为默认 | A |
| 聊天副驾 | [jev-chat-jarvis](https://github.com/jev-chat/jev-chat-jarvis)（5.9k★，微信 / QQ / 飞书）、jev-chat-windows | Jev 判断意图和风险，再给回复候选排序 | Windows 版下载 1.4 万 | B |
| 机器人 | [feishu-jev-bot](https://github.com/Harozou/feishu-jev-bot)、QQ 群机器人、AstrBot 插件、微信插件 JevIntent | 意图 / 情绪 / 风险 | — | B |
| 客服 | [laya-decision-api](https://github.com/bmw8080/laya-decision-api)（PyPI + Gitee）、[jeves-desk](https://github.com/dabaicai001/jeves-desk)（Jev 决策 + Qwen 生成） | 意图、工单分派、路由、审核 | M4 MLX 热调用 8–22 ms | B |
| Agent 路由 / 旁路 | [chainlesschain](https://github.com/chainlesschain/chainlesschain)（`--decision-provider laya`）、jev-dsh-decision、[LumiCore](https://github.com/5pyj29s7cm-tech/LumiCore/blob/main/docs/local-laya-shadow.md)（laya 只做 shadow） | 选 skill / 选模型 | — | B |
| 安全告警 | [sentriq](https://github.com/chenchunrun/sentriq) | laya-mlx 做 8 个快判断，再交 LLM 复核 | — | B |
| 金融 | 魔搭 [laya-astock-sentiment](https://www.modelscope.cn/models/xuff78/laya-astock-sentiment) | A 股舆情微调，1799 道题 | 验证集 top1 0.748 | C |
| 车载 | [inforExtra PR#1](https://github.com/ranpin/inforExtra/pull/1) | 座舱对话抽人员和位置，laya vs Qwen3.8-Flash（510 条） | 全对 72.2% vs 96.3%；p50 44 ms vs 1.29 s | C |
| 语音客服 | [bok-voice](https://github.com/halojerry/bok-voice/blob/main/docs/A_LINE_LOGIC.md) | 粤语快递 / 防诈客服调研 | 零样本提不了准确度 | C |
| 电网运维 | [grid-qa](https://github.com/zhyese/grid-qa/blob/main/docs/laya-decision-model-%E8%B0%83%E7%A0%94.md) | PoC 计划，验收线 ≥+10 个百分点、ECE<0.15 | 未实施 | C |
| 云厂商 demo | [AWS 中国博客](https://aws.amazon.com/cn/blogs/china/laya-service-validation-amazon-sagemaker/) | SageMaker CPU 部署 | 模型延迟 437 ms | C |

**中文微调（新发现，均在魔搭 / GitHub）**：
- [zcgnull/laya-zh-v2](https://www.modelscope.cn/models/zcgnull/laya-zh-v2)（附 [Ask Laya](https://zcgnull-ask-laya.ms.show)）：飞书分诊 choice 31.2%→51.6%，**客服测试集 42.8%→90.9%**。
- [chinese-laya](https://github.com/yanqiangmiffy/chinese-laya)：机翻中文数据微调，34.05%→76.25%，但 ECE 0.076→约 0.15（校准变差）。

**中文测评与踩坑**：
- [yibie/laya-jev-lab](https://github.com/yibie/laya-jev-lab)：40 条中文客服工单，Jev 78%（588 ms）、laya 57%（7.6 ms）；
  **laya 置信度 ≥0.6 就本地回答、否则转 Jev：准确率与 Jev 持平、快 1.8 倍、45% 的流量在本地解决**。
- [awesome-jev-zh](https://github.com/yzfly/awesome-jev-zh)：复现只对一半（AG News laya 赢、emotion Jev 赢 8–9 点）；**提示写法不能跨模型迁移**；
  **int8 动态量化会让结果随 batch 变化**（与我们实测 int8 保真度差一致）。
- [长亭可行性文档](https://github.com/chaitin/Decis/blob/main/docs/feasibility.md)：出厂 ECE 0.466；CPU 上高 QPS 不成立，跨请求批处理实测负收益。
- 其他：zhiyan 历史回归 22 条只中 9 条；[#233](https://github.com/NandhaKishorM/laya/issues/233) 中文贪吃蛇写明向右会撞墙仍选向右；公众号实测《置信度 1.000，却答错了》《中文守卫先别开》。

**国内平台**：魔搭官方镜像（下载 6309）和 GPU studio；MNN 版（鸿蒙 CPU 验证过）、GGUF 版；Gitee / GitCode 镜像；
B 站 1Panel 一键部署教程；中文文档 [jev-docs-zh](https://github.com/Bald0Wang/jev-docs-zh)。
**灰色渠道**（注册机、号池反代）存在，不要碰。钉钉、企业微信、数字人、智能家居、车载量产、电商、金融风控方向都没找到公司级落地声明。

## 7. 对 eidolon 的启示与方案更新

1. **陪伴路由照抄 lemma 的形状**：一次请求问“交付方式 / 动作 / 目标”几道 choice，低置信降级为“只入上下文”或澄清——
   这和团队方案文档里的 request_reply / present / context_only 正好对应。
2. **智能家居照抄 fulloch / HA-Jev 的约束**：只在给定设备 schema 里选；但我们的评测已经证明必须补“多个 / 没有”出口和校准，
   阈值 0.90 在 zero-shot laya 上不安全（高置信错选）。
3. **落地流程照抄社区共识**：shadow → 评测门槛（deer-flow 模板）→ 灰度；钉模型版本；每换模型重新定阈值；敏感动作 fail-closed。
4. **扩大底座候选（新增）**：除了 laya-multilingual 与 MacJev，纳入同一套评测：
   - 中文 laya 微调：**zcgnull/laya-zh-v2**（魔搭，客服 42.8%→90.9%）、chinese-laya；
   - Qwen 小模型路线：decider-0.8B/2B、OpenSparX 0.8B（中文车控，最接近智能家居）；
   - 参照组：AnyJev（免训练，做上限参照）、带弃权槽的 OpenJev Verdict 151M。

   小 LLM 只输出一个选项 token、以 prefill 为主，RK3588 上可以借 RKLLM 前缀缓存；编码器路线则更快、更可控。用同一份数据微调、同一套评测选优。
5. **“本地快判 + 置信度不够再升级”是已被验证的形态**：yibie 的实测（laya ≥0.6 本地、否则转 Jev）准确率持平、45% 流量本地解决；
   sentriq、stuntd、inbox-zero 同理。eidolon 的对应物是：本地 laya 先判，低置信升级给主 LLM（或云端决策 API），
   升级结果回流成训练数据——这就是团队方案文档里 System-1 / System-2 的双层结构。
6. **Jev 本身**：作为线上参照与冷启动老师（给训练数据打软标签）是合理的，但它是外部服务（数据出域），而且校准有已知问题——
   不作为 eidolon 的线上依赖。国内有博查、长亭 Decis 这类兼容 API / 自托管方案，可做对照。

## 8. 没能核实的

GitHub 依赖图显示 0、libraries.io API 已停用，没有权威的“谁依赖 laya”清单；code search 每个查询最多 1000 条
（`"jev-latest"` 约 2.87 万次命中），只看到一部分；inbox-zero / Chatwoot / omi / sure 等在线上是否默认开启无法核实；
X 上 Metaview、Vercel CEO 的说法只看到摘要；TypeSafe 的厂商倍速没有独立复现；Product Hunt 条目没找到。
