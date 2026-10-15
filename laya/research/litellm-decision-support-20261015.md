# LiteLLM 的路由分类器配置，以及它对“快速决策”的支持

日期：2026-10-15。依据：

- LiteLLM 1.105.1（PyPI 当前正式版）的源码与实测；
- GitHub 主干（main）和 1.106.0-rc.3、1.107.0-dev.1 两个预发布标签；
- 近两周的提交和未合并的 PR。

实测对象是 A100 6 号卡上的 pplx-decider-v1.1-27B，端口 9981。

## 结论

1. **LiteLLM 支持决策模型，远不止“路由分类器”一处。** 1.105.1 已经有一整套 Decisions API：
   - SDK：`litellm.systemone` / `litellm.decisions`；
   - 代理端点：`/v1/systemone` 和 `/v1/decisions`；
   - 8 个 provider，其中有 Perplexity 托管的 pplx，也有自托管用的 `hosted_vllm`；
   - 一个决策模型护栏（guardrail）。
2. **我们的 pplx 最合适的接法：** 在 `model_list` 里把它注册成 `hosted_vllm/pplx-decider-v1.1-27b` 部署。路由分类器用 `deployment_name` 引用这个部署，业务方直接调 `/v1/systemone`、做内容护栏，也都用同一个部署。三种接法都已实测通过，见第 3 节。
3. **LiteLLM 正把决策模型当一等公民来做。** 10 月 3 日到 14 日，相关功能几乎每天都有合并。正在接入的厂商有：
   - OpenAI、Azure（Microsoft-Decision-1）；
   - Perplexity、OpenRouter；
   - Cloudflare（Clef）、AWS（Strands Decider）、Databricks（ai_decide）；
   - TypeSafe（Jev）、Bespoke（Nimble）、Laya 等。

   接口收敛成两种形状：OpenAI 形的 `/v1/decisions` 和 System One 形的 `/v1/systemone`。
4. **路由分类器本身的能力没跟上。** 主干上它仍然只问一道档位题、只用 `choice`。按概率设门槛、低置信二次判断、一次问多道题做能力路由，主干都没有。产品化仍要靠 `custom` 插件或独立路由服务；好在这些都可以建在 LiteLLM 的 Decisions API 之上。
5. **配置名在变。** 1.107.0-dev.1 起，`classifier_type: jev` 改名为 `oss_classifier`，`jev_classifier_config` 改名为 `opensource_classifier_config`，并新增 `provider` 字段。LiteLLM 会自动迁移旧名，但要钉住版本，升级前要验证。

## 1. 路由分类器配置（1.105.1）

### 1.1 推荐的配置骨架

```yaml
model_list:
  # pplx 作为决策模型部署：路由分类器、业务决策、护栏共用
  - model_name: pplx-decider
    litellm_params:
      model: hosted_vllm/pplx-decider-v1.1-27b   # 名字要与 SGLang 的 served-model-name 一致
      api_base: http://<A100>:9981               # 不带路径；hosted_vllm 走 /v1/systemone
      api_key: os.environ/PPLX_API_KEY           # 9981 开密钥后才需要；hosted_vllm 默认不要求密钥
  - model_name: cheap
    litellm_params: {model: ...}
  - model_name: strong
    litellm_params: {model: ...}

  - model_name: smart-router
    litellm_params:
      model: auto_router/complexity_router
      complexity_router_config:
        tiers: {SIMPLE: cheap, MEDIUM: strong, COMPLEX: strong, REASONING: strong}  # 按便宜模型的实力定
        classifier_type: jev                    # 1.107 起改名 oss_classifier
        jev_classifier_config:                  # 1.107 起改名 opensource_classifier_config
          deployment_name: pplx-decider         # 引用上面的部署；和 api_base / api_key 二选一
          timeout_ms: 1500
          circuit_breaker_enabled: true
          circuit_breaker_cooldown_seconds: 30
        classifier_fallback: default_model     # 必改：默认退回的启发式对中文无效
        default_model: strong
        classifier_context_window_size: 3
        classifier_context_include_assistant_turns: true
```

另一种写法是不用 `deployment_name`，直接写 `api_base` 加 `api_key`。显式写了 `api_base` 就必须写 `api_key`，没开密钥时随便填一个占位值。

### 1.2 classifier_type 一览

| 取值 | 怎么判 | 备注 |
|---|---|---|
| `heuristic`（默认） | 英文关键词和正则加权打分 | 中文请求基本全判 SIMPLE；它也是失败时的默认退路 |
| `heuristic_v2` | 按请求类型查 UltraFeedback 成功率表 | 实测接近瞎猜 |
| `llm` | 调 `model_list` 里的聊天模型，用内置量表让它选档 | 要求 `json_schema` 输出，DeepSeek 不支持，会静默退回启发式 |
| `heuristic_first` | 先跑启发式，判到 `heuristic_first_max_tier` 以下就不再调 LLM | 中文请求启发式没信号，几乎全会落到 LLM |
| `hybrid` | 启发式分数落在档位边界附近（`hybrid_boundary_margin`）时才调 LLM | 同上 |
| `capability` | LLM 预测“高效模型能解决的概率”p_solve，配阈值和校准 | 需要 `capability_classifier_config` |
| `llm_v2` | 实验性，联合预测任务需求和模型能力 | — |
| `jev` | 调 System One 决策模型，问一道档位选择题 | **pplx 走这条**；1.107 起叫 `oss_classifier` |
| `custom` | 自写插件：`async def classify(context) -> str | None` | 要门槛、二判、多题时用 |

### 1.3 关键配置项

**档位与映射**

| 配置项 | 作用 |
|---|---|
| `tiers` | 档位到模型的映射，可以是单个模型或模型池 |
| `tier_definitions` | 自定义档名和描述；pplx 把描述当选项标准来读。用了它就必须配 `fallback_tier` |
| `tier_labels` | 只改显示名，不影响判档 |
| `enable_non_reasoning_tier` | 在 SIMPLE 之下加一档 NON_REASONING（转述、格式化类请求） |
| `default_model` | 判不出档位时用的模型 |

**jev 分类器（`jev_classifier_config`）**

| 配置项 | 作用 |
|---|---|
| `deployment_name` | 引用一个决策模型部署，与 `api_base` / `api_key` 互斥 |
| `api_base` / `api_key` | 直连地址和密钥 |
| `model` | 默认 `jev-latest`，接 pplx 要改成服务的模型名 |
| `timeout_ms` | 默认 3000 |
| `instructions` | 替换档位题的指令；实测换成中文反而更差，别改 |
| `circuit_breaker_enabled` / `circuit_breaker_cooldown_seconds` | 一次超时后，在冷却期内（默认 30 秒）所有请求都走退路 |

**失败处理**

| 配置项 | 作用 |
|---|---|
| `classifier_fallback` | `heuristic`（默认，对中文无效）或 `default_model`，建议用后者 |
| `fallback_tier` | 用了自定义档位时，失败落到哪一档 |

**判档上下文**

| 配置项 | 作用 |
|---|---|
| `classifier_context_window_size` | 带上之前几轮用户请求，默认 3 |
| `classifier_context_budget_chars` | 引用历史轮次的总字符上限，默认 8000 |
| `classifier_context_per_turn_chars` | 单轮字符上限，默认不设 |
| `classifier_context_include_assistant_turns` | 是否带助手回复，默认否；开了以后，“要”“好的”这类短回复判得更对 |

注意：**当前请求和调用方系统提示都是全文发送**，这些上限只管历史轮次。

**会话与 Agent 流量**

| 配置项 | 作用 |
|---|---|
| `classification_mode` | `every_request` 或 `user_turn`；后者只在用户发新问题时判档，Agent 工具调用的后续轮次沿用 |
| `session_affinity` | 会话首轮选定模型后整个会话都用它；`session_affinity_ttl_seconds` 默认 3600 |
| `deployment_affinity` | 同一会话、同一档位复用同一个部署 |
| `route_housekeeping_to_cheapest_tier` | 编码 Agent 起标题这类杂活直接走最便宜一档 |
| `plan_mode_min_tier` | 编码 Agent 计划模式的最低档位 |
| `stall_escalation_*` | 工具调用卡住、反复报错时自动升一档 |

**能力约束**

| 配置项 | 作用 |
|---|---|
| `modality_routing` | 带图请求只发给支持看图的模型 |
| `enable_context_window_escalation` | 请求装不进某档模型的上下文窗口时升档 |
| `context_compaction` | 接近上下文窗口时压缩历史 |
| `max_tokens_from_tier_model` | 按落到的模型重设 max_tokens，默认开 |

**规则与人工干预**

| 配置项 | 作用 |
|---|---|
| `keyword_tier_rules` | 命中关键词强制某档；配 `semantic_keyword_matching` 可按语义相似度匹配 |
| `escalation_keywords` | 用户写某个口令（默认 `LITELLM ESCALATE`）就强制升一档 |

**其他**

| 配置项 | 作用 |
|---|---|
| `adaptive` / `adaptive_weights` | 在档位池里按质量和成本做多臂老虎机选择 |
| `cache_aware_routing` | 有提示缓存时，换一个已预热的同档或更高档模型，省缓存钱 |
| `return_raw_model_name` | 响应里返回实际模型名，而不是路由器别名 |

### 1.4 判档请求和返回

LiteLLM 发出的请求是 `POST {api_base}/v1/systemone`：

```json
{
  "state": "…（引用的系统提示、之前的轮次）\nClassify this message:\n<当前请求>",
  "model": "pplx-decider-v1.1-27b",
  "questions": {
    "tier": {
      "type": "choice",
      "instructions": "Pick the cheapest tier whose models can fully answer this request. …",
      "criteria": {"SIMPLE": "…", "MEDIUM": "…", "COMPLEX": "…", "REASONING": "…"}
    }
  }
}
```

pplx 返回：

```json
{
  "model": "…",
  "answers": {"tier": {"type": "choice", "choice": "COMPLEX", "confidence": 0.98, "probabilities": {…}}},
  "usage": {"input_tokens": 205, "output_tokens": 0}
}
```

LiteLLM 怎么用这些字段：

- 只用 `choice` 决定转给哪个模型；`choice` 不在档位名里就算失败，走退路。
- `probabilities` 和 `confidence` 只写进日志，字段名是 `classifier_probabilities` / `classifier_confidence`，不参与路由。
- `confidence` 是 SGLang 自己定义的，不等于最高概率；自己设门槛时用 `probabilities` 的最大值。

## 2. Decisions API（1.105.1 已有）

**入口**

| 入口 | 说明 |
|---|---|
| `litellm.systemone(model, state, questions, api_base=…, api_key=…)` / `asystemone` | System One 形 |
| `litellm.decisions(model, input, questions)` / `adecisions` | OpenAI 形；旧的 `decisions(state=…)` 写法已标记弃用 |
| `Router.asystemone(model="<model_list 里的名字>", …)` | 走部署的连接、重试和日志 |
| 代理端点 `/v1/systemone`、`/v1/decisions` | 另有不带 `/v1` 的别名；有代理的鉴权、日志、计费和限流 |

**题型**：每次最多 128 题，一次调用可以混合多种题型。

| 题型 | 返回 |
|---|---|
| `noul`（是非题） | `noul`：判为“是”的概率 |
| `choice`（选择题） | `choice` 加 `confidence` 加 `probabilities` |
| `score`（打分题，最多 10 级） | `score`（期望等级）加 `confidence`、`legend`、`probabilities` |

**provider**

| provider | 路径 / 默认地址 | 说明 |
|---|---|---|
| `hosted_vllm` | `{api_base}/v1/systemone` | **自托管 vLLM / SGLang，pplx 用这个**；默认不要求密钥，环境变量 `HOSTED_VLLM_API_BASE` / `HOSTED_VLLM_API_KEY` |
| `perplexity` | api.perplexity.ai `/v1/decisions` | Perplexity 托管的 pplx-decider（含 v1.1），在境外 |
| `typesafe` | api.typesafe.ai `/v1/systemone` | 托管 Jev |
| `openai` | `/v1/decisions` | OpenAI 的决策模型 |
| `azure_ai` | — | 主干正在加 Microsoft-Decision-1 的 Entra ID 鉴权 |
| `openrouter` | `/alpha/decisions` | — |
| `cloudflare` | — | Clef；主干已支持图片、音频、视频 |
| `strands_decider` | — | AWS AgentCore 上的 Strands Decider |
| 主干新增 | — | `laya`、`bespoke`、`databricks`（ai_decide）、`fireworks_ai` |

`laya` 和 `bespoke` 这两个 provider 有模型名白名单：laya 只收 english / multilingual / typed-decisions，bespoke 只收 nimble 系列。pplx 不在白名单里，所以 pplx 走 `hosted_vllm`。

**决策模型护栏**（`decision_model` guardrail，1.105.1 已有）：

- 用决策模型给每个请求或回答打分。配置一组 `checks`，每条包含 `name`、`instructions`、`threshold`（默认 0.7）、`action`（block 或 log）。超过阈值就拦截或记录。
- 长文本按 `max_input_chars`（默认 24000）切成有重叠的块，逐块检查。
- 流式回答可以每 N 块抽查一次，也可以只在结束时检查一次。
- 可以直接拿 pplx 做输入、输出的内容审核或舆情预筛。这一项**未实测**，它走 OpenAI 形请求，LiteLLM 会转成 System One 形再发给 hosted_vllm。

## 3. 实测：pplx 接 Decisions API 的三种方式（1.105.1）

| 方式 | 写法 | 结果 |
|---|---|---|
| SDK 直调 | `litellm.asystemone(model="hosted_vllm/pplx-decider-v1.1-27b", api_base="http://…:9981", state=…, questions=…)` | 约 250 ms；一次调用同时问了选择题（档位）、是非题（是否要看图）、打分题（难度 0–4），三种答案都正确返回 |
| 注册成部署 | `model_list` 里加 `pplx-decider`，然后 `router.asystemone(model="pplx-decider", …)` | 正常 |
| 路由分类器引用部署 | `jev_classifier_config.deployment_name: pplx-decider` | `cause=jev_classifier`，判出 COMPLEX，转到强档 |

## 4. 版本走向

| 版本 | 状态 | 与决策相关的内容 |
|---|---|---|
| 1.105.1（10-14） | 正式版 | Decisions API、`/v1/systemone`、8 个 provider、决策护栏、路由分类器 `jev` |
| 1.106.0-rc.3（10-14） | 预发布 | 加 laya provider；仍叫 `jev` |
| 1.107.0-dev.1（10-14） | 开发版 | `jev` 改名 `oss_classifier`，`provider` 可选 jev / laya / bespoke / databricks；新增 `local_heuristic`，可指定 `heuristic_first` / `hybrid` 用哪种本地打分 |

**近两周已合并：**

- 10-03 Bespoke Nimble 网关和 OSS 分类器（#44246）
- 10-09 Databricks ai_decide 当 provider 和路由分类器（#45200）
- 10-10 `litellm.systemone`，并把 `litellm.decisions` 收敛成 OpenAI 形（#45846）
- 10-10 决策模型护栏加管理界面（#45663）
- 10-10 Strands Decider 通过 AgentCore ARN 调用（#45678）
- 10-11 界面上可以直接用已部署的决策模型配置自动路由（#45913）
- 10-12 Clef-omni 支持音频和视频（#45811）
- 10-14 Clef 改用 images 字段传图（#45734）
- 10-14 管理界面可添加 Laya 和 Bespoke 模型（#45260）

**未合并、值得关注的 PR：**

| PR | 内容 | 对我们的意义 |
|---|---|---|
| #45368 | 在任意 OpenAI 兼容的聊天模型上合成 Decisions | DeepSeek 这类聊天模型可以用同一个接口当“低置信二判”的决策模型 |
| #45324 | Strands Decider、Cloudflare Clef 也能当路由分类器的 provider | — |
| #44445 | Clef 用精确的 `api_url` | — |
| #45026 | 原生 OpenAI decisions 请求，按输入计费 | — |
| #46539 | Azure Microsoft-Decision-1 走 `/v1/decisions` 和 `/v1/systemone` | — |
| #46162 / #46523 | 新的 provider：v1m、Nace Drex | — |
| #45476 | OpenRouter、Perplexity v1.1 决策模型的计费 | — |

**判断：**

- 决策模型的接入层（统一接口、provider、计费、护栏、管理界面）在快速完善；这部分 LiteLLM 会替我们做掉。
- 路由分类器的策略层没在动：主干上仍是“一道档位题、取 `choice`”。我们实测最有效的几种做法，LiteLLM 短期内大概率不会内置：
  - 按概率设门槛；
  - 只把有把握判为简单的请求交给便宜档；
  - 低置信时二次判断；
  - 一次多题，做能力路由。
- 所以策略层要自己做：写 `custom` 插件，或者做独立路由服务。

## 5. 对产品化的建议

1. **钉版本。** 先用 1.105.1，或者等 1.106 正式版。升级到 1.107 时验证 `jev → oss_classifier` 的自动迁移。
2. **把 pplx 注册成 `hosted_vllm` 部署。** 路由分类器、业务决策调用、内容护栏共用这一个部署，享受代理统一的鉴权、日志、计费和限流。9981 开密钥后，在部署的 `litellm_params.api_key` 里配上。
3. **策略层写 `custom` 插件。** 插件内部调 `router.asystemone(model="pplx-decider", …)`，复用部署的连接和日志。一次问多道题：档位概率、是否要看图、是否要长上下文、属于哪个领域。按概率门槛选档，低置信时改问二判模型。#45368 合并后，二判可以换成 DeepSeek 的决策形接口，统一走 Decisions API。
4. **可选：决策护栏。** 用同一个 pplx 部署做输入、输出审核，这和舆情研判场景直接相关。先实测再上线。
5. **持续跟进。** LiteLLM 的决策相关功能更新很快，每个版本都要看 `complexity_router`、`decisions`、`guardrail_hooks/decision_model` 的变更。
