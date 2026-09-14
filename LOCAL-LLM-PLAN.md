# 本地 LLM 为什么慢，以及修它的计划

**日期**：2026-09-10
**证据来源**：RK3588 / opi5max，release `rk3588-20260909T171415`（全本地配置）
**状态**：调研完成，未改任何代码

一句话：**本地 LLM 慢，主因不是算力也不是绑核，是提示词里 77% 的固定成本每轮重付；
而 Agent 侧一整排默认值都是照云端模型的成本模型定的，本地把 prefill 从"免费"变成
"每 85 token 一秒"之后，这些假设同时失效。**

---

## 1. 触发这次调研的那一轮

2026-09-10 01:43 的一次全双工真机对话，用户问"你叫什么名字？"：

| 阶段 | 耗时 |
| --- | ---: |
| 收音（说完 → 检测到结束） | 1.36 s |
| ASR 完整转写 | 0.85 s |
| 结束判定 + 上下文编译 | 0.36 s |
| **LLM 首字** | **28.1 s** |
| 首字 → 进入 speaking | 3.4 s |
| 播放 | 9.4 s |

板上日志（`turn_timings turn=b576cec1… llm_ttft=28122 output=8849 compile=149`）与
llama-server 的计时一致：

* 提示词 **1452 token**，prefill **28.05 s**（51.8 tok/s）
* 生成 18 token，8.83 s，**1.92 tok/s**
* 上下文编译只花 **149 ms** —— 不是编译慢，是编出来的东西太大

那一轮**没有**超时、没有超上下文、没有凭据错误。30 秒首输出窗口生效，剩 1.7 秒余量。

---

## 2. 一个必须先说的方法错误

此前 HOST-RK3588.md §2.29 报的 "decode 5.2 tok/s、生产提示词 278 token、冷首字 3.3 s"
**是错的**，错在基准的尺寸：

* "278 token" 是从日志里一行 `n_past = 271` 推的，那多半是某轮的**增量**而不是全量。
  真实生产提示词是 **1452**。
* 所有 decode 数字都是在 **120–160 token 上下文**下测的。而 decode 吞吐**随上下文
  长度显著下降**——这个规律同一天在 memory 抽取那一测里已经出现过（4929 token 上下文
  下 decode 掉到 1.06 tok/s），当时写进了记录，然后没有被接到 LLM 基准上。

§2.29 关于**核分配**的结论仍然成立（见 §6），错的只是它报的绝对性能。

### 2.1 补上的曲线（实测，冷前缀，每档换 salt）

| 上下文 | prefill | decode |
| ---: | ---: | ---: |
| 133 tok | 84.3 tok/s | **5.42** |
| 321 | 89.9 | 4.67 |
| 620 | 86.2 | 4.00 |
| 918 | 81.8 | **3.49** |
| 1216 | 79.2 | **2.93** |
| 1452（生产实测） | 51.8 | **1.92** |

**prefill 基本恒定在 80–90 tok/s；decode 随上下文单调下降。**

`ops/component.toml:88` 声明的门槛是 **3.43 tok/s** —— "让模型跟得上正在说出口的话"。
按曲线插值，**decode 在约 950 token 上下文处穿过这个门槛**。

> **本地 LLM 在生产上下文长度下达不到它自己声明的门槛。**
> 这个结论不依赖首字：哪怕前缀 100% 命中、prefill 归零，1.92 tok/s 仍然供不上 TTS 说。
> 绑核、预热、批大小全都只作用于 prefill，一个都救不了。

---

## 3. 1452 token 是谁占的

用 llama-server 的 `/tokenize`（模型自己的分词器）逐段称重，板上真实 genome：

| 成分 | 真 token | 占比 | `chars//3` 估的 |
| --- | ---: | ---: | ---: |
| **工具 schema × 3** | **715** | **49%** | 1025（高估 45%） |
| &nbsp;&nbsp;`delegate_to_coworker` | 463 | | 673 |
| &nbsp;&nbsp;`get_weather` | 143 | | 198 |
| &nbsp;&nbsp;`get_time` | 109 | | 154 |
| **harness 策略** | **398** | **27%** | 255（低估 35%） |
| persona（stable） | 122 | 8% | 73（低估 40%） |
| 段标签 × 4 | ~100 | 7% | 148 |
| 当前请求 | 9 | | 8 |
| 小计 | **1344** | | |
| + 对话模板脚手架 | ~108 | | |
| **= llama-server 报的** | **1452** ✓ | | |

记忆、历史、摘要、承诺在这一轮都是 0——**新会话，还没有任何记忆**。

三条读法：

1. 回答"你叫什么名字？"，**七成的提示词是工具定义和工具相关策略**。
2. persona——这个伴侣的人格——占 **8%**，在自己的提示词里排第三，前面是两个它这辈子
   不会用到的工具。
3. **工具 715 + harness 398 = 1113，占 77%，是每轮都付的固定成本**，与用户说了什么无关。

---

## 4. 找到的缺陷

### 4.1 预算的尺子是坏的

`eidolon_agent/domain/context/compiler.py:1045`：

```python
def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 3) if text else 0
```

对英文（约 4 字符/token）**高估 45%**，对中文（约 1–1.5 字符/token）**低估 35–40%**。
工具 schema 是英文、persona 和 harness 是中文，两个方向的误差在总数上碰巧抵消了，
但分账不可信。**一个中文伴侣产品，它的预算尺子把中文少算三分之一，而整个预算系统
跑在这把尺子上。**

### 4.2 会裁的看不见，看得见的不会裁

* `ContextBudget.prune()` 与 `_budget_totals()` 只接受 `segments`。工具 schema 不是
  segment，走请求的 `tools=[]`（`--jinja` 会把它们渲染进提示词，llama-server 计入
  prompt token）。**所以 `max_token_budget: 6000` 从未包含提示词里最大的那一块。**
* `harness/realtime.py:84-93` 的 `tool_schema_budget()` 算出 `schema_budget_exceeded`，
  全仓**只有一个消费者**——`turn_trace_summary.py:158` 里的一个布尔字段。
  实际 715（估算器口径 1025）对 `tool_schema_budget_tokens: 800`，这个标志现在就是
  true，而什么都没发生。

### 4.3 工具无条件注册

`app/runtime/bootstrap.py:304-310` 把四个工具全部注册，harness 只藏了 `emit_event`。
没有按场景、按 host、按模型能力的裁剪。`delegate_to_coworker` 一个 463 token，用途是
"把复杂任务交给后台 cowork"。

harness 策略 398 token 里约一半也是工具相关的（"delegate_to_coworker 是复杂任务的唯一
委托入口"、"调用后不要编造最终结果"、"工具结果返回后再总结"…）。**工具不上场时这些
句子也不该上场——它们和工具是一件事，不是两件。**

### 4.4 输出侧：两个都是声明而非行为

* **`max_tokens` 根本没传。** `domain/agent/turn.py:304` 那次 `stream()` 只给了
  tools / model / temperature / request_id。回复长度**无界**——这一轮 18 token 停下来
  是模型自己吐了 EOS。极端情况它能写到上下文填满（2048 − 1452 = 596 token），
  在 1.92 tok/s 下是 **5 分钟**。
* **`output_reserve_tokens: 500` 什么都没预留。** 它被声明 → 拷进 harness budget →
  拷进 guard dict → 渲染进 trace，然后结束。`ContextBudget.prune()` 用原始 6000，
  不减这 500。

唯一真正约束输出的是 channel 的 `aggregator_hard_max_chars`（本地 60 / 云端 80），
但它切的是**每段**，不是总长。

### 4.5 契约被临时改动摘掉了

`eidolon_channel/…/grpc_llm.py:58` 现在是：

```python
# Temporary diagnostic allowance requested for RK3588 cold-prompt testing.
FIRST_DELTA_TIMEOUT_S = 30.0
```

提交 `d26f7b3` 把 `give_up_after_s(generation_allowance_s())` 的推导连同 `turn_latency`
的 import 一起删了。于是 `turn_latency` 契约目前只管住 local_asr（5.4 s）和
local_tts（4.2 s），**LLM 那一跳已经脱离契约**。

### 4.6 上下文预算和这台主机供得起多少无关

`max_token_budget: 6000` 是产品常量。RK3588 上可持续约 **950** token（decode 穿过
3.43 门槛处），差 **6.3 倍**。Agent 不知道对面是 DeepSeek 还是板上的 1.7B。

---

## 5. 贯穿这一切的模式

六处默认值，每一处**对云端模型都是对的**：

| 位置 | 现值 | 它假设的成本模型 |
| --- | --- | --- |
| `litellm_provider.warmup()` | 送 `content="."`，注释写"预热 DNS/TCP/TLS/client cache" | 云端：握手贵，prefill 免费 |
| `channel.yaml` `preemptive.enabled` | `false`，理由"A/B 显示 eos=400 时首音无收益" | 云端：prefill 免费，提前算没用 |
| `bootstrap.py:304-310` | 工具无条件全挂，715 token/轮 | 云端：1000 token 无感 |
| `TurnSettings.max_token_budget` | 6000，且看不见工具 | 云端：上下文便宜 |
| `TurnSettings.output_reserve_tokens` | 500，纯装饰 | 云端：输出不会撑爆窗口 |
| `grpc_llm.FIRST_DELTA_TIMEOUT_S` | 30.0 硬编码 | 临时诊断，没撤回 |

**没有一处是 bug，它们都是对着云端算过的。** 本地模型把 prefill 从"免费"变成
"每 85 token 一秒"，这六个假设同时失效。这也是为什么单点优化（绑核、换量化）
救不回来——要改的是成本模型本身。

---

## 6. 已经做完的部分（release `rk3588-20260909T171415`）

§2.29 的核分配修复仍然有效，曲线证明它在各个上下文尺寸上都成立：

* LLM 的 prefill 与 decode 由 llama.cpp 自己的两个掩码分开
  （`--cpu-mask f --threads 4 --cpu-strict 1 --cpu-mask-batch ff --threads-batch 8`），
  prefill 从钉 A55 的 20.2 tok/s 提到 **80–90**。
* `EIDOLON_TTS_CPU_AFFINITY=4-6` 接线——这一行此前一直是注释，而 §2.24/§2.25 的整机
  余量正是在"TTS 钉 A76"的前提下测的，生产跑的一直是没测过的组合。接上之后
  TTS + decode 同轮的缓冲最低点从 **−620 ms（真断音）** 变 **+795 ms**。

**它只是没打在要害上**：同样的 4.3 倍，对着 1452 token 是 18 秒而不是 3.2 秒。

---

## 7. 计划

### P0 —— 先把尺子和闸门修对（没有设计取舍）

| # | 事 | 位置 |
| --- | --- | --- |
| P0-1 | **换掉 `_estimate_tokens`**：用模型的分词器，或至少中英分开的系数 | `eidolon_agent/domain/context/compiler.py:1045` |
| P0-2 | `FIRST_DELTA_TIMEOUT_S` 改回契约推导，撤 `d26f7b3` 的硬编码 | `eidolon_channel/…/grpc_llm.py:58` |
| P0-3 | `max_tokens` 真的传下去 | `eidolon_agent/domain/agent/turn.py:304` |
| P0-4 | `output_reserve_tokens` 真的从预算里减掉 | `ContextBudget.prune()` |

**P0-1 必须排第一**：在尺子修对之前，后面所有预算都是假的。

### P1 —— 最大的杠杆：工具进预算，并按需暴露

| # | 事 |
| --- | --- |
| P1-1 | `ContextBudget` 要看得见工具 schema（现在会裁的看不见、看得见的只报告） |
| P1-2 | 工具按需暴露，而不是无条件注册四个 |
| P1-3 | harness 策略里工具相关的句子随工具一起进出——它们是一件事 |

**收益**（按 §2.1 的曲线推算，标注为推断）：

| | 现在 | P1 之后 |
| --- | ---: | ---: |
| 提示词 | 1452 | **~537** |
| decode | 1.92 tok/s（低于 3.43 门槛） | **~4.2 tok/s**（越过门槛） |
| 冷 prefill | 28.05 s | **~6.3 s** |

> ⚠️ **这不是纯技术改动。** 工具是产品功能，按需暴露意味着某些轮次里 agent 确实做不了
> 那件事。这个取舍要产品决定。

### P2 —— 架构层：可持续上下文由 Host 宣告

| # | 事 |
| --- | --- |
| P2-1 | Host 宣告一个"可持续上下文"，Agent 按它做预算，而不是按产品常量 6000 |
| P2-2 | `--ctx-size` 由那个宣告推出，不是魔数 2048 |

**这个形状本仓已经解过一次。** TTS 区分了两个上限：

* `protocol.MAX_TEXT_CHARACTERS = 400` —— 请求会不会被拒
* `config.SAFE_TEXT_CHARACTERS = 60` —— 听的人会不会听到断音；这台 Host 实测出来的
  特性，经 `/v1/info` 的 `safe_text_characters` 字段告知调用方（HOST-RK3588.md §2.28）

上下文缺的是同一对：**"装不装得下"（ctx-size 2048）和"答不答得够快"（~950）是两回事**，
现在只有前者，而且它不由 Host 宣告。

### P3 —— 第一轮

| # | 事 |
| --- | --- |
| P3-1 | `warmup` 预热真实的稳定前缀，而不是 `"."` |
| P3-2 | 重新评估 `preemptive.enabled`——那次 A/B 是对着**云端** LLM 测的，在那里 prefill 免费所以"提前算没收益"；本地 prefill 每 85 token 一秒，同一个 A/B 会得出相反结论 |

---

## 8. 一个短期内会撞上的预测

固定底座 1113 token（工具 715 + harness 398），`--ctx-size 2048`，只剩 **935 token** 给
历史、记忆和回复。`history_context_window: 4`、`memory_top_k: 5`——聊上几轮就顶穿 2048。
顶穿之后 llama.cpp 挪窗口，**顺带毁掉前缀缓存**，于是每一轮都退化成冷 prefill。

**01:43 那一轮是新会话、记忆为空，是最好的情况，不是典型情况。**

---

## 9. 对端云方案选择的影响

这份调研改变了此前"本地 LLM 余量 1.5×、本地 TTS 只有 1.02–1.18×"的判断——那个 1.5×
是在 120 token 上下文下算的。按真实生产上下文，**本地 LLM 的余量是 0.56×
（1.92 / 3.43），比 TTS 还差**。

但 P1 做完之后（提示词 → ~537，decode → ~4.2），余量回到 **1.2×**。
**所以方案选择应当在 P1 落地之后重新评估，而不是现在。**

对云端路径 P0-1 / P0-3 / P1-1 **同样是收益**：715 token/轮的工具定义在云端是钱和缓存
命中率的问题，而它恰好在提示词最前面、最稳定，是缓存最友好的位置
（DeepSeek 缓存命中价是未命中的 1/30），现在这个优势没被利用。

---

## 10. 发布与协作注意

* **板子是共享的。** 本次调研全程只读，没有重启任何服务、没有部署。
* 2026-09-10 02:04 起板上 `/etc/eidolon/channel.yaml` 变为
  `stt_provider: bailian` / `tts_provider: bailian`，当前 release
  `rk3588-cloud-20260910`（02:05 激活）——另一个 session 正在部署的云端配置。
  01:43 那一轮在此之前，跑的是全本地配置，数据不受影响。
* settings 改动随 release 走；改变一个 settings 值**含义**的改动需要两次发布
  （cross-release 规则：候选 settings 必须同时被当前 release 的解释器和候选解释器读懂）。
  P2-1 属于这一类。
* 涉及四个仓：`eidolon_agent`（P0-1/3/4、P1 全部、P3-1）、
  `eidolon_channel`（P0-2、P3-2）、`eidolon_sdk`（P2-1 的契约）、
  `eidolon_models`（P2-2 的 `--ctx-size`）。

---

## 11. 还没测的

* **P1 之后的真实提示词尺寸**。~537 是按"去掉三个工具 + 一半 harness"推的，实际取决于
  按需暴露怎么设计。
* **暖前缀在生产路径下的真实命中率**。`types.py` 把
  `PERSONA_STATE / SUMMARY / MEMORY / COMMITMENT / REALTIME` 全标成 `volatile`（每轮重算），
  而 `SEGMENT_ORDER` 让它们之后的一切每轮重读。同一个文件记着一条 RK3588 实测：
  一个只变了两位小数的 `realtime` 块，历史在它后面时重读 113 token，历史在它前面时
  27 token。生产路径下每轮的增量到底多大，没有测过。
* **P1 之后的三方共存**（LLM + TTS + ASR 同时）。§2.29 只测了两方。
