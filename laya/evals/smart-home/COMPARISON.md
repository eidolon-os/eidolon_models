# 智能家居评测：不同 checkpoint 对比

同一套 182 条用例、同样的三道题与问法、同一台 Mac（torch + MPS）。模型：laya-multilingual、laya-multilingual·head1024、laya-cn-a、MacJev-322M-4K、laya-zh-v2、decider-0.8B、OpenSparX-cabin-0.8B。

## 结论

两个社区 checkpoint 都已按各自训练时的格式接入（作者自带的 `predict.py` / `macjev.py` 与我们的服务在同一输入上概率一致），
所以下面的差异来自模型本身，不是接入方式。

1. **MacJev 最好，但只好一点**：意图 66%→73%，控制召回 67%→79%，隐含意图 10%→35%，分房间意图 94%；
   控制端到端 41%→**46%**，仍低于规则基线的 60%。相对官方修好 18 个意图、弄坏 5 个，**没有明显退步**——
   代价是更容易把非命令判成控制（非命令场景意图 82%→59%）。它的温度把概率压得很平，p≥0.9 的覆盖只剩 2%。
2. **laya-cn-a 反而更差**：控制端到端 26%。它在状态查询上更好（40%→75%），但意图修好 15 个、弄坏 43 个，
   控制召回掉到 44%，而且更自信。它只在自己的中文意图格式上训练、没有回放通用数据，是典型的“训窄了”。
3. **头部预算不是瓶颈**：同一权重把预算从 256 放到 1024，只多对 1 条（38 个选项的大户型 90%→95%）。
4. **没有任何 checkpoint 会用“多个设备 / 家里没有”这两个出口**（4 个模型都是 0%–7%）。
   社区的微调数据里没有这类“以上都不是 / 同时多个”的选项，这只能靠我们自己的数据教会。
5. **互补性很低**：三道题各挑最强的模型组合，端到端仍是 46%（就是 MacJev 自己）；逐条“任一模型答对”的上限也只有
   意图 81% / 设备 76%。剩下的是所有模型都错的硬骨头：隐含意图、多设备与场景、家里没有、部分状态查询、重噪声语音。
   **没有现成模型能直接用，也不能靠拼接现有模型解决——必须用自己的数据微调。**
6. 速度相同：四个模型结构一样，MPS 上每条（3 道题）p50 约 82 ms；头部预算放大后 p95 从 91 ms 升到约 143 ms
   （38 个选项的用例不再被截断、序列变长）。

## 第二轮：中文 laya 微调与两个 Qwen-0.8B 小模型（2026-09-25 晚）

同一套 182 条、同样三道题，再加三个底座：魔搭上的 **laya-zh-v2**（中文分诊 + 客服 RLCD 微调）、
**decider-0.8B**（Qwen3.5-0.8B-Base，在答案槽读字母 logit，训练数据全英文）、
**OpenSparX-cabin-0.8B**（Qwen3.5-0.8B + LoRA + pointer head，中文车控合成数据）。后两个不是 laya，
用各自作者的推理代码接进同一个 Jev 协议（decider 用作者自带的服务；OpenSparX 只有 `inference.py`，
包成 [adapters/opensparx_serve.py](adapters/opensparx_serve.py)，分支预算从 256 放到 2048 才装得下 38 台设备）。

7. **laya-zh-v2 没有带来质变**：端到端 44%，介于官方（41%）和 MacJev（46%）之间；意图 +18 / −8，
   隐含意图仍 0%。它的一个好处是意图校准最好：p ≥ 0.9 时覆盖 37%、准确 97%。
8. **两个 Qwen-0.8B 零样本都超过了规则基线**：decider **64%**、OpenSparX **65%**，比最好的 laya 检查点高 18–19 个点，
   分房间 88% / 69%、38 个选项的大户型 90% / 90%、状态查询意图 100% / 60%。
   **decider 是干净的赢**：意图 +38 / −12、设备 +19 / −2；校准最好——设备 p ≥ 0.9 覆盖 64%、准确 98%，意图 41% / 99%。
   **OpenSparX 的高分有水分**：它把几乎所有话都判成"控制"（非命令场景意图 0%，控制召回 100% 但精确率 84%），
   隐含意图场景的"100%"只是因为它对什么都说控制；设备题 +18 / −13，弄坏的不少。
9. **两个出口还是没人会用**：家里没有该设备 0% / 0%，多设备 14% / 21%。换底座不解决这个问题，只能靠数据。
10. **代价是延迟，RK3588 上已实测、放不下**：MPS 上 decider fp16 p50 **573 ms**，OpenSparX fp32 **1.1 s**，laya 82 ms。
    2026-09-25 晚在 opi5max 上用 llama.cpp b10865 跑 decider-0.8B 的 GGUF（转换要加 `--no-nextn` 去掉 MTP 层），
    40 条样本、每条 3 道题约 551 token 的 prefill：

    | 配置 | p50 | p95 | 吞吐 | 与 Mac fp16 答案一致（120 题） |
    |---|---|---|---|---|
    | f16，4×A76 | 7.50 s | 10.9 s | 74 tok/s | 120/120，Δp 0.001 |
    | **q8_0，4×A76** | **4.16 s** | 6.3 s | 135 tok/s | 119/120，Δp 0.006 |
    | q4_0，4×A76 | 3.72 s | 5.7 s | 150 tok/s | 106/120，意图准确 34 → 27，**不能用** |
    | q8_0，8 核全开 | 5.37 s | 7.3 s | 105 tok/s | 同 q8 |
    | q8_0，4×A55 | 19.1 s | — | 29 tok/s | （36 条） |
    | q8_0，A76，一行三题 | 4.27 s | 6.4 s | 127 tok/s | 115/120，Δp 最大 0.48 |

    laya 在同一块板上是 163 ms（CPU）/ 59 ms（NPU）。差 25–70 倍。瓶颈是 prefill 的 token 数：设备题一行 296 token
    （20 台设备各带描述）；缩到只给设备名、按房间预筛到 ≤ 8 个候选，估计 1.5 s 上下，仍差一个量级。
    8 核全开比只用 A76 慢（A55 拖慢批处理）；一行三题不省时间（state 只有 14 token，没有可共享的前缀）；int4 翻答案、int8 安全。
    **用户结论：耗时不满足需求，小 LLM 路线不上端侧。**
11. **互补组合**：OpenSparX 意图 + decider 设备 + OpenSparX 动作 = 71%，但 OpenSparX 的"意图优势"来自过度触发，
    这个组合在真实流量里会把闲聊当命令，不能当方案。

**这轮改变的判断**：零样本下，"小 LLM 读选项 logit"路线在中文上比"编码器 + 决策头"路线强约 20 个点、校准好得多，
但 RK3588 上一条用例 4 秒，端侧放不下。所以它的角色定为**老师**（给训练数据打软标签、做上限参照），
端侧底座仍是编码器路线（laya-multilingual / MacJev），靠我们自己的数据微调把 41–46% 追上去。

## 后续方案

1. **起点**：端侧底座只走编码器路线——官方 laya-multilingual 为主线，同一份数据在 MacJev 上也训一次做 A/B。
   laya-cn-a、laya-zh-v2、OpenSparX 不作为起点；decider-0.8B 已在 RK3588 上实测 4 s/条，不上端侧，
   但用来给训练数据打软标签（零样本 64%，比人工从零标快）。
2. **配方**（社区验证过、并被本次对比印证）：解冻最后 6–8 层 + 决策头；软标签交叉熵，score 题加序数损失；
   训练时打乱选项顺序、随机化设备与成员清单；**混入通用决策数据做回放**（MacJev 有、laya-cn-a 没有，这是两者最关键的差别）；
   训练后按题型在留出集上拟合温度。
3. **数据**：按“硬骨头”造——
   - 每类题目有 15%–25% 的样本以“多个 / 没有 / 不确定 / 需要澄清”为正确答案，教会模型使用出口；
   - 隐含意图、状态查询（“好了吗 / 还要多久”）、“开始 / 回去充电”类动词；
   - 大量非命令负例（提到设备的闲聊、常识问题、购物），防止像 MacJev 那样过度触发；
   - 同音错字用 TTS→ASR 回环生成；反事实最小对（只改一处，标签就翻转）。
4. **评测门槛**：本套 182 条 + 陪伴路由金标集 + 客服集都作为留出集，每个新 checkpoint 必须
   **所有场景都不退步**才能替换线上版本；敏感设备（门锁、车库门、安防）照旧强制二次确认。

## 多场景要不要合在一起训？——要，但“场景”作为题目的一部分，而不是前置的硬分类器

- **证据**：只训单一场景的 laya-cn-a 把别的格式训坏了（意图 −43 / +15）；混合数据加回放的 MacJev 在陌生任务上有提升且几乎不退步（+18 / −5）。
  laya 的任务是由“题目 + 选项”定义的，多个场景本来就能共用一个 checkpoint；合训还能共享中文口语、语音噪声、称呼与指代的理解，
  部署也只需要一个常驻模型。
- **边界只有合训才学得到**：代价最高的错误都出在场景交界处——团队群聊里说“打开客厅的灯”、“空调一般开多少度”是提问不是控制、
  “悟空，把灯关了”既是在叫伙伴又是家居指令。这些对照样本只存在于混合数据里。
- **场景判断本身也是一道题，但别做成串联的第一关**：两级串联的错误会相乘（两级各 90%，合起来只有 81%）。建议：
  1. 先用确定的上下文（当前是哪个 App / 模式 / 设备）——免费、可靠，能定的就不问模型；
  2. 需要问时，把“场景”和“意图、目标”放在同一次请求里一起问（一次前向，每多一道题在 MPS 上约 +25 ms），由编排器综合决定；
     或者更进一步，把场景直接并进意图的选项（如：控制家居 / 查询家居 / 找伙伴聊天 / 对某位伙伴说话 / 传话 / 停止 / 无关），少一级就少一次误差；
  3. 场景题同样要有“不确定 / 同时涉及多个”的出口，配合校准后的阈值走澄清。
- **合训的护栏**：按场景做平衡采样（按数据量做温度采样，别让大场景淹没小场景）；按题型分别校准温度；
  每个场景一份留出评测，出现持续的负迁移、或部署约束不同（例如 RK3588 端侧只能用 256 token）时，才把那一块拆成单独的 checkpoint——
  而不是一开始就按场景拆模型。

## 总体

| 指标 | laya-multilingual | laya-multilingual·head1024 | laya-cn-a | MacJev-322M-4K | laya-zh-v2 | decider-0.8B | OpenSparX-cabin-0.8B |
|---|---|---|---|---|---|---|---|
| 意图（三分类） | 66% | 66% | 51% | 73% | 71% | 80% | 84% |
| 是否控制：精确率 / 召回率 | 95% / 67% | 95% / 67% | 98% / 44% | 90% / 79% | 97% / 71% | 98% / 81% | 84% / 100% |
| 设备 | 71% | 72% | 68% | 71% | 70% | 82% | 74% |
| 动作 | 76% | 76% | 74% | 76% | 76% | 85% | 89% |
| **控制端到端** | **41%** | **42%** | **26%** | **46%** | **44%** | **64%** | **65%** |
| 设备 p≥0.9：覆盖 / 准确 | 70% / 88% | 70% / 88% | 35% / 95% | 2% / 100% | 0% / 0% | 64% / 98% | 48% / 97% |
| 意图 p≥0.9：覆盖 / 准确 | 18% / 85% | 18% / 85% | 36% / 60% | 0% / 0% | 37% / 97% | 41% / 99% | 65% / 93% |
| 延迟 p50 / p95 | 86 / 93 ms | 82 / 144 ms | 82 / 144 ms | 83 / 142 ms | 85 / 99 ms | 573 / 622 ms | 1115 / 1638 ms |
| 头部预算 / 最大长度 | 256 / 1024 | 1024 / 1024 | 512 / 1024 | 1024 / 4096 | 256 / 512 | — / — | — / — |

规则基线（同一套用例）：意图 76%、设备 71%、控制端到端 60%。

## 分场景：意图

| 场景 | laya-multilingual | laya-multilingual·head1024 | laya-cn-a | MacJev-322M-4K | laya-zh-v2 | decider-0.8B | OpenSparX-cabin-0.8B | 规则基线 |
|---|---|---|---|---|---|---|---|---|
| `01-explicit-control` | 82% | 82% | 55% | 92% | 88% | 98% | 100% | 100% |
| `02-implicit-intent` | 10% | 10% | 5% | 35% | 0% | 25% | 100% | 5% |
| `03-room-disambiguation` | 75% | 75% | 44% | 94% | 100% | 100% | 100% | 100% |
| `04-status-query` | 40% | 40% | 75% | 50% | 65% | 100% | 60% | 95% |
| `05-non-command` | 82% | 82% | 68% | 59% | 77% | 55% | 0% | 77% |
| `06-multi-device-scene` | 50% | 50% | 43% | 57% | 57% | 64% | 100% | 79% |
| `07-asr-noise` | 70% | 70% | 65% | 80% | 75% | 80% | 100% | 30% |
| `08-large-inventory` | 85% | 85% | 40% | 90% | 85% | 95% | 100% | 100% |
| `09-device-not-in-home` | 90% | 90% | 50% | 90% | 90% | 100% | 100% | 90% |

## 分场景：设备

| 场景 | laya-multilingual | laya-multilingual·head1024 | laya-cn-a | MacJev-322M-4K | laya-zh-v2 | decider-0.8B | OpenSparX-cabin-0.8B | 规则基线 |
|---|---|---|---|---|---|---|---|---|
| `01-explicit-control` | 98% | 98% | 98% | 98% | 95% | 100% | 100% | 92% |
| `02-implicit-intent` | 45% | 45% | 45% | 45% | 40% | 75% | 50% | 15% |
| `03-room-disambiguation` | 69% | 69% | 56% | 62% | 62% | 94% | 75% | 44% |
| `04-status-query` | 100% | 100% | 85% | 100% | 100% | 100% | 90% | 95% |
| `06-multi-device-scene` | 0% | 0% | 7% | 0% | 0% | 14% | 21% | 93% |
| `07-asr-noise` | 85% | 85% | 85% | 85% | 85% | 95% | 85% | 35% |
| `08-large-inventory` | 90% | 95% | 85% | 95% | 90% | 100% | 95% | 90% |
| `09-device-not-in-home` | 0% | 0% | 0% | 0% | 10% | 0% | 0% | 100% |

## 分场景：控制端到端

| 场景 | laya-multilingual | laya-multilingual·head1024 | laya-cn-a | MacJev-322M-4K | laya-zh-v2 | decider-0.8B | OpenSparX-cabin-0.8B | 规则基线 |
|---|---|---|---|---|---|---|---|---|
| `01-explicit-control` | 68% | 68% | 40% | 75% | 72% | 95% | 90% | 90% |
| `02-implicit-intent` | 0% | 0% | 0% | 0% | 0% | 15% | 45% | 5% |
| `03-room-disambiguation` | 56% | 56% | 31% | 50% | 56% | 88% | 69% | 38% |
| `06-multi-device-scene` | 0% | 0% | 0% | 0% | 0% | 7% | 21% | 71% |
| `07-asr-noise` | 45% | 45% | 50% | 55% | 50% | 75% | 70% | 20% |
| `08-large-inventory` | 65% | 70% | 25% | 75% | 65% | 90% | 90% | 90% |
| `09-device-not-in-home` | 0% | 0% | 0% | 0% | 10% | 0% | 0% | 90% |

## 相对 laya-multilingual：修好了多少、弄坏了多少

| 模型 | 意图 修好 / 弄坏 | 设备 修好 / 弄坏 | 动作 修好 / 弄坏 |
|---|---|---|---|
| laya-multilingual·head1024 | +0 / −0 | +1 / −0 | +0 / −0 |
| laya-cn-a | +15 / −43 | +7 / −12 | +4 / −6 |
| MacJev-322M-4K | +18 / −5 | +2 / −2 | +0 / −0 |
| laya-zh-v2 | +18 / −8 | +1 / −3 | +1 / −1 |
| decider-0.8B | +38 / −12 | +19 / −2 | +15 / −3 |
| OpenSparX-cabin-0.8B | +51 / −19 | +18 / −13 | +20 / −3 |

## 互补性：三道题各取一个模型的答案组合

| 意图来自 | 设备来自 | 动作来自 | 控制端到端 |
|---|---|---|---|
| OpenSparX-cabin-0.8B | decider-0.8B | OpenSparX-cabin-0.8B | 71% |
| OpenSparX-cabin-0.8B | decider-0.8B | decider-0.8B | 69% |
| OpenSparX-cabin-0.8B | OpenSparX-cabin-0.8B | decider-0.8B | 66% |
| OpenSparX-cabin-0.8B | OpenSparX-cabin-0.8B | OpenSparX-cabin-0.8B | 65% |
| decider-0.8B | decider-0.8B | decider-0.8B | 64% |

（同一模型三道题都用自己的答案：OpenSparX-cabin-0.8B 65%，decider-0.8B 64%，MacJev-322M-4K 46%，laya-zh-v2 44%，laya-multilingual·head1024 42%，laya-multilingual 41%，laya-cn-a 26%）

逐条取“任一模型答对就算对”的上限：intent 99%，device 86%，action 97%。

## 所有模型都答错的用例（新数据必须覆盖的“硬骨头”）

| 场景 | 意图全错 | 设备全错 | 例子 |
|---|---|---|---|
| `01-explicit-control` | 0 | 0 |  |
| `02-implicit-intent` | 0 | 3 | 阳光太刺眼了；衣服洗好了，要晾一下；脚底下好冰 |
| `03-room-disambiguation` | 0 | 1 | 卧室大灯关了，床头灯留着 |
| `04-status-query` | 0 | 0 |  |
| `05-non-command` | 1 | 0 | 空调一般开多少度最省电 |
| `06-multi-device-scene` | 0 | 9 | 把家里的灯全部关掉；我要睡觉了；我出门了 |
| `07-asr-noise` | 0 | 0 |  |
| `08-large-inventory` | 0 | 0 |  |
| `09-device-not-in-home` | 0 | 9 | 启动洗碗机；把地暖打开；烤箱预热到180度 |

## 复现

```bash
cd laya
EIDOLON_LAYA_MODEL_DIR=models/laya-cn-a/178eb2c0 scripts/eidolon-laya fetch
EIDOLON_LAYA_MODEL_DIR=models/macjev-322m-4k/92b182e6 scripts/eidolon-laya fetch
HEAD_MAX_LEN=1024 LABEL_PREFIX=laya-multilingual-h1024@ evals/smart-home/run_all.sh
MODEL_DIR=models/laya-cn-a/178eb2c0 LABEL_PREFIX=laya-cn-a@ evals/smart-home/run_all.sh
MODEL_DIR=models/macjev-322m-4k/92b182e6 LABEL_PREFIX=macjev-322m-4k@ evals/smart-home/run_all.sh
EIDOLON_LAYA_MODEL_DIR=models/laya-zh-v2/92ae01f5 scripts/eidolon-laya fetch   # 魔搭
MODEL_DIR=models/laya-zh-v2/92ae01f5 LABEL_PREFIX=laya-zh-v2@ evals/smart-home/run_all.sh
```

两个 Qwen 小模型不是 laya，各自在独立的 venv 里起 Jev 协议的服务，再用 run_eval.py 打它：

```bash
# decider：作者自带的服务（pip install 'decider-ai[serve]'；numpy<2，所以要单独 venv）
DECIDER_MODEL=<Mapika/decider-0.8b 快照目录> DECIDER_DEVICE=mps uvicorn decider.serve:app --port 8773
python3 evals/smart-home/run_eval.py --url http://127.0.0.1:8773 --label decider-0.8b@mac-mps
# OpenSparX：只有 inference.py，用 adapters/opensparx_serve.py 包成服务（钉作者的 transformers==5.8.1、peft==0.19.1）
python evals/smart-home/adapters/opensparx_serve.py --repo <adapter 快照> --base <Qwen3.5-0.8B 快照> --device mps --port 8772
python3 evals/smart-home/run_eval.py --url http://127.0.0.1:8772 --label opensparx-cabin-0.8b@mac-mps
```
