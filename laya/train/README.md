# eidolon-laya-train：决策模型的训练工作流

从"一个场景的题目定义"到"一个能被 `eidolon-laya serve` 直接加载的模型目录"，中间每一步都是
**读文件、写文件的独立命令**，任何一步都能单独重跑、换实现、被别的场景复用。一次训练就是一个
`runs/<id>/` 目录，每一步的输入输出和 manifest 都在里面。

```
scenario.yaml ──gen──▶ cases.jsonl ──label──▶ labeled.jsonl ──augment──▶ augmented.jsonl
                                                                              │
     其它场景 / 回放数据 (labeled.jsonl ...) ─────────────────────────────────┤
                                                                              ▼
                                                              assemble ──▶ dataset/{train,val,calib}.jsonl
                                                                              │
                                              train ──▶ checkpoint/ ◀─────────┘
                                                            │
                                            calibrate ──▶ checkpoint/rl_agent_config.json(temperature)
                                                            │
                                          eval ──▶ eval/*.json（按场景 / 题型 / 切片的准确率、ECE；与基线的门槛比较）
                                                            │
                                        package ──▶ models/<name>/<rev>/          ← 到这里为止与推理平台无关
                                                            │
                          ┌─────────────────────────────────┼───────────────────────────┐
                    torch 直接 serve                 export ──▶ onnx/               （rknn / axera / coreml …）
                                                            │
                                   evals/<scenario>/run_all.sh MODEL_DIR=… ──▶ 用原来的服务评测再打一遍
```

**统一的部分**：从场景到 `models/<name>/<rev>/` 全部与推理平台无关——同一个打包目录，torch 后端直接加载，
ONNX 由 `export` 生成，RKNN / AXERA / CoreML 之类是各自平台的导出器，读同一个目录、写到它旁边的子目录。
换场景（陪伴路由、客服、智能家居）只换 `scenarios/<name>/` 下的文件，命令一个字不改。

目录：

```
laya/train/
├── README.md
├── scenarios/<name>/
│   ├── scenario.yaml          题目定义（本场景唯一必需的文件）
│   ├── gen.yaml               用哪些生成器、各生成多少
│   ├── augment.yaml           用哪些增强、比例
│   ├── assemble.yaml          混合哪些来源、切分比例、锁定评测集
│   ├── train.yaml             训练配方
│   ├── eval-import.yaml       把人工评测集转成锁定集
│   ├── generators/*.py        template 生成器
│   ├── prompts/*.txt          llm 生成器的提示词
│   └── eval/*.jsonl           锁定评测集（只读）
├── pipelines/*.yaml           把上面串起来的一条命令
├── data/                      外部数据集副本（gitignore）
└── runs/<id>/                 每次 run 的产物（gitignore）
```

代码在 `src/eidolon_laya_train/`，命令 `eidolon-laya-train <stage> ...`，安装 `uv sync --extra torch --extra train`
（导出 ONNX 再加 `--extra export`）。

---

## 1. 模型与训练方法

### 1.1 模型是什么

laya 的 `DecisionModel` = **双向编码器（mmBERT-base，22 层 × 768，256k 词表，中文在预训练里）+ 决策头**：

```
[CLS] <题型> 指令 [SEP] [MASK] 选项0 [MASK] 选项1 … [SEP] state [SEP]
```

每个选项前放一个 `[MASK]`；编码后取各 `[MASK]` 位置的向量，加上题型 embedding，过 2 层 Transformer 头
（选项之间在这里互相看见），再由 `LayerNorm → Linear → GELU → Linear(1)` 给每个选项一个 logit，
softmax 就是各选项的概率。三种题型共用同一个头：

| 题型 | 选项 | 输出 |
|---|---|---|
| `choice` 单选 | criteria 的每个键（`名字: 描述`） | 各选项概率、argmax |
| `noul` 是否 | 固定两项 `false / true` | P(true) |
| `score` 有序打分 | 每个等级一项 `level i: 描述` | 概率分布和期望值 |

一道题一次前向，**不生成文本**，所以延迟就是一次编码器前向（RK3588 NPU 59 ms@64 token）。
模型的输入长度预算：`max_len`（整条序列）、`head_max_len`（指令 + 选项部分）。38 台设备带描述时
`head_max_len` 要放到 512，否则每个选项名被截到 4 个 token（配置问题，不是模型问题）。

### 1.2 从哪里起训

**主线：`models/laya-multilingual/1c5edc17/torch`**（官方多语 checkpoint）。它已经会"认设备"（零样本设备题
71%），我们微调的是"判断是不是命令"和"用出口"。`train.yaml` 的 `init` 指向它。
也可以 `encoder: qihoo360/Zhinao-ChineseModernBert` 从一个裸编码器起训（头随机初始化，需要更多数据和 epoch）——
底座选型的依据在 [../research/base-selection-20260925.md](../research/base-selection-20260925.md)。

### 1.3 损失：为什么是这几项

训练目标是**让模型给出的概率分布逼近目标分布**，而不只是"argmax 对"。目标分布（`target`）来自软标签（老师）
或金标（one-hot；多个可接受答案平分）。`train.yaml` 里可组合的四项：

| 项 | 公式 | 作用 | 默认 |
|---|---|---|---|
| soft-CE | −Σ target·log p | 主项，逼近目标分布 | 1.0（必开） |
| Brier | Σ (p − target)² | 压过度自信，校准更好（Von 的配方，jabr 上 ECE 明显更低） | `brier_weight: 0.5` |
| proper score | −(log score + 0.5·spherical − RPS) | laya 的严格 proper 评分规则；RPS 只对 score 题生效，惩罚"等级隔得远" | `proper_weight: 0` |
| policy gradient | 对 logit 加噪采样 G 组、按 proper score 打分、GRPO 式优势 | laya 官方微调笔记本的 RLCD 项；`pg_weight: 1, brier_weight: 0` 就是官方配方 | `pg_weight: 0` |

默认用 soft-CE + Brier：在我们的评测上 1 个 epoch 就把 ECE 从 0.114 压到 0.045，而且没有采样噪声，
训练确定、快。官方的 PG 项留作对照实验（它的理论价值是"直接优化校准"，实测收益要看数据）。

### 1.4 训练循环里的几个要点

- **选项打乱**：每个 epoch 重新随机排列 choice / noul 的选项（score 的等级保持顺序），目标分布跟着排——
  模型学不到"第一个选项更常对"这种位置捷径。laya 官方预训练就是这么做的。
- **部分解冻**：默认只训最后 8 层编码器 + 决策头（`unfreeze_layers: 8`；`-1` 全训）。数据少时全训会把
  底座的通用能力训坏（laya-cn-a 就是例子：训窄了，意图 −43）。
- **学习率**：编码器 2.5e-5、头 1e-4（官方笔记本的值），线性 warmup 6% 再线性衰减，AdamW wd 0.01，梯度裁 1.0。
- **按长度分桶再打乱**：省 padding，MPS 上 432 条记录（1,300 道题）一个 epoch 58 秒。
- **act head 不训**：那是 laya 的"升级 / 不升级"头，我们的路由由编排层决定，它没用。
- **验证集选 best**：按 `val.jsonl` 的 NLL 保存最好的 epoch，不是最后一个。
- **checkpoint 就是 laya 目录格式**：`model.safetensors + rl_agent_config.json + encoder/config.json + tokenizer/`，
  服务代码一行不改就能加载。

### 1.5 校准

训练完的 logit 通常过尖或过平。`calibrate` 在**没参加训练的** `calib.jsonl` 上，按 (题型, 选项数) 分桶
（`choice:2 / 3-5 / 6-10 / 11+`、`noul:2`、`score:…`），对每桶用黄金分割搜一个温度 T 使 NLL 最小，
夹在 [0.5, 5]（laya 运行时也这么夹，防止一个坏温度把 0.24 放大成 0.99）。结果写进
`rl_agent_config.json` 的 `temperature` 和 `temperature_by_options`，推理时自动用。
**校准集必须留出**：在训练过的数据上拟温度会得到退化的尺度（官方笔记本 #191 修过这个坑）。

### 1.6 评测与门槛

`eval` 对锁定集逐题给出 pred / p_top / 是否命中，聚合成：总体、按场景、按题型、按题目、按切片（tag）
的准确率和 ECE，以及 p ≥ 0.5 / 0.7 / 0.9 的覆盖率与准确率（决定上线时的置信阈值）。
`--baseline` 给一个之前的评测目录时做**门槛比较**：任何场景 / 题目 / 切片（≥ 10 条）的准确率掉超过
`--tolerance` 就 FAIL。门槛拦的是"总分涨、切片塌"——第一次冒烟就发生了：总分 70 → 79%，
非命令切片 82 → 0%（模型学会了"什么都是控制"）。

评测集是**锁定**的：`assemble` 按 id 和 utterance 哈希双重排除，生成的数据撞上评测句子也进不了训练。

---

## 2. 训练数据：原理、逻辑、怎么衔接训练

### 2.1 一条记录

```json
{"id": "smart-home/02-07", "scenario": "smart-home", "source": "template:rules",
 "state": {"utterance": "好热啊"},
 "questions": {"intent": {"type": "choice", "instructions": "...", "criteria": {"控制": "...", "查询": "...", "无关": "..."}},
               "device": {"type": "choice", "instructions": "...", "criteria": {"客厅空调": "客厅·空调", "...": "...", "没有对应的设备": "..."}}},
 "labels": {"intent": {"gold": "控制"},
            "device": {"gold": ["客厅空调", "主卧空调"], "target": {"客厅空调": 0.6, "主卧空调": 0.35, "...": 0.05}}},
 "tags": ["implicit-intent", "home:apartment"]}
```

`state + questions` 就是 Jev 协议的请求体——**同一条既能训练，也能原样打服务评测**。
`labels` 里 `gold`（金标，可以是多个可接受答案）和 `target`（软标签分布）至少一个；
训练用 `target`，没有就由 `gold` 生成 one-hot。这就是数据和训练方法的衔接点：**§1.3 的损失逼近的
就是这个 `target`**。

### 2.2 数据从哪来：四层，各管一件事

| 层 | 阶段 | 产什么 | 原理 |
|---|---|---|---|
| ① 场景定义 | `scenario.yaml` | 题目：题型、指令、选项、**出口** | 题目是训练和评测共用的契约；出口（"多个设备 / 没有对应的设备"）作为普通选项固定排在动态清单后面，模型才有机会学会用它 |
| ② 生成 | `gen` | `state` + 每条的动态选项 + 已知的金标 | 三种生成器（下节） |
| ③ 标注 | `label` | `target` 软标签 | 老师模型（任何 `/v1/systemone`）给分布；有金标时 `target = α·onehot(gold) + (1−α)·teacher`（默认 α 0.7） |
| ④ 增强 | `augment` | 派生记录 | 出口注入、候选缩减、同音错字（下节） |

然后 `assemble` 把多份产物按权重混合、按场景平衡、稳定切分、排除锁定集。

### 2.3 三种生成器

**`template`（模板）**——Python 函数，按"切片"造句：措辞模板 × 户型里的设备 × 随机口语前后缀，金标由模板决定。
最便宜、可控、保证每个切片都有覆盖（明确指令 / 隐含意图 / 状态查询 / 非命令 / 多设备 / 家里没有）。
缺点是措辞单一——冒烟里非命令只有 8 句模板，模型直接学成"都是控制"。**模板负责覆盖，不负责多样性。**

**`llm`（大模型合成）**——OpenAI 兼容的 chat 端点 + 提示词模板（`prompts/*.txt`，里面放题目 JSON、一个上下文、要几条）
→ 返回 JSON 数组的 `{state, dynamic?, labels?, tags?}`。负责**语言多样性**和**难例**：反事实最小对（只改一个词标签就翻）、
提到设备但不是命令的闲聊、含糊需要澄清的话、多轮上下文。提示词里要求模型自己给金标，`label` 再用老师覆核。
- 端点：`EIDOLON_TRAIN_LLM_BASE_URL / _MODEL / _API_KEY`（或 gen.yaml 里写 `base_url / model / api_key_env`）。
- **推荐模型**：中文场景用 **Qwen3.6 / Qwen3.8 系列（DashScope `qwen-plus` 一档）或 DeepSeek-V4**——
  中文口语自然、能按 schema 吐 JSON、便宜；社区的中文 laya 数据（datawhale cookbook）用的就是 DeepSeek 三轮标注投票。
  本地可用 llama-server 上的 Qwen（同一个接口），但小模型（≤ 7B）造的句子同质化严重，只适合造"量"不适合造"难"。
- 每次调用采样一个上下文（户型 / 人设 / 场景），`contexts_file` 一行一个，所以同一提示词能覆盖很多户型。

**`import`（外部数据）**——把已有数据集转成记录：`evals_cases`（我们的人工评测格式）、`records`（别的 run 的产物）。
主要用途是**回放**：把通用决策数据按 0.2–0.3 的权重混进来，防止把底座训窄。可用的公开中文集：
chinese-laya（4,800 题，机翻 typed-decisions，保留概率分布）、laya-mlx-zh（9,240 题，模板 + 软标签）、
Open-Jev 数据集里的中文行——写一个 adapter 就能接。

### 2.4 老师：软标签用谁

| 老师 | 中文零样本（我们的 182 条） | 延迟 | 用法 |
|---|---|---|---|
| **decider-0.8B**（推荐） | 端到端 64%，校准最好 | Mac 570 ms/条 | `uv run … uvicorn decider.serve:app --port 8773`，`label --teacher http://127.0.0.1:8773` |
| decider-2B / 4B | 更准（JevBench #1 是 4B） | 更慢，需 GPU | 同上，离线批量打标可以接受 |
| Jev（TypeSafe API） | 参照 | 云端 | 数据出域，只做对照 |
| 当前最好的 laya 检查点 | 自蒸馏 | 82 ms | 迭代后期用，防遗忘 |

老师只在**离线打标**时用，延迟无所谓；它的价值是给出"多个可接受答案之间怎么分"和"错误选项上残余多少质量"，
这正是 soft-CE 和 Brier 需要的。老师不会用的出口（decider 的"家里没有"也是 0%），靠金标 α 兜住。

### 2.5 增强：为什么是这三种

- **出口注入**（`exit_injection`）：把金标设备从清单里删掉，答案变成"没有对应的设备"。这是唯一能批量制造
  "家里没有"正例的办法——真实语料里这种句子和有设备的句子长得一模一样，区别只在清单。
- **候选缩减**（`option_subset`）：金标 + 5 个随机设备 + 出口。让模型见过小户型，不依赖固定的清单长度和位置。
- **同音错字**（`char_confusion`）：按表替换（灯→登、空调→空掉）。模拟 ASR 错字；更好的做法是 TTS → ASR 回环，
  写成一个 transform 插件就能换。

增强的产物带 `derived_from`，切分时和原记录走同一个哈希，不会一个进 train 一个进 val。

### 2.6 数据配比的经验

- 每类题目 **15–25% 的样本以出口为正确答案**（不确定 / 多个 / 没有），否则模型永远不选出口。
- **非命令负例要比命令多样**：提到设备的闲聊、常识问答、购物、抱怨——这是最容易塌的切片。
- **回放 20–30%**：混入通用决策数据，MacJev 有回放（+18 / −5）、laya-cn-a 没有（+15 / −43）。
- 隐含意图（"好热啊"）的金标是产品策略，先定策略再造数据；多个可接受答案就写成列表。
- 数据按"上一轮评测的失败切片"造，不按直觉造。

---

## 3. 一次 run 怎么跑

```bash
cd laya
uv sync --extra torch --extra train --extra export
eidolon-laya-train run --pipeline train/pipelines/smart-home-mps.yaml --run-id r3
```

单步：

```bash
t=eidolon-laya-train
$t gen      --scenario train/scenarios/smart-home --config train/scenarios/smart-home/gen.yaml --out runs/r3/cases.jsonl
$t label    --input runs/r3/cases.jsonl --out runs/r3/labeled.jsonl --teacher http://127.0.0.1:8773 --alpha 0.7
$t augment  --scenario … --config …/augment.yaml --input runs/r3/labeled.jsonl --out runs/r3/augmented.jsonl
$t assemble --config …/assemble.yaml --out runs/r3/dataset --run-dir runs/r3
$t train    --config …/train.yaml --dataset runs/r3/dataset --out runs/r3/checkpoint
$t calibrate --checkpoint runs/r3/checkpoint --calib runs/r3/dataset/calib.jsonl
$t eval     --checkpoint runs/r3/checkpoint --eval-set …/eval/locked-182.jsonl --out runs/r3/eval --baseline runs/r2/eval
$t package  --checkpoint runs/r3/checkpoint --name laya-smart-home           # → models/laya-smart-home/<rev>/
$t export   --model-dir models/laya-smart-home/<rev>                          # → onnx/
MODEL_DIR=models/laya-smart-home/<rev> LABEL_PREFIX=laya-smart-home@ evals/smart-home/run_all.sh mac-torch-mps mac-onnx-cpu
```

新场景：复制 `scenarios/smart-home/`，改 `scenario.yaml` 的题目、写一个 `generators/*.py` 或一份 `prompts/*.txt`、
把人工评测集用 `eval-import.yaml` 转成锁定集。其余文件基本不用动。

---

## 4. 已跑过的结果（2026-09-25，Mac MPS）

模板生成器 398 条 → 增强后 524 条唯一记录（1,572 道题，2 条和锁定集撞 utterance 被拦下）→ 从 laya-multilingual 起训：

| | 微调前 | 1 epoch（58 s） | 3 epoch（180 s） |
|---|---|---|---|
| 总准确率（182 条 / 465 题） | 70.3% | 78.9% | **80.9%** |
| 意图 / 设备 / 动作 | 66 / 71 / 76% | 85 / 74 / 76% | 87 / 73 / 82% |
| ECE | 0.114 | **0.045** | 0.060 |
| p ≥ 0.9 覆盖 / 准确 | 46% / 90% | 54% / 99% | — |
| 隐含意图 / 多设备 / 家里没有 | 22 / 40 / 45% | 55 / 71 / 65% | 63 / 83 / 70% |
| **非命令** | 82% | **0%** | 18% |

门槛判定 **FAIL**：非命令切片塌了——模板负例太少太单一。这是数据问题，正是 §2.3 里 `llm` 生成器该补的；
框架本身从生成到打包再到用 torch / ONNX 服务评测，三条路数字一致（意图 85.2 / 设备 73.8 / 动作 76.4）。

已验证：gen(template / import)、augment、assemble、train、calibrate、eval、package、export、`run` 整链。
**尚未对真实端点验证**：`llm` 生成器、`label` 老师打标（代码在，端点没起过）。
