# eidolon-laya-train：决策模型的训练工作流

从"一个场景的题目定义"到"一个能被 `eidolon-laya serve` 直接加载的 checkpoint"，中间每一步都是
**读文件、写文件的独立命令**，任何一步都能单独重跑、换实现、被别的场景复用。没有全局状态，
没有一个大脚本；一次训练就是一个 `runs/<id>/` 目录，里面每一步的输入输出和 manifest 都在。

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
                                          eval ──▶ eval/*.json（按场景、题型的准确率 / ECE，与基线的门槛比较）
                                                            │
                                         export ──▶ onnx/（复用 eidolon-laya export-onnx）
```

## 一条记录长什么样（所有阶段共用）

```json
{
  "id": "smart-home/02-07",
  "scenario": "smart-home",
  "source": "template",                       // template | llm | import:<name> | teacher:<model>
  "state": {"utterance": "好热啊"},
  "questions": {
    "intent": {"type": "choice", "instructions": "...", "criteria": {"控制": "...", "查询": "...", "无关": "..."}},
    "device": {"type": "choice", "instructions": "...", "criteria": {"客厅空调": "客厅·空调", "...": "...", "没有对应的设备": "..."}}
  },
  "labels": {
    "intent": {"gold": "控制"},                                  // 人工 / 规则金标（可以是列表：多个可接受答案）
    "device": {"gold": ["客厅空调", "主卧空调"], "target": {"客厅空调": 0.6, "主卧空调": 0.35, "...": 0.05}}
  },                                                             // target = 软标签（老师分布，或金标混合）
  "tags": ["implicit-intent"],
  "split": "train"                                               // assemble 之后才有
}
```

`state` / `questions` 就是 Jev 协议的请求体，所以**同一条记录既能训练也能直接打服务评测**。
`labels` 里 `gold` 和 `target` 至少一个；训练用 `target`，没有 `target` 时由 `gold` 生成 one-hot（多个可接受答案平分）。

## 阶段

| 命令 | 输入 | 输出 | 说明 |
|---|---|---|---|
| `gen` | `scenario.yaml` + 生成器配置 | `cases.jsonl` | 生成器是插件：`template`（Python 模块产 case）、`llm`（OpenAI 兼容 API + 提示词模板 + JSON schema）、`import`（把外部数据集——chinese-laya、laya-mlx-zh、Open-Jev、我们的 evals——转成记录） |
| `label` | `cases.jsonl` | `labeled.jsonl` | 老师打软标签：任何 `/v1/systemone` 服务（decider、laya、Jev）；有金标时 `target = α·onehot(gold) + (1−α)·teacher` |
| `augment` | `labeled.jsonl` | `augmented.jsonl` | 变换插件：`exit_injection`（把金标设备从清单里删掉，答案变成"没有对应的设备"）、`asr_noise`（同音错字）、`option_subset`（随机缩减候选）；每条产物带 `derived_from` |
| `assemble` | 多个 `*.jsonl` + 混合配置 | `dataset/` | 按权重混合、按场景 / 题型平衡采样、去重、按 id 哈希切 train / val / calib、**排除 locked 评测集的 id**、写 manifest（来源、条数、哈希） |
| `train` | `dataset/` + 训练配置 | `checkpoint/` | laya `DecisionModel`（mmBERT / 任意 ModernBERT 图）；损失 soft-CE + w·Brier（+ 可选 proper-scoring RL 项）；选项打乱；解冻最后 N 层；输出 laya 格式（`model.safetensors` + `rl_agent_config.json` + tokenizer/encoder 配置），`eidolon-laya serve` 直接加载 |
| `calibrate` | `checkpoint/` + `calib.jsonl` | 更新 `rl_agent_config.json` | 按 (题型, 选项数) 分桶拟合温度，夹在 [0.5, 5] |
| `eval` | `checkpoint/` + 评测集 | `eval/<set>.json` | 逐场景 / 逐题型准确率、ECE、按阈值的覆盖 / 准确；与基线 run 比较，输出门槛结论（不退步才算过） |
| `export` | `checkpoint/` | `onnx/` | 调 `eidolon-laya export-onnx` |
| `run` | `pipeline.yaml` | `runs/<id>/` | 按顺序执行上面各步，每步写 manifest；同一 pipeline 改一个参数就是一个新 round |

## 为什么这样切

- **场景只描述题目，不描述数据来源**：`scenario.yaml` 说"有哪几道题、选项从哪来（固定列表或 state 里的某个字段）、哪些是出口"；数据可以来自模板、LLM、外部数据集，或三者混合。换场景（陪伴路由、客服）只加一个 yaml 和一个生成器。
- **软标签是一等公民**：老师（decider-0.8B 零样本 64%、Jev）打的分布直接进训练；金标只是 α=1 的特例。
- **回放是 assemble 的事，不是 train 的事**：训练脚本只看 `dataset/`，不知道什么是回放；防遗忘靠 assemble 时把别的场景 / 通用数据按权重混进去。
- **评测集是锁定的**：`assemble` 按 id 排除 locked 集；`eval` 只读不改。182 条智能家居 + feishu_zh 就是第一批 locked 集。
- **checkpoint 就是服务的模型目录**：不需要"转换"这一步；`models/<name>/<rev>/manifest.json` 那套校验照旧。

## 目录

```
laya/train/
├── README.md                      本文
├── scenarios/
│   └── smart-home/
│       ├── scenario.yaml          题目定义
│       ├── generators/            本场景的 template 生成器（Python）
│       └── prompts/               llm 生成器的提示词
├── pipelines/                     可复用的 pipeline.yaml
├── data/                          外部数据集的本地副本（gitignore，manifest 记 sha256）
└── runs/                          每次 run 的产物（gitignore，只提交 manifest 与 eval 结果）
```

代码在 `src/eidolon_laya_train/`，命令 `eidolon-laya-train <stage> ...`，安装 `uv sync --extra train`。

## 第一次冒烟（2026-09-25，Mac MPS）

模板生成器 398 条 → 增强后 524 条唯一记录（1,572 道题，2 条和锁定集撞 utterance 被拦下）→
从 laya-multilingual 起训 **1 个 epoch、58 秒** → 按桶校准 → 182 条锁定集：

| | 微调前 | 微调后 |
|---|---|---|
| 总准确率（465 道题） | 70.3% | **78.9%** |
| 意图 / 设备 / 动作 | 66 / 71 / 76% | 85 / 74 / 76% |
| ECE | 0.114 | **0.045** |
| p ≥ 0.9 覆盖 / 准确 | 46% / 90% | 54% / **99%** |
| 隐含意图 / 多设备 / 家里没有 | 22 / 40 / 45% | 55 / 71 / 65% |
| **非命令** | 82% | **0%** |

门槛判定 **FAIL**——非命令切片从 82% 掉到 0%：模型学会了"什么都是控制"（和 OpenSparX 一样的过度触发）。
这是数据问题（模板里非命令负例只有 8 句、措辞单一），不是框架问题；门槛正是为了拦这种"总分涨、切片塌"的 checkpoint。
下一步是用 llm 生成器补多样化的非命令负例和更多隐含意图，再跑 3 个 epoch。
