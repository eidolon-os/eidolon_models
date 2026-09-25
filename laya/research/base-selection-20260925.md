# 端侧中文快速决策模型：底座选型

快照 2026-09-25。起因：用户给了两篇文章（[Pinggy：开源 Jev 替代](https://pinggy.io/blog/best_open_source_jev_alternatives_self_hosted_decision_models/)、
[HF 博客：Jev vs laya](https://huggingface.co/blog/sora-2/jev-vs-laya-hosted-api-or-open-weights-2026-guide)），要求找一个**适合微调训练的开源快速决策模型**。
第二篇没有增量（Jev vs laya 的泛泛对比，JevBench v1.3.0 laya 54.4 / Jev 74.4）；第一篇的增量是 Von、GLiNER2、NanoJev、jevlike、SemIf 和 jabr 基准。
这些加上再搜出来的 Kotoba typed-decisions、Mirave、Reflex-S1、360 / 蚂蚁的中文 ModernBERT，本文按我们的约束逐一过一遍。

**约束**（来自前面几轮评测和 RK3588 实测）：
1. RK3588 上 < 200 ms —— 编码器体量（≤ 350M，非嵌入 ≤ 120M），能导静态 ONNX / RKNN；Qwen-0.8B 路线实测 4 s/条，已出局。
2. 中文（最好 100+ 语）。
3. 能用自己的数据微调：有训练脚本、数据格式清楚、单卡几小时以内。
4. choice / noul / score 三种题型，选项 ≥ 38（大户型设备清单），要有"以上都不是"出口。
5. 出厂精度不重要——所有模型在我们的评测里都要微调；重要的是**底座的中文能力 + 头的结构 + 训练配方 + 部署路径**。

## 1. 候选逐个过

| 候选 | 底座 / 参数 | 中文 | 题型 | 微调 | ONNX / NPU | 延迟证据 | 判定 |
|---|---|---|---|---|---|---|---|
| **laya-multilingual**（官方） | mmBERT-base 322M（非嵌入 110M） | XNLI-zh 77.7；三套社区中文微调配方 | 三种原生（qtype embedding + 按题型 / 选项数分桶温度） | Kaggle 笔记本（RLCD + 温度拟合）；chinese-laya、laya-zh-v2、laya-mlx-zh 都训过 | 官方 ONNX；**我们已在 RK3588 NPU 跑通 59 ms@64 / 105 ms@128**；AXERA AX650 S256 27.7 ms；LiteRT S26 54 ms | 板上 163 ms CPU / 59 ms NPU | **主线** |
| MacJev-322M-4K | 同上微调 | 我们评测 46%（最好） | 同上 | 同上 | 同上 | 同上 | A/B 对照 |
| [Von](https://github.com/wfzyx/von) 1.2 | ModernBERT-large 395M（英文） | **无**；`language: en`，训练 100% 英文 | 三种；option-marker 单序列，MLP 头 | `train_option_marker.py`，JSONL 含软标签，4×T4 / 1×A10G | **无 ONNX**（只有 OpenVINO IR）；1.2 的顺序不变性靠自定义 4D mask + 重置 position_ids，导出负担 | 唯一 CPU 数字 p50 0.42 s（x86）；MPS 56 ms | 出局；**抄配方** |
| [GLiNER2.5-Decide](https://huggingface.co/fastino/GLiNER2.5-Decide) | DeBERTa-v3-large 340M（英文） | 无 | 单 / 多标签；score 用 "0…k-1" 标签模拟；无弃权 | `ExtractorTrainer` + LoRA，JSONL；第三方 140 行 36 s | 官方无 ONNX；社区 q4 13 问对 12 | 48 核 Xeon 64 token 167 ms；9800X3D 300 ms | 出局 |
| GLiNER2.5-multi-Decide（09-24） | mDeBERTa-v3-base 287M | 有底座、无评测（6 句情感 100%） | 同上 | 同上 | DeBERTa 解耦注意力上 RKNN **无先例**（TensorRT 要自定义插件） | 未找到 | 可零样本试一次，不作起点 |
| [OpenJev Verdict](https://huggingface.co/heman10x/rlcd-modernbert-151m) | GLiClass-ModernBERT 151M（英文） | 无 | 三种；**固定 24 选项 + 1 弃权槽** | CE + Brier + 温度；只用 Banking77 / CLINC | ONNX / WASM 35 ms（K=5，< 71 token） | — | 出局（24 选项装不下 38 台设备；2.0 权重拿不到）；**抄弃权槽** |
| [Kotoba typed-decisions](https://github.com/kotoba-lang/typed-decisions) | DeBERTa-v3-large 435M；**ModernBERT-base 149M 变体** | 英文 | 三种（≤ 255 选项） | `modal_app.py`，H100 $0.16–0.26 训完；ModernBERT-base 0.717 | 社区 ONNX | ModernBERT-base forward 19 ms（H100） | 出局；**关键消融值得抄**（见 §2） |
| [Mirave](https://github.com/edoigtrd/Mirave) | XLM-R-large 560M 冻结 + LoRA r16 + 2.1M pointer head | XLM-R 100 语；训练集非英文只 4.2K 行 | 三种；0.797 / ECE 0.022 | `train.py`，16 GB 单卡 3 epoch | 未找到；XLM-R 是纯 BERT 图，最稳 | 未找到 | 560M 超预算；**XLM-R-base + 同配方**可做保守备选 |
| [Reflex-S1](https://huggingface.co/Gowtham25/reflex-s1) | 自研 MoR + MoE 23M / 83M | 英文 | 预置 schema | 有仓库 | 未找到 | L40S 6.6–11.7 ms | 出局（预置 schema、英文） |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | Qwen3-0.6B + 20 万参数头 | 玩具数据一半中文 | 三种；每候选一条路径 + 集合注意力 | 完整脚本，软标签 CE，0.6B 全参 17 GB | 无；CUDA-only 是硬编码 | A100 7 路径 27 ms；**每候选重复整个前缀**，输入比 decider 更长 → 板上 ≥ 4 s | 出局 |
| [jevlike](https://github.com/vinnylarouge/jevlike) | 字节编码器 4 万参数 / 任意冻结 HF 编码器 + 30 行注意力头 | 随编码器 | 只有 choice | 极简；编码器硬冻结 | 零障碍 | 亚毫秒 | 出局（winnow 复现 AUC 0.57；已有 laya 全面覆盖） |
| [SemIf](https://github.com/TheoLeeCJ/SemIf-OpenJev) | 冻结 Qwen，读字母 logit | 无评测 | choice ≤ 16；无 score | 无（零训练） | — | 0.6B 平衡准确率 0.44（随机） | 出局；只抄它的 llama.cpp 读 logit 代码 |
| decider-0.8B / Kev-0.8B | Qwen3.5-0.8B | decider 中文零样本 64% | 三种 | Kev 脚本最好 | GGUF 需 `--no-nextn` | **板上 4.2 s/条** | 出局；**当老师** |

**中文编码器底座**（无现成决策头，要自己训）：

| 底座 | 参数 | 上下文 | 中文证据 | NPU 图 | 备注 |
|---|---|---|---|---|---|
| [Zhinao-ChineseModernBert](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert)（360，2026-03） | 345M（卡片称非嵌入 ~110M，需核实） | 1536 | 1T token（> 65% 中文）；**CLUE 74.63**（RoBERTa-wwm-large 73.63） | 与 ModernBERT-base 同图（rk-transformers 已有 RKNN） | 中文每参数最强；丢掉多语 |
| [mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) | 140M（非嵌入 42M） | 8192 | XNLI-zh 73.2 | 同 mmBERT 图、同 tokenizer | laya 头可直接换底座重训做快速档 |
| [chinese-modernbert-large-wwm](https://huggingface.co/feynmanzhao/chinese-modernbert-large-wwm)（蚂蚁） | 377M（28 层） | 8192 | 1.2T 中文；CMNLI 83.96 | ModernBERT 图 | 28 层比 22 层慢，超预算边缘 |
| XLM-R-base | 278M | 512 | 100 语 | 纯 BERT 图，RKNN w8a8 有先例 | Mirave 配方在 large 上验证过 |
| mDeBERTa-v3-base / EuroBERT-210m / gte-mlm-base / jina-v2-zh | 278–310M | — | 有 | 各有硬伤（解耦注意力无插件 / optimum 无导出 / 自定义算子） | 不推荐 |

## 2. 各家训练配方里值得搬到我们头上的东西

1. **Von**：option-marker 头 `LayerNorm → Linear(H, H/2) → GELU → LayerNorm → Linear(H/2, 1)`；损失 **CE + 0.5·Brier**，支持软标签蒸馏；每个选项只 attend 前缀和自己的 4D mask + 每个选项 position id 从前缀长度重新开始 → **选项顺序不变**（JevBench hard 111 题 × 4 种顺序翻转 49.5% → 0）；输入条件温度（特征：bias / entropy / log10(tokens) / n_options）；noul 零样本先验去偏；score 读期望值不读 argmax（MAE 0.36 vs 0.42）。
2. **Kotoba 消融**：在自定义 `[OPT]` 标记 token 上放打分头"学不动"（lr 1e-5 … 1e-4 全失败），改为**对每个选项文本 token 取均值**（span-mean）立刻收敛。laya 用 `[MASK]` 标记 + 2 层 Transformer 头能训是因为它有 RLCD 预训练阶段；我们从头加头时要做 marker vs span-mean 的对照。
3. **Verdict**：固定弃权槽 `__insufficient_evidence__` + CE + Brier + 温度 → 域外弃权召回 97.5%（但硬负例下塌到 18%）。我们评测里"多个 / 没有"两个出口 0%，就该这么做——只是槽要当普通选项训，且硬负例要进数据。
4. **Kev**：注册式 round（先写 PLAN 再训）、locked 分区只读一次、paired bootstrap 2,000 次、公开否决清单；`--init_from` 保通用能力（回放）。
5. **Mirave**：冻结底座 + LoRA r16 + 2.1M pointer head，16 GB 单卡就能训——预算紧时的降级方案。
6. **laya 官方**：qtype embedding + 按 (题型, 选项数) 分桶温度；RLCD 的 proper-scoring 奖励。

## 3. 中文训练数据可以拿来做回放的

| 数据 | 规模 | 来源 | 公开 |
|---|---|---|---|
| [chinese-laya](https://github.com/yanqiangmiffy/chinese-laya) `datasets/all_zh/` | 4,800 训 / 600 验 / 600 校准 / 2,000 测 | LocalLLaMA/typed-decisions 机翻，保留原概率分布 | 是 |
| [laya-mlx-zh](https://github.com/ZLHAOOO/laya-mlx-zh) `/data` | 9,240 训 + 989 验 + 41 人写测 | 模板 + 规则扰动，软标签 | 是 |
| [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | 520K 行，mailroom-control 114,800 行含 400 组英 / 中 / 土家族 | 程序化生成，CC0 | 是 |
| [datawhale jev-cookbook 第十章](https://github.com/datawhalechina/jev-cookbook) | 计划 1,900 case ≈ 9,500 题 | ShareGPT-zh + DeepSeek 三轮标注投票 | 脚本公开，生成物未发布 |
| laya 官方 feishu_zh / zh_short_commands | 64 + 18 | 手写 / 合成 | 是，**明令不得用于训练**（当评测） |
| zcgnull/laya-zh-v2 的 9,500 项 | — | 合成 | **未公开**（只有权重） |

魔搭上没有独立的中文 laya 训练集；我们自己的 182 条智能家居 + 陪伴路由金标是这个生态里少见的干净标签。

## 4. 决定

**底座：mmBERT-base，从 laya-multilingual 的权重起训（主线），MacJev 做 A/B。** 理由：
- 唯一"多语编码器 + 已训好的三题型头 + 三套中文微调配方 + 三个平台（RKNN / AXERA / LiteRT）的静态导出先例"齐全的底座；RK3588 NPU 我们已量过 59 ms。
- 其它编码器候选要么英文（Von、GLiNER2、Verdict、Kotoba、Reflex），要么没决策头要从零训（Zhinao、mmBERT-small、XLM-R），要么 NPU 路径无先例（DeBERTa 系）。
- Qwen 路线（decider / Kev / NanoJev / SemIf）板上 ≥ 4 s，只当老师。

**两个挑战者，各做一次小实验后再定去留：**
1. **Zhinao-ChineseModernBert + 自训头**（Von 头 + CE+Brier，span-mean vs marker 对照）：中文每参数最强，图与 ModernBERT-base 一致。如果在同一份数据上比 laya 主线高 5 个点以上，值得为它丢掉多语。
2. **mmBERT-small + laya 头重训**：快速档（预期板上 NPU ~30 ms），看中文掉多少。

**先做的三件事：**
1. 用 rk-transformers（rknn-toolkit2 2.3.2）把 mmBERT-base 和 Zhinao 各编一次 fp16 RKNN（S256），板上量 encoder-only 延迟——半天。
2. 造数据：智能家居 + 陪伴路由自有数据（按"硬骨头"切片：隐含意图、状态查询、多设备、家里没有、非命令负例、ASR 错字），decider-0.8B 打软标签 + 人工复核；回放用 chinese-laya + laya-mlx-zh + Open-Jev 中文行。每类题 15–25% 样本以出口为正确答案。
3. 训练脚本：在 laya 0.3.7 的 DecisionModel 上实现 CE + 0.5·Brier、选项顺序不变 mask、固定弃权槽、按题型温度；注册式 round，182 条 + feishu_zh 当 locked 评测。

**不做的：** 不换到 Von / GLiNER2 / Verdict / Kotoba 的权重；不再评估任何 Qwen 小模型的端侧延迟；不用 jevlike / SemIf。
