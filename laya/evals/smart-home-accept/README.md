# 智能家居验收集（1,141 条）

**只在候选版本上跑、只看汇总，永不进训练，也不做误差分析。**（`train/EXPERIMENTS.md` 停止规则第 1 条）
它用来验两条安全标准：**误触发**（无关 → 控制）≤ 2%、**自动执行精度**（判为控制且三题 p ≥ 0.9 时三题全对）≥ 99%。
v2-test 只有约 40 条自动执行、65 条无关，验不了这两条。

## 设计

- 4 个**验收专用户型**（`homes/`，训练、开发、v2 都没用过）：acc-couple（年轻夫妻 + 猫，21 台）、acc-family4（四口之家，30 台，两个孩子房间同类设备多）、
  acc-senior（独居老人，9 台，有紧急呼叫和药盒）、acc-house（带花园车库的独栋，34 台，名字带楼层）。
- 4 位撰写者**四种写法**独立撰写，都不读训练数据、不读任何评测集：
  `W1-day`（一户人家的一天）、`W2-device`（每台设备被提到的方式）、`W3-people`（不同的人怎么说：老人、孩子、口音、醉酒……）、
  `W4-commands`（真实使用里的指令形式，主要供自动执行样本）。
- 撰写者被要求**不写自己拿不准的句子**（两种读法都通的改写或删掉）：金标清楚，但真正模糊的日常说法偏少——验收结果是"说法清楚时"的安全性。
- 组成：**无关 450** / 控制 591 / 查询 100；acc-family4 382 / acc-couple 285 / acc-house 275 / acc-senior 199。
- 和其它评测集（v1 / v2 / v3）或本集同户型近似重复的 59 条已删（`dropped_overlap.jsonl`，都是控制 / 查询的标准短句）；
  与它近似重复的 137 行训练数据已按惯例删除（`../smart-home-v2/purged.jsonl`）。
- **全量双人盲标**（`qa/blind-*.jsonl`）：意图一致 100%（1,141）、设备 100% / 严格同集合 99.7%、动作 100% / 97.9%，0 处分歧；
  14 行列表完整性差异按 LABELING §8 取并集（带 `adjudicated`）。**两位标注者都是模型**，一致率高估了人工之间的一致性；
  宣布安全标准达标之前，应由人工抽检约 100 条金标。

## 自动执行样本够不够

自动执行精度 ≥ 99% 要在约 300 个自动执行样本上才有意义（0 错时 95% 上界约 1%）。591 条控制里有多少被自动执行取决于模型的覆盖率
（r10 在没见过的户型上约 28%）——覆盖率低时样本不足，照实报告区间，不外推。

## 用法

```bash
uv run --extra torch --extra train eidolon-laya-train gen --scenario train/scenarios/smart-home \
  --config train/scenarios/smart-home/eval-import-accept.yaml --out train/scenarios/smart-home/eval/locked-accept.jsonl
python3 evals/double_annotation.py evals/smart-home-accept
# 只对候选版本、只看汇总：
uv run --extra torch --extra train eidolon-laya-train eval --checkpoint <candidate> \
  --eval-set train/scenarios/smart-home/eval/locked-accept.jsonl --no-rows --out <dir>
```
