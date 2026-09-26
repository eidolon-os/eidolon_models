# 智能家居开发集 v3（360 条）：是不是在要求助手

**开发集**：只用于评测和误差分析（可以逐条看），永不进训练——`assemble.yaml` 的 `locked_eval` 排除它，
`purge_training_overlap.py` 清理与它近似重复的训练行。

## 为什么要 v3

r11 把 v2-dev 的误触发压到 1/33，但 v2-test 从 r9 起一直停在 65 条里错 12 条左右；两种造训练数据的办法（照例子、按言语行为分类）
都没动它。v2-dev 的无关样本太少，已经不能指导误触发的工作——需要一套无关句多、且和训练数据**独立产生**的开发集。

## 设计

- 户型：v2 的 4 个评测专用户型（loft / threegen / smallflat / bigvilla），训练从没见过。
- 两位撰写者**用不同的生成方式**独立写，都不读训练数据、不读任何评测集：
  - `A-day-in-life`（threegen 99 + smallflat 81）：先设想住户，按时间过一整天和周末，写这家人真会说出口的话；
  - `B-device-mentions`（loft 73 + bigvilla 107）：逐台设备设想它被提到的情境（评价、抱怨、报修、教别人用、叫别的语音助手……）。
- 每位 180 条：**无关 108（60%）**、控制 54、查询 18；无关句大多提到户型里的设备、一半以上带动作词；控制和查询里有与无关句同框架的真请求。
- **全量双人盲标**（`qa/blind-*.jsonl`，第二位标注者不看金标），`evals/double_annotation.py` 比对：
  意图一致 100%（360）、设备 100% / 严格同集合 97.2%、动作 99.0% / 89.2%；1 处设备分歧（A-157）裁为并集，11 处列表完整性差异按 LABELING §8 取并集，
  改过的行带 `adjudicated`。注意两位标注者都是模型，一致率高估人工之间的一致性。
- 撰写后与训练数据近似重复 34 行（多为"热水器烧好了没"这类短句，两边独立写到一起），按惯例删训练行（`../smart-home-v2/purged.jsonl`）。
  **r12 及之前的模型训练时见过这 34 行**，它们对应的约 25 条 v3 用例对这些模型偏易，报告时另给去掉这些用例后的数字。

## 用法

```bash
uv run --extra torch --extra train eidolon-laya-train gen --scenario train/scenarios/smart-home \
  --config train/scenarios/smart-home/eval-import-v3.yaml --out train/scenarios/smart-home/eval/locked-v3-dev.jsonl
python3 evals/double_annotation.py evals/smart-home-v3
```
