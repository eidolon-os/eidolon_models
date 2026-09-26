# 智能家居锁定评测集 v2（500 条）

**只用于评测，永不进训练。** `train/scenarios/smart-home/assemble.yaml` 的 `locked_eval` 按 id 和句子哈希排除它；
撰写训练数据时还要跑下面的近似重复清理。

## 为什么要 v2

v1（`evals/smart-home/`，182 条）到 r7 已经 95%，一个切片 20 条，变一条就是 5 个点，区分度快用完了；而且它只用
apartment / house / villa 三个户型——训练数据也大量用这三个户型，v1 测不出模型是在"读设备清单"还是"背设备名"。

## 设计

- **4 个评测专用户型**（`homes/`，训练里从未出现）：loft（LOFT 复式，17 台，柜机 / 挂机 / 筒灯 / 灯带）、
  threegen（三代同堂，25 台，老人房 / 儿童房各有空调和灯）、smallflat（一室一厅，10 台，名字就是品类）、
  bigvilla（三层 + 地下室，40 台，名字带楼层，测长清单）。另外复制了 v1 的三个户型，约 40% 用例用老户型，方便和 v1 对比。
- **写的时候不看训练数据**：四个撰写者被要求不读 `train/data/`，照 v1 的标注口径独立写，只避开 v1 的句子。
- **10 个切片**：v1 的 9 个 + `10-boundary`（问句形式的命令 vs 能力问题、闲聊夹指令、纠正、取消、定时、陌生人让开门、数字不是参数）。

| 切片 | 条数 |
|---|---|
| 01-explicit-control | 60 |
| 02-implicit-intent | 60（含 10 条房间里没有对应设备、7 条"像但不是"的对照） |
| 03-room-disambiguation | 50 |
| 04-status-query | 60（含 10 条无疑问词查询、8 条对照） |
| 05-non-command | 70 |
| 06-multi-device-scene | 50（含 10 条场景词但只针对一台的对照） |
| 07-asr-noise | 40 |
| 08-large-inventory | 30（bigvilla 20 + villa 10） |
| 09-device-not-in-home | 50（26 条近邻错配、10 条名字相近但真的存在的对照、5 条查询不存在的设备） |
| 10-boundary | 30 |

两条有争议的金标放宽为"两种都算对"（`note` 字段写明）：smallflat "客厅里热得像蒸笼"（唯一的空调在卧室）、
threegen "主卧最近返潮"（除湿机在儿童房）。

## 和训练数据的隔离

- `check_leakage.py`：v2 每句对比 v1、全部撰写训练数据（去标点后相同 / 编辑距离 ≤ 2）、外部数据（去标点后相同）。
- `purge_training_overlap.py --apply`：从撰写训练数据里删掉与 v1 或 v2 完全相同或近似重复的行，被删的行记在 `purged.jsonl`。
  选择删训练行而不是改评测句——评测句要保持自然说法，训练数据多，删几十行没有代价。
- 2026-09-26 第一次清理：删了 108 行（57 行近似 v2、**51 行近似 v1**）。v1 以前只按完全相同排除，r3–r7 都在 31 条 v1 用例的
  近似副本上训练过；这 31 条上 r7 是 98.7%，其余 151 条是 94.6%，r7 的 v1 总分因此偏高约 0.7 个点。r8 起两套锁定集都是干净的。
- **以后每加一批撰写数据，先跑一次 `purge_training_overlap.py`。**

## 用法

```bash
# 转成记录格式（每条金标按户型校验）
uv run --extra torch --extra train eidolon-laya-train gen --scenario train/scenarios/smart-home \
  --config train/scenarios/smart-home/eval-import-v2.yaml --out train/scenarios/smart-home/eval/locked-v2-500.jsonl
# 评测一个 checkpoint（pipeline 的 eval 步骤会同时评 v1 和 v2）
uv run --extra torch --extra train eidolon-laya-train eval --checkpoint <ckpt> \
  --eval-set train/scenarios/smart-home/eval/locked-v2-500.jsonl --out <dir>
python3 evals/smart-home-v2/check_leakage.py
```

## 第一次结果（r7 及之前的 checkpoint 训练时还有 57 行近似 v2 的数据，数字略偏高；r8 起干净）

| | v1（182） | v2（500） | v2 老户型 | v2 新户型 | v2 控制端到端 |
|---|---|---|---|---|---|
| r4 | 92.0% | 81.7% | 83.6% | 80.5% | 59.4% |
| r6 | 92.7% | 83.9% | 85.7% | 82.7% | 63.2% |
| r7 | 95.3% | 85.7% | 87.9% | 84.2% | 65.9% |

- 排序和 v1 一致（r4 < r6 < r7）。
- 没见过的户型只差约 3.7 个点：模型基本是在读设备清单。
- v1 → v2 掉约 10 个点主要来自措辞（独立撰写，老户型也只有 87.9%），最弱的是**设备题**（75%）、隐含意图 76%、边界 75%。
