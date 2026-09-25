# 智能家居控制决策评测

测 laya 在智能家居语音 / 文字指令上的两件事：**是不是控制命令**，**控制哪台设备**（外加做什么动作），
以及在 Mac 上各后端的延迟与内存。结果与结论见 [REPORT.md](REPORT.md)；
与社区微调模型（MacJev、laya-cn-a）的对比与后续方案见 [COMPARISON.md](COMPARISON.md)。

```text
smart-home/
├── taxonomy.md            设备分类（15 类）与来源
├── homes/                 户型：apartment、house 各 18 台设备；villa = 两者合并 36 台
├── scenarios/<NN-名字>/    每个场景一个目录：cases.jsonl + README.md
│   ├── 01-explicit-control      明确指令，15 类逐一覆盖
│   ├── 02-implicit-intent       隐含意图（“好热啊”）
│   ├── 03-room-disambiguation   同类设备分房间
│   ├── 04-status-query          状态查询
│   ├── 05-non-command           提到设备但不是指令
│   ├── 06-multi-device-scene    多设备 / 场景
│   ├── 07-asr-noise             语音识别错字
│   ├── 08-large-inventory       38 个选项（语句同 01/03）
│   └── 09-device-not-in-home    家里没有该设备
├── run_eval.py            调 /v1/systemone 跑一遍（含规则基线），写 results/<label>/
├── make_report.py         汇总 results/ → REPORT.md（开头放 conclusions.md）
├── compare_models.py      多个 checkpoint 对比 → COMPARISON.md（开头放 comparison_conclusions.md）
├── run_all.sh             起服务并评测：默认 mac-torch-mps；也可指定 mac-torch-cpu / mac-onnx-cpu(-t6)
├── conclusions.md         人工撰写的结论
└── results/               每次运行的 predictions.jsonl 与 summary.json
```

## 运行

```bash
cd laya
uv sync --all-extras && scripts/eidolon-laya fetch && scripts/eidolon-laya export-onnx   # 首次
evals/smart-home/run_all.sh                  # 默认 mac-torch-mps，约 25 秒，最后生成 REPORT.md
evals/smart-home/run_all.sh mac-onnx-cpu     # 需要时再测 CPU 后端（报告里保留了 2026-09-25 的 CPU 数据）
```

对已在运行的服务（例如 ECS）评测：`python3 evals/smart-home/run_eval.py --url http://8.141.101.214:8771 --label ecs-torch-cpu`，
再 `python3 evals/smart-home/make_report.py`。两个脚本只依赖 Python 标准库。

## 用例格式

```json
{"id": "02-01", "home": "apartment", "text": "好热啊",
 "gold": {"intent": "控制", "device": ["客厅空调", "主卧空调"], "action": ["打开或启动", "调低或减小"]}}
```

`device` / `action` 可以是可接受答案列表；没有 `device` 的用例不对设备计分，没有 `action` 的不对动作计分。
意图标签：控制 / 查询 / 无关；设备标签：该户设备名 + “多个设备或整屋” + “没有对应的设备”。

这套用例是**评测集**：做微调时不要放进训练数据。
