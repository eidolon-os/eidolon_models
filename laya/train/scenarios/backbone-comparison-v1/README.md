# 四模型比较的复现入口

- 设计及检查后的容量口径：[DESIGN.md](DESIGN.md)
- 版本、来源和源码哈希：[PROVENANCE.json](PROVENANCE.json)
- 逐组统计与成对比较：[comparison.json](comparison.json)
- 人类可读表格：[TABLE.md](TABLE.md)
- 结论：[RESULTS.md](RESULTS.md)

本目录脚本从仓库根目录运行；模型、官方运行时代码、完整概率及日志存放在 `laya/train/runs/backbone-comparison-v1/`，不提交权重。

```sh
PYTHONPATH=../eidolon_sdk HF_HUB_OFFLINE=1 laya/.venv/bin/python laya/train/scenarios/backbone-comparison-v1/compare.py preflight
PYTHONPATH=../eidolon_sdk HF_HUB_OFFLINE=1 laya/.venv/bin/python laya/train/scenarios/backbone-comparison-v1/compare.py laya
PYTHONPATH=../eidolon_sdk HF_HUB_OFFLINE=1 laya/.venv/bin/python laya/train/scenarios/backbone-comparison-v1/compare.py openjev
PYTHONPATH=../eidolon_sdk HF_HUB_OFFLINE=1 laya/.venv/bin/python laya/train/scenarios/backbone-comparison-v1/compare.py decider
PYTHONPATH=../eidolon_sdk HF_HUB_OFFLINE=1 laya/.venv/bin/python laya/train/scenarios/backbone-comparison-v1/compare.py jevk5
PYTHONPATH=../eidolon_sdk HF_HUB_OFFLINE=1 laya/.venv/bin/python laya/train/scenarios/backbone-comparison-v1/report.py
```

推理脚本要求可访问 MPS，拒绝覆盖已有结果；要重跑应另建实验目录并明确记录改动。r6/r7 分别使用 `laya-r6`、`laya-r7` 参数。模型必须先按 PROVENANCE 的固定版本下载；离线标志保证评分时不访问 Hub。

`fetch_weights.py` 是本次下载使用的分段续传辅助脚本，使用 Hub LFS SHA256 验证最终文件；官方运行时代码未经修改。JevK5 的加载依赖 accelerate 1.15.0 / psutil 7.2.2 放在本次运行目录的 runtime-deps，未修改原有 venv 和锁文件。

`check_backend.py` 对开发集固定位置 0、42 的两个输入进行 CPU float32 对照，核验 MPS float16 数值路径；它不是额外选型集，也不拟合参数。

统计中的 `existing_threshold_0_8_exploratory` 是沿用旧策略阈值的补充分析：把低于 0.8 的 argmax 映射到弃权。不是搜索最佳阈值，也不等同于新候选的完整 SDK 回放或校准。主要结果始终为官方温度下的 argmax。
