# JevK5 快速参与决策

独立的 `eidolon-models-jevk5` uv 项目，服务智能陪伴和 IP 角色团的参与决策：由谁回应、等待、结束、或需要澄清。最终回复交给其他大模型生成。本项目的首轮流程是实验性本地推理与适配训练，尚未提供生产 HTTP 服务、Host 接入或端侧速度保证。

## 边界

```text
jevk5/
├── pyproject.toml / uv.lock              独立依赖环境
├── src/eidolon_models_jevk5/
│   ├── engine.py / cli.py               原生选项概率推理、适配器加载
│   ├── generate.py / prepare.py         有预算的合成、盲审、显式审核、冻结切分
│   ├── train.py / evaluate.py           有界训练、分项开发评估、候选门槛
│   └── vendor/jevk5/                    固定版本上游代码及许可证
├── models/jevk5/c4f7fdb3/manifest.json   模型来源与文件摘要
├── experiments/companion-ip-v1/         训练前协议、审核与结果
├── runs/                               本地适配器、原始概率和日志（忽略）
└── private/                            教师请求/响应账本（忽略）
evals/participation/                     仓库级共享评测协议和数据
```

推理、训练不导入 Laya Python 包，不使用 Laya 虚拟环境。首轮数据准备通过清单只读历史 Laya 路径下的合成 train/val 文件，保留来源哈希；这不是运行时依赖。历史实验不移动，避免破坏原有路径和复现记录。

## 安装与本地推理

以下命令从仓库根目录运行：

```bash
uv sync --locked --project jevk5
uv run --locked --project jevk5 eidolon-jevk5 \
  --model jevk5/models/jevk5/c4f7fdb3/weights \
  --file /absolute/path/to/decision-record.json
```

输入为 `state` 和 `questions.move` 的 choice 记录，可附带离线 labels/meta，但它们不进入模型。单题2至16选项、最多4096 token，超限报错，不静默截断。输出各选项概率。适配后传 `--adapter /absolute/path/to/epoch-N`；加载时校验基座权重摘要。默认 MPS，可显式选择 CPU。上游温度仅为固定读出参数，不能把输出概率直接视为已校准的业务可靠性。

权重未提交Git；本机从已经校验的历史缓存建立硬链接，按不可变资产使用，不覆盖写入。换机器需要按 `models/jevk5/c4f7fdb3/manifest.json` 取得相同文件并验证摘要。适配器单独保存，不合并覆盖基座。上游代码保留 Apache-2.0 许可证；`vendor/jevk5/UPSTREAM.json` 记录提交、原始哈希和本地修改。

## 数据生成、审核与训练

首轮规范见 [DESIGN.md](experiments/companion-ip-v1/DESIGN.md)。生成器只发参与规范和合成文本，API密钥仅作官方端点认证；无内置源码或真实聊天上传逻辑。调用账本在请求前落盘，失败/超时也计入24次上限，不自动重试含糊状态。当前实验已经执行过，重复命令只允许复用已完成结果，不用于开启新一轮额度。

```bash
uv run --locked --project jevk5 python -m eidolon_models_jevk5.generate \
  --out jevk5/private/companion-ip-v1 --preview

# 实际调用需要环境变量 EIDOLON_IP_DATA_API_KEY。
uv run --locked --project jevk5 python -m eidolon_models_jevk5.generate \
  --out jevk5/private/companion-ip-v1
```

生成结果必须逐条审阅并写入 `audit.json` 的 accept/correct/reject 及理由。明确修正保留原标签，语义不稳健的记录剔除。结构修复脚本属于首轮事件记录，不是通用自动纠错。新数据的本地编写诊断场景另存共享 `evals/participation/authored/`，按来源单独统计；不是独立人工验收。

```bash
uv run --locked --project jevk5 python -m eidolon_models_jevk5.prepare \
  --repo . --generated jevk5/private/companion-ip-v1 \
  --audit jevk5/experiments/companion-ip-v1/audit.json \
  --authored evals/participation/authored/companion-ip-v1.json \
  --out evals/participation/data/companion-ip-v1

HF_HUB_OFFLINE=1 uv run --locked --project jevk5 python -m eidolon_models_jevk5.train \
  --model jevk5/models/jevk5/c4f7fdb3/weights \
  --data evals/participation/data/companion-ip-v1 \
  --out jevk5/runs/companion-ip-v1-seed71
```

冻结数据和运行目录拒绝覆盖。首轮训练至多2个epoch，每个epoch每快照只取一个候选变体，按任务/家族加权。原始模型和每个epoch均评估同一DEV；严重正常对话回归会提前停止，所有门槛未过则 `selected_epoch=null`。无自动发布、阈值搜索或测试集推理。训练结果与速度数据见本轮 `RESULTS.md`（运行完成后生成）。

```bash
uv run --locked --project jevk5 python -m pytest -q jevk5/tests
```
