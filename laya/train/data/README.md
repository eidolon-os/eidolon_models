# 训练数据存放

```
train/data/
├── authored/<scenario>/*.jsonl   人工 / 大模型直接撰写的用例（提交入库）。一行一条，格式见下
├── external/<name>/              下载的外部数据集副本（gitignore），manifest 记 sha256 与来源
└── README.md
```

## authored 格式（撰写格式，比记录格式简单；`gen` 的 `authored_cases` adapter 转成记录）

```json
{"home": "apartment", "text": "麻烦把客厅主灯调暗一点",
 "intent": "控制", "device": "客厅主灯", "action": "调低或减小", "tags": ["explicit-control"]}
{"home": "house",     "text": "好像有点冷",
 "intent": "控制", "device": "地暖", "action": ["打开或启动", "调高或增大"], "tags": ["implicit-intent"]}
{"home": "villa",     "text": "我要睡了",
 "intent": "控制", "device": "多个设备或整屋", "action": "关闭或停止", "tags": ["multi-device"]}
{"home": "apartment", "text": "把烤箱预热到两百度",
 "intent": "控制", "device": "没有对应的设备", "action": "设为指定的数值或模式", "tags": ["device-not-in-home"]}
{"home": "house",     "text": "洗碗机现在洗完了吗", "intent": "查询", "device": "洗碗机", "tags": ["status-query"]}
{"home": "apartment", "text": "我们家那台扫地机器人上周又卡沙发底下了", "intent": "无关", "tags": ["non-command"]}
```

- `home`：`apartment` / `house` / `villa`（villa = 前两者设备之和），设备名必须是该户型清单里的名字（见 `evals/smart-home/homes/`）。
- `device`：一个名字、多个可接受名字的列表、或两个出口 `多个设备或整屋` / `没有对应的设备`；`无关` 不填。
- `action`：`打开或启动 / 关闭或停止 / 调高或增大 / 调低或减小 / 设为指定的数值或模式 / 暂停 / 上锁`，可列表；`查询` / `无关` 不填。
- `tags` 第一个是切片名，评测按它分组看成绩。
- **不得抄 `evals/smart-home/scenarios/*/cases.jsonl` 里的句子**（锁定评测集），`assemble` 也会按句子哈希再拦一次。

校验：`uv run --extra torch --extra train eidolon-laya-train gen --scenario train/scenarios/smart-home --config train/scenarios/smart-home/gen-authored.yaml --out /dev/null`，
任何一条金标不是合法选项都会报错并指出行号。

## 当前数据（2026-09-26）

### authored/smart-home：1989 条（去重前；跨文件重复的句子 gen 时跳过）

撰写者：Claude（当前最强的可用模型）按切片 × 户型系统撰写，金标随句子一起给，经 `authored_cases` adapter 逐条校验
（设备名是该户型的合法选项、动作合法、意图与字段一致），并已剔除与 182 条锁定评测集相同的句子。

| 切片 | 条数 |
|---|---|
| `non-command` | 502 |
| `explicit-control` | 279 |
| `boundary` | 276 |
| `implicit-intent` | 248 |
| `multi-device` | 165 |
| `device-not-in-home` | 157 |
| `asr-noise` | 150 |
| `status-query` | 101 |
| `room-disambiguation` | 75 |
| `elliptical` | 21 |
| `long-form` | 15 |

意图：控制 1201 / 查询 138 / 无关 650；
出口：多个设备或整屋 191 / 没有对应的设备 184；
户型：apartment 869 / house 697 / villa 423。

文件：

| 文件 | 内容 |
|---|---|
| `explicit-and-rooms.jsonl` | 明确指令（36 台设备每台 ≥ 3 句）+ 分房间消歧 |
| `noncommand-and-query.jsonl` | 非命令负例（8 个子类）+ 状态查询（每台 2 句） |
| `implicit-multi-missing.jsonl` | 隐含意图 + 多设备 / 场景 + 家里没有 |
| `asr-and-boundary.jsonl` | 语音识别错字 + 边界难例（最小对、否定、定时、转述） |
| `misc-forms.jsonl` | 省略句、长句、villa 专属 |
| `b2-noncommand.jsonl` | 第二批：非命令 300（9 个子类，"像命令但不是"） |
| `b2-implicit-boundary.jsonl` | 第二批：隐含意图 150 + 边界 150 |
| `b2-control-multi-missing-asr.jsonl` | 第二批：明确指令 100 + 多设备 80 + 家里没有 70 + 错字 50 |

### external：三个公开数据集（回放用，副本 gitignore，`manifest.json` 记来源和 sha256）

| 目录 | 来源 | 许可 | 转成记录 | 说明 |
|---|---|---|---|---|
| `zhihao_smarthome/` | [zhihao666/smartHome](https://huggingface.co/datasets/zhihao666/smartHome) | Apache-2.0 | 5,500 | 每条自带家庭设备清单；同名设备按房间区分，房间没说清时多个都算对（784 条） |
| `scenic_repo/` | [huluk98/SCENIC](https://github.com/huluk98/SCENIC) | MIT | 9,772 | 从确认回复反解房间 / 设备 / 动作；固定 71 台设备的"SCENIC 户型"，每条随机取金标 + 11 台；多动作 → 多个设备或整屋；2,049 条没有动作标签 |
| `massive_zh/` | [AmazonScience/massive](https://huggingface.co/datasets/AmazonScience/massive) zh-CN | CC-BY-4.0 | 5,080 | iot 880 条当控制正例（apartment 户型）；其余场景每个抽 300 当"无关"负例；**audio / play / music 排除**（家里有智能音箱，那是真命令） |

下载：`huggingface_hub.snapshot_download` 或 `git clone`，放到对应目录（国内可设 `HF_ENDPOINT=https://hf-mirror.com`）。
转换：`gen --config train/scenarios/smart-home/gen-external-<name>.yaml`（每个来源一份配置，assemble 分别给权重）。
SCENIC 的 `iot_instruction_benchmark_200.json` 是它自己的评测集，不要拿来训练。

### 怎么组合：分开存放，assemble 时按权重混合

见 `train/scenarios/smart-home/assemble.yaml`：authored 1.0（主体）、template 0.3（只补覆盖）、
外部同域回放约 40%（zhihao 0.15 / scenic 0.1 / massive 0.2）。分开存放是为了能单独调权重、审计、下线某个来源，
门槛退步时能定位是哪个来源造成的；混合只发生在 assemble 这一步。
