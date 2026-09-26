# 训练数据存放

```
train/data/
├── authored/<scenario>/*.jsonl   人工 / 大模型直接撰写的用例（提交入库）。一行一条，格式见下
├── homes/<scenario>/*.json       只用于训练的户型（设备清单）
├── external/<name>/              下载的外部数据集副本（gitignore），manifest 记 sha256 与来源
├── qa/                           双人盲标抽检：blind-<批>.jsonl、report-<批>.json、disagreements-<批>.jsonl
├── LABELING.md                   标注规范（训练和评测数据共用）
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

## 当前数据（2026-09-26，r12）

### authored/smart-home：8,255 条（去重前；同户型同一句跨文件重复时 gen 保留先出现的）

撰写者：Claude 按 `LABELING.md` 撰写，金标随句子一起给，经 `authored_cases` adapter 逐条校验（设备名是该户型的合法选项、动作合法、意图与字段一致）；
每批写完跑 `evals/smart-home-v2/purge_training_overlap.py`，与两套锁定集的近似重复为 0；r9 起每批抽 10% 双人盲标（`qa/`）。

| 切片（首个切片标签） | 条数 |
|---|---|
| `implicit-intent` | 1,932 |
| `non-command` | 1,653 |
| `explicit-control` | 1,283 |
| `status-query` | 838 |
| `boundary` | 748 |
| `room-disambiguation` | 558 |
| `device-not-in-home` | 537 |
| `multi-device` | 442 |
| `asr-noise` | 140 |
| `action-granularity` | 88 |
| `elliptical` | 21 |
| `long-form` | 15 |

意图：控制 5,347 / 查询 843 / 无关 2,065；
出口：多个设备或整屋 500 / 没有对应的设备 741；
户型：v1 的三个户型 4,962（60%），15 个训练专用户型 3,293。

训练专用户型在 `homes/smart-home/`（评测专用户型 loft / threegen / smallflat / bigvilla 永不用于训练）：

| 户型 | 设备 | 条数 | 用途 |
|---|---|---|---|
| brand | 18 | 281 | 设备按 app 里的默认名（品牌 / 型号）命名的两居 |
| countryside | 15 | 221 | 农村自建房 |
| duplex | 22 | 316 | 复式两层 |
| elder | 15 | 271 | 老人住的两居 |
| family3 | 20 | 316 | 三室两厅带儿童房 |
| kids | 17 | 261 | 有婴儿的两居 |
| mega | 37 | 267 | 大平层 |
| minimal | 5 | 133 | 极简一居 |
| northflat | 19 | 95 | 北方老小区两居 |
| office | 12 | 177 | 居家办公的一居 |
| pets | 15 | 242 | 养一猫一狗的两居 |
| rental | 12 | 246 | 合租房 |
| southyard | 22 | 108 | 南方带院子的平层（老两口） |
| studio | 10 | 209 | 单身公寓 |
| townhouse | 33 | 150 | 联排别墅 |

文件：

| 文件 | 条数 | 内容 |
|---|---|---|
| `asr-and-boundary.jsonl` | 217 | 语音识别错字 + 边界难例（最小对、否定、定时、转述） |
| `b2-control-multi-missing-asr.jsonl` | 294 | 第二批：明确指令 100 + 多设备 80 + 家里没有 70 + 错字 50 |
| `b2-implicit-boundary.jsonl` | 293 | 第二批：隐含意图 150 + 边界 150 |
| `b2-noncommand.jsonl` | 300 | 第二批：非命令 300（9 个子类，"像命令但不是"） |
| `b3-boundary-hard.jsonl` | 299 | 第三批：边界难例 |
| `b3-multi-implicit.jsonl` | 300 | 第三批：多设备 150 + 隐含意图 150 |
| `b3-new-homes.jsonl` | 330 | 第三批：训练户型上的各切片 |
| `b3-query-rooms-actions.jsonl` | 298 | 第三批：查询 + 分房间 + 动作粒度 |
| `b4-language-variety.jsonl` | 300 | 第四批：措辞变化（口语、倒装、方言词） |
| `b4-more-on-b3-homes.jsonl` | 292 | 第四批：训练户型补量 |
| `b4-new-homes.jsonl` | 355 | 第四批：更多训练户型上的各切片 |
| `b4-query.jsonl` | 300 | 第四批：状态查询 300 |
| `b5-implicit-room.jsonl` | 237 | 第五批：隐含意图 + 房间词 |
| `b5-similar-name-absent.jsonl` | 236 | 第五批：名字相近但没有 / 名字相近且存在的对照 |
| `b6-outside-keep.jsonl` | 400 | r9：家外面的感受、保持现状（无关）+ 家里同类感受、让运行中的设备停（控制） |
| `b6-to-others.jsonl` | 360 | r9：对家人 / 宠物 / 孩子说话（无关）+ 对助手说的对照 |
| `b7-alias-reference.jsonl` | 350 | r10：用别名 / 功能 / 位置指代存在的设备 |
| `b7-function-match.jsonl` | 200 | r10：功能对应（脚冷 → 地暖，不是冰箱） |
| `b7-qualifier-negation.jsonl` | 350 | r10：房间 / 楼层限定词、一句两台其中一台被否定或保留 |
| `b8-addressee-quote.jsonl` | 360 | r11：说话对象不是助手、引述 / 转述 / 朗读 + 真请求对照 |
| `b8-mood-meta.jsonl` | 360 | r11：非请求语气、谈论助手、自己去做、知识问题 + 真请求对照 |
| `b9-need-comfort.jsonl` | 338 | r12：隐含意图——温度（含暖气阀 / 电暖器 / 酒柜）、光线（刺眼 → 窗帘、起夜 → 夜灯）、睡眠噪音 |
| `b9-need-condition.jsonl` | 340 | r12：隐含意图——空气、湿度、衣物（湿 → 干衣机，不是除湿机）、清洁 |
| `b9-need-life.jsonl` | 316 | r12：隐含意图——植物 / 鱼池 / 宠物、将要做的事、安全与生活事件 |
| `explicit-and-rooms.jsonl` | 227 | 明确指令（36 台设备每台 ≥ 3 句）+ 分房间消歧 |
| `implicit-multi-missing.jsonl` | 239 | 隐含意图 + 多设备 / 场景 + 家里没有 |
| `misc-forms.jsonl` | 113 | 省略句、长句、villa 专属 |
| `noncommand-and-query.jsonl` | 251 | 非命令负例（8 个子类）+ 状态查询（每台 2 句） |

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
