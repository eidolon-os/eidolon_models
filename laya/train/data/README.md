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
