# laya 决策模型服务

[laya](https://github.com/NandhaKishorM/laya)（开源的 Jev 仿品，encoder 型、非自回归）的
`laya-multilingual` checkpoint，包成一个 HTTP 服务，可在 Mac 本地或云服务器上启动。

- **两个后端，配置切换**：`torch`（PyTorch 权重，Mac 上可用 MPS）/ `onnx`（ONNX Runtime，
  不需要 torch）。两者共用同一份前后处理（`sequence.py`），所以切换后端不改变语义：
  torch 后端与上游 laya（仓库内副本，当前 v0.3.20，见 `src/eidolon_models_laya/vendor/laya/VENDOR_VERSION`）逐字节一致，onnx 与 torch 的 logit 差 ~2e-5（测试守护）。
- **接口兼容 Jev / laya**：`POST /v1/systemone`，请求体 `{state, questions}` 不变，
  回复在原有 `answers`/`usage` 之外多带 `truncated`、`backend`、`timing_ms`。
- **注意：这是 zero-shot 底座**，中文对话路由必须微调后才可用；这里先把服务跑起来，不耽误联调。

## 目录

```text
laya/
├── pyproject.toml / uv.lock      独立 uv 项目；extras: torch / onnx / export
├── deploy/services.toml          每个 Laya 服务的全部设置（家居、参与决策、服务器的 systemone）
├── scripts/eidolon-laya          启动器：本地、服务器同一个
├── src/eidolon_models_laya/
│   ├── config.py                 services.toml 的一张服务表 → Settings（后端随 Host 的 rknpu2）
│   ├── artifacts.py              manifest、按 revision 拉取、sha256 校验
│   ├── sequence.py               不依赖 torch 的拼序列/解码（移植自 laya，随 vendor 副本升级）
│   ├── backends.py               TorchBackend / OnnxBackend：只做前向
│   ├── engine.py                 校验 → 拼序列 → 前向 → 解码
│   ├── export.py                 PyTorch → ONNX，并与 PyTorch 对数后写 export.json
│   ├── export_npu.py             PyTorch → 按长度分档的静态形状 ONNX + CPU 侧表（给 NPU 转换器），逐档对数
│   ├── service.py                aiohttp HTTP API
│   └── cli.py                    fetch | export-onnx | export-npu | doctor | serve | predict
├── models/laya-multilingual/1c5edc17/
│   ├── manifest.json             提交：来源、revision、每个文件的 sha256
│   ├── torch/                    fetch 得到（gitignore）
│   ├── onnx/                     export-onnx 生成（gitignore）：model.onnx + model.onnx.data + export.json
│   └── npu/                      export-npu 生成（gitignore）：hidden_l<L>.onnx + tok_emb_fp16.npy + type_emb.npy + scorer.npz
├── examples/xiyouji-addressee.json
├── deploy/                       systemd unit + 部署到 ECS 的脚本；rk3588/laya_npu.py（板上：转 RKNN、在 NPU 上跑评测输入）
└── tests/                        单测（无模型）+ 与上游对拍（需模型）
```

## 本地（Mac）

```bash
cd laya
uv sync --all-extras                 # torch + onnx + 导出工具
git lfs pull                         # 产品模型（家居 c10、参与决策 p4）的权重是 LFS 对象
scripts/eidolon-laya --service smart_home doctor
scripts/eidolon-laya --service smart_home serve    # 127.0.0.1:8771，torch（MPS），本机无需 key
```

每个服务的设置都在 `deploy/services.toml`（一张服务表，下面按后端分表）；没有环境变量。后端随 Host：
声明了 `rknpu2`（`EIDOLON_HOST_CAPABILITIES`，Ops 写入）用 NPU，否则用 torch。单次覆盖用命令行：
`--backend`、`--port`、`--model-dir`、`--max-len` 等。工具命令针对一个模型目录：
`scripts/eidolon-laya --model-dir models/laya-multilingual/1c5edc17 fetch`（644 MB，按 manifest 的 revision
校验 sha256）、`... export-onnx`、`... export-npu`；不起服务只跑一次：
`scripts/eidolon-laya --model-dir <模型目录> predict --file examples/xiyouji-addressee.json`。

国内网络拉权重：`scripts/eidolon-laya fetch --endpoint https://hf-mirror.com`
（走镜像时自动关闭 Xet，否则 Xet 会直连官方 CAS 返回 401）。

## 按问题分配上下文（2026-10-08）

`POST /v1/systemone` 的每个 question 可选 `state`，覆盖本次请求的公共 state；未提供时行为不变。
一次请求仍按原有 engine 批处理/并行机制运行，token 上限、截断记录按问题独立计算。
回复 `features.question_state=true` 表示支持该能力。客户端不得假设旧服务已支持；新版 Agent
在能力缺失或输入截断时交给 LLM，不执行基于不完整上下文的提案。

用途：当前句 intent/device/action 保持原输入，上下文消歧题才读取有限历史；避免历史中的动作
污染当前句分类。没有训练、修改权重或变更部署固定版本。
实际 Mac c4/c10 回放、失败探索与复跑方法见 [模型优先验证报告](evals/model-first-20261008/README.md)。

## 调用

```bash
curl -s localhost:8771/v1/systemone -H 'content-type: application/json' \
  -d @examples/xiyouji-addressee.json
```

对外监听时带上 key（服务表的 `api_key_file`）：`-H "Authorization: Bearer $(cat <api_key_file>)"`。

Python（`examples/`，只用标准库，`LAYA_URL` 指定服务地址，默认 ECS）：

- `demo.py`：**单文件**，客户端 + 客服路由合在一起，拷到任何 Python 3.8+ 上直接跑：
  `python3 demo.py`、`python3 demo.py "我的账号被锁了"`、`--raw` 打印原始回答、`--url` 换服务

- `laya_client.py`：可直接拷走的 `LayaClient`（503 自动按 `Retry-After` 重试、`LayaError`、
  可选 key、默认绕过本机代理）
- `python_client.py`：六种请求形状（听话人、多人点名、下一步、打分、打断意图、长历史截断）
- `customer_service_router.py`：智能客服工单路由——一次问部门/紧急度/退款/流失/人工 5 个问题，
  再由阈值规则决定队列、优先级、自动派单还是转人工分诊

| 路径 | 鉴权 | 说明 |
|---|---|---|
| `GET /healthz` | 否 | 存活 |
| `GET /readyz` | 否 | 模型已加载 |
| `GET /v1/info` | 是 | 模型、revision、后端、线程、限额 |
| `POST /v1/systemone` | 是 | `{"state": 任意 JSON, "questions": {...}, "options": {"truncate_left": false}}` |

错误统一为 `{"error": {"code", "message"}}`：400 问题定义/请求体不合法，401 key 不对，
413 请求体超限，503 `busy`（排队已满，带 `Retry-After`；不会无限排队）。

问题类型：`choice`（单选，`criteria` 为 `{标签: 描述}` 或标签列表）、`score`（有序等级列表）、
`noul`（是/否概率）。每个问题都会把 `state` 完整编码一次，问题越多越慢。

**截断**：每个问题的 token 预算默认是 checkpoint 训练长度 1024（服务表的 `max_len` 或 `--max-len` 可调，
最高 8192，但 CPU 上成本随长度平方增长）。超长的 `state` 默认**丢掉末尾**——把当前这句话放在
`state` 最前面，或传 `"options": {"truncate_left": true}` 保留末尾；被截断的问题会列在回复的
`truncated` 里。

## 服务器（ECS）

```bash
deploy/ecs/deploy.sh root@eidolon          # 同步代码、装依赖、拉权重、导出 ONNX、装 systemd 并启动
```

脚本幂等，可重复执行（首次约 3 分钟，之后主要是同步代码）。服务器上：

- 代码 `/opt/eidolon-laya`，Python 装在 `/opt/eidolon-laya/.python`（服务用户读得到）
- 配置 `/etc/eidolon-laya/services.toml`（`[systemone]` 一张表，格式同 `deploy/services.toml`），
  key 在 `/etc/eidolon-laya/api-key`（640，服务用户可读）；首次部署生成，之后不覆盖。旧的
  `eidolon-laya.env` 在下次部署时迁移一次（保留后端、地址和 key）
- 服务 `systemctl status eidolon-laya`，日志 `journalctl -u eidolon-laya -f`，以
  `eidolon-laya` 系统用户运行，`MemoryMax=4G`
- 取 key：`ssh root@eidolon "cat /etc/eidolon-laya/api-key"`
- 切后端：把 `services.toml` 的 `[systemone.<后端>]` 改成另一个后端后 `systemctl restart eidolon-laya`

**对外访问**：`http://8.141.101.214:8771`，服务监听 `0.0.0.0:8771`，需在阿里云安全组放行入方向
TCP 8771（服务器本机没有防火墙）。注意这是明文 HTTP，bearer key 在网络上可见；只给自己用时
安全组源地址建议填自己的出口 IP。放行前可用隧道：`ssh -N -L 18771:127.0.0.1:8771 root@eidolon`。

可选：走宿主机 nginx 的 443（HTTPS，复用 `*.yangtzeailab.com` 通配证书，需要该子域名的 DNS
A 记录）——`EIDOLON_LAYA_NGINX_SERVER_NAME=eidolon-laya.yangtzeailab.com deploy/ecs/deploy.sh`。
脚本装 `conf.d/<name>.conf`（模板 `deploy/nginx/eidolon-laya.conf`），先 `nginx -t`，失败自动回滚。
本机配了 HTTP 代理时，用 `curl --resolve` 验证要加 `--noproxy '*'`。

依赖默认从阿里云 PyPI 镜像装（仍按 `uv.lock` 的版本和哈希；镜像缺的版本回退 pypi.org），
`EIDOLON_LAYA_PYPI_MIRROR=` 置空则用 `uv sync` 直连。

ECS（1 物理核 / 7.5 GB，与其它服务共用）实测，示例请求（2 个问题，337 token）：

| 后端 | 单次延迟 | 常驻内存 |
|---|---|---|
| torch，1 线程（默认） | 1.15–1.35 s | 1.96 GB |
| onnx，1 线程 | 1.45–1.64 s | 0.93 GB |
| 对照：Mac M3 Pro，torch MPS | 38 ms | — |

**别在这台服务器上跑 `scripts/eidolon-laya test` 的模型对拍**：服务已占 2 GB，对拍再加载两份模型
会触发 OOM。在服务器上用 `scripts/eidolon-laya test -m "not model"`，对拍在本机跑。

## IP 角色团 participation v2（显式启用）

`/v1/systemone` 是通用 Laya 接口，不能作为 Agent 的 `participation.url`。角色团使用
`POST /v1/participation/decide`，请求和响应以 `eidolon_sdk.biz.participation` 为唯一契约。
家居工件没有角色团 profile，家居服务上该端点返回 503；`/readyz` 仍只表示通用模型可用。

训练产物通过准入后，模型目录中的 `manifest.json` 须为 `task_profiles.ip_team.participation`
固定 `participation.json` 的路径与 SHA-256。profile 必须声明任务、`ip-team-v3` 输入格式、
同一模型 revision、校准策略版本、置信度阈值和最多 6 个候选；将这两份文件都列入 Ops
工件的逐文件 digest。服务表写了 `participation = true`（`deploy/services.toml` 的 `[participation]`）
时，启动才会校验该 profile 并开放端点。Ops 对本地角色团路由检查 `/participation/readyz` 的任务与版本；
校验失败会使发布就绪门槛失败。

`participation.json` 的固定字段示例（实际阈值由独立评估校准，不使用此示例值上线）：

```json
{"schema_version":1,"task":"ip_team.participation","state_format":"ip-team-v3","model_revision":"<manifest revision>","policy_version":"<calibration version>","min_confidence":0.8,"max_candidates":6}
```

当前 `ip-team-v3` 模型输出只有动作和候选槽位。运行时按请求中的动态候选 ID 映射，
不通过名称猜成员；任何输入或选项截断、未知动作、低置信度均弃权。分类头不能生成具体
澄清问题，所以 `clarify` 目前弃权；不能靠固定台词补出 `instruction`。训练任务若调整
输入格式或提供经验证的澄清任务输出，需更新版本化 profile 和对应适配器测试后再发布。

## 测试

```bash
scripts/eidolon-laya test            # 单测 + 对拍；没有权重/ONNX 时对拍测试自动跳过
```

## 评测

- [`evals/smart-home/`](evals/smart-home/)：智能家居控制决策（是不是命令、控制哪台设备、什么动作），
  15 类设备、9 个场景、182 条用例，Mac 上各后端的准确率与延迟见 [REPORT.md](evals/smart-home/REPORT.md)。

## 许可

checkpoint：Apache-2.0（convaiinnovations/laya）；基座编码器 mmBERT-base：MIT；
`sequence.py` 移植自 laya 源码（Apache-2.0，文件头注明改动）；`src/eidolon_models_laya/vendor/laya/` 是上游包按 git tag 的原样副本（上游会从 PyPI 删旧版本，所以不依赖 PyPI），用 `scripts/sync-laya-vendor.py <tag>` 升级，升级门槛是 `tests/test_parity.py` 和 `evals/smart-home/run_all.sh` 的数字不动。本目录其余代码随仓库许可。

2026-10-09：c10 已完成 [OPI5 Max NPU 隔离验证](evals/c10-npu-20261009/README.md)。精度与接纳决定通过，长上下文仍有偶发超过 1 秒；尚未切换生产 c4。
