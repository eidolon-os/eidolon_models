# Eidolon host-independent streaming ASR

状态：`mac-and-pi5-poc`

## 1. 边界

首版由中文流式 Paraformer 和 final-only CT-Transformer 标点模型组成：

```text
16 kHz mono PCM16
        │
        ▼
eidolon_channel VAD ── START/END_OF_SPEECH、barge-in、EOT
        │
        ▼
eidolon-asr WebSocket ── Paraformer streaming INT8 ONNX
        │
        ├── interim transcript
        └── end_utterance ── CT-Transformer punctuation ── final transcript
```

ASR 服务不再次执行 VAD，也不包含 offline ASR、ITN、LM 或说话人模型。标点只处理 final，
避免 interim 因标点预测变化而反复改写。

## 2. Host 模型

“Host 无关”指同一源码、命令、协议、配置字段、模型校验和测试集，不表示所有 Host 使用
同一二进制执行 provider。

| Host | 当前 backend | 启动命令 | 说明 |
| --- | --- | --- | --- |
| macOS arm64 | `onnx-cpu` | `./scripts/eidolon-asr serve` | 开发与正确性基线 |
| Raspberry Pi 5 arm64 | `onnx-cpu` | 相同 | 边缘 CPU 性能基线 |
| RK3588 arm64 | 暂为 `onnx-cpu` | 相同 | 有 RKNN 制品后，`auto` 才允许选择 NPU |

不根据机器名称散落业务分支。Host 检测只用于诊断和选择已交付 backend；请求与返回格式
始终不变。若显式请求尚不存在的 `rknn` backend，启动必须失败，不能静默回退并伪称 NPU。

## 3. WebSocket contract v1

端点：`GET /v1/stream`

连接后服务发送 `connected`。每段语音由 Channel 发起：

```json
{
  "type": "start",
  "stream_id": "room-id",
  "utterance_id": "turn-id",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le"
}
```

随后发送二进制 PCM frame。Channel VAD 停声时发送：

```json
{"type": "end_utterance"}
```

服务返回可重复的 transcript 事件：

```json
{
  "type": "transcript",
  "stream_id": "room-id",
  "utterance_id": "turn-id",
  "revision": 3,
  "text": "累计识别文本。",
  "raw_text": "累计识别文本",
  "delta": "本次新增文本",
  "is_final": true,
  "language": "zh",
  "provider": "onnx-cpu",
  "model_id": "iic/paraformer-zh-streaming-onnx@2.0.5",
  "punctuation_model_id": "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx@2024-09-25",
  "audio_ms": 1800,
  "decode_ms": 73.2,
  "rtf": 0.041,
  "punctuation_ms": 1.1,
  "total_inference_ms": 74.3,
  "total_rtf": 0.0413
}
```

一个连接可以依次处理多段 utterance。第二段使用 `start_utterance`，每段拥有独立 frontend
和 decoder cache。当前 CPU backend 共享一份 ONNX 权重并串行执行推理，避免边缘 Host 为并发
stream 重复加载约 228 MB 权重。

服务支持多个 WebSocket 和多个 utterance 同时存活，每路 frontend/cache 相互隔离。当前
`onnx-cpu` 为保证 `funasr-onnx` 共享模型对象的状态安全，会把实际推理调用串行化；因此这里的
“支持并发”是正确性并发，不等于无限吞吐。实测 Pi 5 建议最多 4 路实时流，8 路会开始积压。

`text` 是每个 revision 的权威完整文本；final 可能通过标点插入修订已有字符，调用方不能只拼接
`delta`。`raw_text` 保留 Paraformer 原文。`decode_ms` 与 `rtf` 只计算 ASR，便于和原基准直接
比较；`punctuation_ms` 单列，`total_inference_ms` / `total_rtf` 包含两者。

## 4. 健康检查

- `/healthz`：进程事件循环存活；
- `/readyz`：模型已通过 checksum、成功加载且可接受请求；
- `/v1/info`：协议、音频格式、endpoint owner、backend 和模型身份。

服务在模型 checksum 或加载失败时不进入监听状态。运行时不访问 ModelScope，不允许隐式下载。

## 5. 测试矩阵

`./scripts/eidolon-asr test` 在每个 Host 运行同一套测试：

1. Artifact：固定 revision、必需文件和 SHA-256；
2. Host/config：`auto` 的 CPU 基线与未交付 RKNN 的 fail-closed；
3. Protocol：采样率、声道、PCM 格式、消息顺序和错误关闭；
4. Lifecycle：同一 WebSocket 连续多 utterance、interim/final revision；
5. Real model：官方中文 WAV 经真实 ASR 与标点 ONNX 输出有效中文和句末标点；
6. Live service：实际启动进程，通过 WebSocket probe 输入完整音频并收到 final；
7. Raspberry Pi：记录冷启动、峰值 RSS、ASR decode、标点耗时和总 RTF；
8. Concurrency：生成四种确定性音频流，按 1/2/4/8 路实时发送并记录握手、首 interim、
   EOT-final、overhang 和包含排队的 RTF。

当前 PoC gate：

- checksum、单元和协议测试全部通过；
- 官方 5.55 秒中文样本得到非空且包含稳定关键词的识别结果；
- 至少产生一个 interim 和一个 final；
- Mac 与树莓派输出不要求字节完全相同，但核心中文关键词必须一致；
- 树莓派 5 整段 RTF `< 1.0`，能实时处理单路会话；
- 服务重启后不联网下载模型。

### 2026-08-26 实机结果

| Host | 测试 | 结果 |
| --- | --- | --- |
| Mac Apple Silicon | 全套测试 | 19 passed，2.10 s |
| Mac Apple Silicon | 5.55 s 单路 | ASR 346 ms，标点 2.2 ms，文本正确且带句号 |
| Raspberry Pi 5 / 4 cores / aarch64 | 全套测试 | 19 passed，6.16 s |
| Raspberry Pi 5 | 5.55 s 单路 | ASR 836 ms，标点 3.1 ms，文本正确且带句号 |
| Raspberry Pi 5 | 常驻服务 RSS | 无标点约 420 MiB，带标点约 720 MiB |
| Raspberry Pi 5 | HTTP + WebSocket live probe | ready、7 interim、1 带标点 final 均通过 |

测试 final 为“欢迎大家来体验达摩院推出的语音识别模型。”。ASR 与标点权重分别约 227 MiB
和 274 MiB；常驻服务只在启动时加载一次，不会为每个 utterance 重复加载。
完整并发结果见 [BENCHMARK.md](BENCHMARK.md)。

CER、噪声、远场、回声和八小时 soak 需要 Eidolon 自有音频集，不能用一个上游样本冒充产品
验收；本仓库先把这些 case 的入口和指标字段固定下来。

## 6. 服务管理

手工与开发启动完全相同。产品部署时，Mac 由现有 supervisord adapter 调用同一个命令，Pi
由 systemd 调用同一个命令；差异只属于 Host 的进程管理器。Pi unit 模板位于
`deploy/systemd/eidolon-asr.service`。

## 7. RK3588 后续路径

RK3588 到位后先运行 `onnx-cpu` 与树莓派相同测试，建立架构正确性基线。只有完成以下事项后
才增加 `rknn`：

1. 固定转换工具链、量化数据来源、RKNN runtime ABI；
2. 保存 `.rknn`、转换 manifest 和 checksum；
3. 实现同一 `StreamingBackend` 接口；
4. 复跑全部 contract 与中文音频测试；
5. 对比 ONNX/RKNN 文本漂移、RTF、RSS 和温控降频。
