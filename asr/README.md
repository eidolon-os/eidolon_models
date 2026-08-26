# ASR models

Eidolon OS 私有化语音识别模型与统一推理服务。

## 当前选型

默认部署三个互补模型：

- `paraformer-zh-streaming/2.0.5`：中文流式 Paraformer INT8 ONNX。
- `paraformer-zh-offline/2.0.5`：中文离线 Paraformer INT8 ONNX，EOT 后重新识别整句。
- `punc-ct-transformer-zh-cn/2024-09-25`：中文 CT-Transformer 量化 ONNX，final-only 标点恢复。

仍刻意不包含：

- VAD：由 `eidolon_channel` 统一负责起声、停声、抢话和 EOT；
- ITN、LM、speaker model：待自有测试证明需要后再增加。

interim 保持流式 ASR 原文；`end_utterance` 后先由 offline 模型对完整音频做 second pass，再对
权威 final 恢复标点。服务同时返回 `streaming_text`、`raw_text`、`final_revised`、
`offline_decode_ms` 和 `punctuation_ms`，便于 Channel 诊断修订与时延。offline 与标点默认启用；
容量或 A/B 对照可分别关闭：

```bash
./scripts/eidolon-asr --no-offline serve
./scripts/eidolon-asr --no-punctuation serve
```

## 统一启动

Mac Apple Silicon、树莓派 5 和普通 Linux arm64 使用完全相同的命令：

```bash
./scripts/eidolon-asr doctor
./scripts/eidolon-asr verify
./scripts/eidolon-asr serve
```

默认监听 `127.0.0.1:8767`：

```bash
curl http://127.0.0.1:8767/healthz
curl http://127.0.0.1:8767/readyz
curl http://127.0.0.1:8767/v1/info
```

真实模型推理和在线服务探测：

```bash
./scripts/eidolon-asr infer tests/data/asr_example_zh.wav
./scripts/eidolon-asr probe tests/data/asr_example_zh.wav
```

并发实时流基准：

```bash
./scripts/eidolon-asr bench tests/data/asr_example_zh.wav --concurrency 1,2,4,8
```

## 容量与排队

默认容量是 **64 个 WebSocket 连接、2 个实时 utterance、6 个 FIFO 排队
utterance**。连接本身不预占推理槽；收到 `start` 后才申请槽位。排队期间服务在内存中暂存
PCM16，轮到后补跑已收到的音频，不切换或降级到其他 ASR。

- 最大 utterance：60 秒；
- 最大排队等待：10 秒；
- 2 个实时槽和 6 个排队位都满时，返回可重试的 `capacity_exceeded`；
- 等待超过 10 秒时，返回可重试的 `capacity_timeout`；
- 第 65 个 WebSocket 在升级前收到 HTTP 503；
- 客户端断开时自动取消排队或释放槽位。

默认值可通过同名启动参数或环境变量调整：

```bash
./scripts/eidolon-asr \
  --max-connections 64 \
  --realtime-slots 2 \
  --max-queued-utterances 6 \
  --max-utterance-seconds 60 \
  --max-queue-wait-seconds 10 \
  serve
```

对应环境变量为 `EIDOLON_ASR_MAX_CONNECTIONS`、`EIDOLON_ASR_REALTIME_SLOTS`、
`EIDOLON_ASR_MAX_QUEUED_UTTERANCES`、`EIDOLON_ASR_MAX_UTTERANCE_SECONDS` 和
`EIDOLON_ASR_MAX_QUEUE_WAIT_SECONDS`。Pi 5 实测建议保持 2 个实时槽；提高连接数不会增加模型
内存，提高实时槽数才会增加并行 cache、CPU 争用和尾延迟。

完整测试：

```bash
./scripts/eidolon-asr test
```

Python 固定为 3.12，由 `uv.lock` 固定跨平台依赖。Host 不改变服务协议和命令；当前 Mac
与树莓派 5 都解析为 `onnx-cpu`。将来只有在仓库加入经过校验的 RKNN 制品后，RK3588
才会把 `auto` 解析为 `rknn`，Channel 不需要修改。

启动脚本优先直接使用项目已有的 `.venv`；仅在环境尚未创建时才调用 `uv` 引导安装。
因此部署完成后的启动和重启不会重新解析依赖，也不会下载模型。

详细协议、部署和测试设计见 [DESIGN.md](DESIGN.md)。
Mac 与 Raspberry Pi 5 的并发时延结果见 [BENCHMARK.md](BENCHMARK.md)。
