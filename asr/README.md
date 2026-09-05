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

默认监听 `127.0.0.1:8768`：

```bash
curl http://127.0.0.1:8768/healthz
curl http://127.0.0.1:8768/readyz
curl http://127.0.0.1:8768/v1/info
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

### CPU 核绑定

绑核用一个变量控制，**线程数会自动跟随**，不需要再单独设：

```bash
EIDOLON_ASR_CPU_AFFINITY=4,5 ./scripts/eidolon-asr serve   # 绑 2 个 A76，池自动=2
```

语法同 `taskset`（`4,5`、`0-3`、`4-7`、`0-3,7`）。**留空=不绑核**，这是当前
RK3588 上的方案。intra_op 池按 `os.process_cpu_count()` 取数并夹到 4，它遵守
亲和性掩码——所以绑几个核就开几个线程，不会出现「绑 2 个核却开 4 个线程」的
超订（那会让 RTF 差 2.5 倍，见 HOST-RK3588.md §2.16 ③）。

`EIDOLON_ASR_THREADS` 仍可手工覆盖线程数，但那会解除上述联动，仅供排查。
在没有亲和性接口的平台（macOS）设置 `EIDOLON_ASR_CPU_AFFINITY` 会直接报错，
不会静默忽略——静默的绑核失败正是超订的来源。

部署时不要把值写死在 unit 里：`deploy/systemd/eidolon-asr.service` 通过
`EnvironmentFile=-/etc/eidolon/cpu-allocation.env` 读取，全板分配集中在
`deploy/cpu-allocation.env` 一个文件里，改完 `systemctl restart eidolon-asr`
即可，不需要 `daemon-reload`。

`eidolon-asr doctor` 会报告 `cpu_affinity` / `process_cpu_count` /
`intra_op_threads` 三项，用来确认实际生效的是什么。

要改这个决策，先重跑证据——不必手写脚本：

```bash
scripts/asr-affinity-sweep                          # 默认候选：不绑核 / 4,5 / 4-7 / 0-3
scripts/asr-affinity-sweep -r 5 -n "LLM 并发中" - 4,5   # 只比两个候选
```

它逐个候选起停 serve、跑同一段音频，输出可直接粘进 HOST-RK3588.md 的表格，
并记录温度与条件。换新板子时先跑 `scripts/host-probe` 做一次硬件盘点
（CPU 频率与 governor、NPU/GPU、OpenCL/Vulkan、存储、网络）。

完整测试：

```bash
./scripts/eidolon-asr test
```

Python 固定为 3.13（`requires-python = ">=3.13,<3.14"`），由 `uv.lock` 固定跨平台依赖。
Host 不改变服务协议和命令；当前 Mac 与树莓派 5 都解析为 `onnx-cpu`。将来只有在仓库加入经过校验的 RKNN 制品后，RK3588
才会把 `auto` 解析为 `rknn`，Channel 不需要修改。

启动脚本优先直接使用项目已有的 `.venv`；仅在环境尚未创建时才调用 `uv` 引导安装。
因此部署完成后的启动和重启不会重新解析依赖，也不会下载模型。

详细协议、部署和测试设计见 [DESIGN.md](DESIGN.md)。
Mac 与 Raspberry Pi 5 的并发时延结果见 [BENCHMARK.md](BENCHMARK.md)。
