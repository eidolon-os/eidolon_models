# ASR models

Eidolon OS 私有化语音识别模型与统一推理服务。

## 当前选型

第一版只部署一个模型：

- `paraformer-zh-streaming/2.0.5`：中文流式 Paraformer INT8 ONNX。

刻意不包含：

- offline ASR：先节省约一份 ASR 权重空间，不做第二遍纠错；
- VAD：由 `eidolon_channel` 统一负责起声、停声、抢话和 EOT；
- punctuation：当前 EOT 本身没有标点恢复模型，且现有百炼 adapter 实际传递的是 raw text；
- ITN、LM、speaker model：待自有测试证明需要后再增加。

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
