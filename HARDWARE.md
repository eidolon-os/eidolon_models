# Host 能力基线

状态：`rk3588-verified` · 实测日期 2026-09-04

本文只做**跨 Host 横向对照**。每台机器的实测明细各自成文，见 §2。
推断和计划在 [VERIFICATION.md](VERIFICATION.md)，不在这里。

## 1. 三台 Host 对照

| | macOS arm64 | Raspberry Pi 5 | **RK3588 / Orange Pi 5 Max** |
| --- | --- | --- | --- |
| 定位 | 开发与正确性基线 | 边缘 CPU 基线（已放弃部署模型） | **本地推理主机** |
| 大核 | Apple Silicon | 4×A76 @ **2400** MHz | 4×A76 @ **2256** MHz |
| 小核 | — | — | 4×A55 @ 1800 MHz |
| 内存 | — | 8 GB | **16 GB** |
| NPU | — | 无 | **RKNPU2 ×3 核，driver v0.9.8** |
| 系统盘 | — | SD | **NVMe，读 2.4 / 写 2.0 GB/s** |
| ASR 三模型常驻 RSS | ~1184 MiB | ~1011 MiB | **1036 MiB** |
| ASR 单路 total RTF（`infer`，5.55s 样本） | 0.074 | 0.218 | **0.153** |
| 内存带宽 | 未测 | 未测 | **19–21 GB/s** |
| ASR 并发 1/2/4/8 total RTF | — | 0.181 / 0.396 / 0.449 / 超时 | **0.234 / 0.371 / 0.388 / 超时** |

**RK3588 的 A76 主频比 Pi 5 低约 6%，但实测两机在 ASR 上大致持平**（见 [HOST-RK3588.md §2.6](HOST-RK3588.md)）。主频不是主导因素——内存带宽和核数更重要。**不要用主频推断性能。**

同时：截至 2026-09-04，唯一实测过的 NPU 流式 ASR 路径（zipformer encoder）比 ONNX CPU **慢 2.2 倍**，所以"NPU 是这块板唯一的能力增量"这一说法尚未被证实。

## 2. 各 Host 的实测明细

* **RK3588 / Orange Pi 5 Max** —— [HOST-RK3588.md](HOST-RK3588.md)。
  系统与 NPU、内存带宽、存储散热、音频、funasr 2pass、Qwen3-1.7B、
  前缀缓存、CosyVoice2 分组件与端到端、Kokoro 对照、bge、
  以及 ASR/LLM/TTS/bge 的联合压测与 CPU/NPU 分配方案。
* **Raspberry Pi 5** —— 已放弃部署模型，仅保留上表中的对照数字。
* **macOS arm64** —— 开发与正确性基线，仅保留上表中的对照数字。
