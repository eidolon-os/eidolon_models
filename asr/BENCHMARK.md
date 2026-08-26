# ASR concurrent streaming benchmark

测试日期：2026-08-26。样本为 5.547 秒、16 kHz mono PCM16 中文语音。基准在内存中确定性
生成四类音频流：原始、降低 6 dB、约 30 dB SNR 轻噪声、低音量加轻噪声；四种流均识别为：

> 欢迎大家来体验达摩院推出的语音识别模型

每路客户端按 100 ms 一帧实时发送，并发级别为 1、2、4、8。表中时延均为该并发级别的
最差一路；每组样本数较少，因此不把结果包装成稳定的生产 p95。

## 指标定义

- connect：建立本机 WebSocket 到收到 `connected`；
- start ack：发送 `start` 到收到 `utterance_started`；
- first interim：发送首帧音频到收到首个非空 interim；
- EOT-final：发送 `end_utterance` 到收到 final；
- overhang：整路完成墙钟时间减去音频时长；
- RTF：服务端累计 decode 时间除以音频时长，包含共享模型锁的排队时间；小于 1 才能持续实时处理。

## Mac Apple Silicon

| 并发 | connect ms | start ack ms | first interim ms | EOT-final ms | overhang ms | 最大 RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.1 | 1.8 | 1744 | 29 | 29 | 0.065 |
| 2 | 2.3 | 0.6 | 1767 | 52 | 53 | 0.114 |
| 4 | 2.5 | 0.7 | 1815 | 76 | 77 | 0.156 |
| 8 | 2.0 | 0.9 | 1890 | 143 | 143 | 0.298 |

8 路仍有足够实时余量，且全部流返回 7 个 interim 和正确 final。

## Raspberry Pi 5

| 并发 | connect ms | start ack ms | first interim ms | EOT-final ms | overhang ms | 最大 RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.6 | 0.9 | 1826 | 71 | 71 | 0.145 |
| 2 | 6.8 | 0.8 | 1884 | 173 | 173 | 0.305 |
| 4 | 3.1 | 2.1 | 2103 | 694 | 695 | 0.657 |
| 8 | 13.5 | 2.9 | 2526 | 2881 | 2881 | 1.333 |

Pi 5 建议先把单实例并发预算设为 4 路。8 路协议和状态隔离仍正确，但已超过持续实时容量，
会累积排队延迟。若产品要求 8 路，应使用多实例/更强 Host，或在 RK3588 上验证 NPU backend。

## 复现

先启动服务，再运行同一个跨 Host 命令：

```bash
./scripts/eidolon-asr serve
./scripts/eidolon-asr bench tests/data/asr_example_zh.wav --concurrency 1,2,4,8
```

默认是实时节奏；`--burst` 可用于只看最大吞吐，不应与实时交互时延混为一谈。命令输出保留
每一路原始指标和各组 p50/p95/max，适合后续 RK3588 直接复跑。
