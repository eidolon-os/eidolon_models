# ASR concurrent streaming benchmark

## 2-pass Mac / Pi 5 复测（2026-08-26）

加入默认启用的 offline Paraformer second pass 后，在同一 Mac、同一 5.547 秒样本和相同四种
音频变体上复测。first interim 基本不变；新增成本集中在 EOT-final。下表仍取每组最差一路：
`offline decode` 与流式 `decode` 一样包含等待共享 ASR 锁的时间，因此并发 2 路的 223 ms
反映了一路等待另一路完成 second pass，而不是单次模型计算突然变慢。

| 并发 | first interim ms | EOT-final ms | offline decode ms | total RTF | 旧 EOT-final ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1739 | 128 | 98 | 0.075 | 34 |
| 2 | 1764 | 247 | 223 | 0.120 | 61 |
| 4 | 1819 | 474 | 91 | 0.254 | 106 |
| 8 | 1947 | 1013 | 120 | 0.545 | 166 |

四种完整语音变体的 streaming/offline 文本均一致，final 正确且带句号。额外的截断音频测试
确认 offline 能改写整句，但改写并不天然等于纠错：例如 1.2 秒截断从“欢迎”改为“欢迎你”，
2.8 秒截断也出现过更差的尾字。因此 `final_revised` 表示发生 revision，不表示质量已提升；是否
默认开启最终应以 Eidolon 自有完整语句、噪声和远场数据集的 CER 为准。

Pi 5 使用完全相同的仓库、模型、命令和本机 WebSocket 路径复测；四种完整音频变体同样全部
返回正确文本，且本次 streaming/offline 文本一致：

| 并发 | first interim ms | EOT-final ms | offline decode ms | total RTF | 旧 EOT-final ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1790 | 330 | 246 | 0.192 | 76 |
| 2 | 1897 | 622 | 471 | 0.411 | 192 |
| 4 | 2251 | 1548 | 819 | 0.881 | 733 |
| 8 | 3230 | 5382 | 1415 | 1.798 | 2892 |

Pi 5 的默认 2-pass 单实例并发预算建议从原来的 4 路下调到 **2 路**。4 路最大 total RTF
仍小于 1，但余量只剩约 12%，且 EOT-final 已到 1.55 秒；8 路明显超出实时容量。

offline 权重约 227 MiB。三模型服务常驻 RSS：Mac 约 1184 MiB，Pi 5 约 1011 MiB；相对各自
streaming + punctuation 基线，分别增加约 340 MiB 和 291 MiB。

测试日期：2026-08-26。样本为 5.547 秒、16 kHz mono PCM16 中文语音。基准在内存中确定性
生成四类音频流：原始、降低 6 dB、约 30 dB SNR 轻噪声、低音量加轻噪声；开启默认标点后
四种流均识别为：

> 欢迎大家来体验达摩院推出的语音识别模型。

每路客户端按 100 ms 一帧实时发送，并发级别为 1、2、4、8。表中时延均为该并发级别的
最差一路；每组样本数较少，因此不把结果包装成稳定的生产 p95。

## 指标定义

- connect：建立本机 WebSocket 到收到 `connected`；
- start ack：发送 `start` 到收到 `utterance_started`；
- first interim：发送首帧音频到收到首个非空 interim；
- EOT-final：发送 `end_utterance` 到收到 final；
- overhang：整路完成墙钟时间减去音频时长；
- ASR RTF：服务端累计 decode 时间除以音频时长，包含共享 ASR 模型锁的排队时间；
- offline decode / RTF：second pass 推理及共享 ASR 锁等待时间；
- punctuation：final-only 标点恢复耗时，包含共享标点模型锁的排队；
- total RTF：streaming、offline 和 punctuation 累计时间除以音频时长，小于 1 才能持续实时处理。

## 历史基线：Mac Apple Silicon（streaming + punctuation）

| 并发 | connect ms | start ack ms | first interim ms | EOT-final ms | overhang ms | 标点 ms | ASR RTF | total RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.1 | 0.9 | 1742 | 34 | 35 | 2.2 | 0.062 | 0.063 |
| 2 | 2.3 | 1.0 | 1752 | 61 | 62 | 1.6 | 0.093 | 0.094 |
| 4 | 1.9 | 1.2 | 1813 | 106 | 106 | 2.0 | 0.175 | 0.175 |
| 8 | 2.1 | 0.9 | 1928 | 166 | 167 | 2.8 | 0.320 | 0.320 |

8 路仍有足够实时余量，且全部流返回 7 个 interim 和正确 final。

## 历史基线：Raspberry Pi 5（streaming + punctuation）

| 并发 | connect ms | start ack ms | first interim ms | EOT-final ms | overhang ms | 标点 ms | ASR RTF | total RTF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.4 | 0.6 | 1839 | 76 | 76 | 3.1 | 0.151 | 0.151 |
| 2 | 2.9 | 1.9 | 1888 | 192 | 192 | 11.5 | 0.339 | 0.340 |
| 4 | 2.8 | 2.0 | 2102 | 733 | 733 | 12.3 | 0.732 | 0.732 |
| 8 | 10.5 | 10.1 | 2480 | 2892 | 2892 | 12.4 | 1.374 | 1.375 |

该历史配置建议单实例最多 4 路；默认 2-pass 配置已根据新实测下调为 2 路。若产品要求更多
并发，应使用更强 Host，或在 RK3588 上验证 NPU backend。

## 标点开关 A/B

同一版本代码、同一音频和同一 Host 分别启动有/无标点服务。表中仍是最差一路；单次运行的
EOT 会受共享 ASR 锁调度影响，因此判断标点的直接成本应看服务端 `punctuation ms`，不能把两次
运行的 EOT 差全部归因于标点。

| Host | 并发 | 无标点 EOT ms | 有标点 EOT ms | 标点自身 ms | 无/有标点 total RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mac | 1 | 29 | 34 | 2.2 | 0.061 / 0.063 |
| Mac | 2 | 45 | 61 | 1.6 | 0.104 / 0.094 |
| Mac | 4 | 73 | 106 | 2.0 | 0.181 / 0.175 |
| Mac | 8 | 134 | 166 | 2.8 | 0.317 / 0.320 |
| Pi 5 | 1 | 85 | 76 | 3.1 | 0.145 / 0.151 |
| Pi 5 | 2 | 175 | 192 | 11.5 | 0.313 / 0.340 |
| Pi 5 | 4 | 504 | 733 | 12.3 | 0.664 / 0.732 |
| Pi 5 | 8 | 2987 | 2892 | 12.4 | 1.378 / 1.375 |

标点权重目录为 274 MiB。常驻 RSS 的成对测量为 Mac 约 542 → 844 MiB、Pi 5 约
420 → 720 MiB，即增加约 300 MiB；这是标点模型比推理时延更显著的成本。

## 与当前 eidolon_channel 百炼 STT 对比

百炼通过 `eidolon_channel` 的实际 `fun-asr-realtime-2026-02-28` WebSocket 路径测试；本地列为
本页默认启用 offline second pass 与标点后的结果。三个实现都返回正确且带句末标点的文本。

| 并发 | first interim：Mac / Pi 5 / 百炼 ms | EOT-final：Mac / Pi 5 / 百炼 ms |
| ---: | ---: | ---: |
| 1 | 1739 / 1790 / 1897 | 128 / 330 / 1255 |
| 2 | 1764 / 1897 / 1857 | 247 / 622 / 793 |
| 4 | 1819 / 2251 / 1505 | 474 / 1548 / 883 |
| 8 | 1947 / 3230 / 2455 | 1013 / 5382 / 1092 |

Mac 本地在全部并发的 EOT-final 都优于百炼。Pi 5 在 1～2 路也优于百炼；4 路开始落后，
8 路已明显计算饱和。百炼没有公开服务端 decode 时间，因此不能把它的墙钟完成比例与本地
compute RTF 直接比较。

## 复现

先启动服务，再运行同一个跨 Host 命令：

```bash
./scripts/eidolon-asr serve
./scripts/eidolon-asr bench tests/data/asr_example_zh.wav --concurrency 1,2,4,8
```

默认是实时节奏；`--burst` 可用于只看最大吞吐，不应与实时交互时延混为一谈。命令输出保留
每一路原始指标和各组 p50/p95/max，适合后续 RK3588 直接复跑。
