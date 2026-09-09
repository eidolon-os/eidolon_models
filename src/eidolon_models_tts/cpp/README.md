# CosyVoice2 流式引擎

这是本机 TTS 的推理引擎。它以前只存在于板子上一份手工放置的副本里
（`/root/cv2/cpp`），没有 git、没有构建来源——也就是说它跑得通，但发布不了。
现在它在这里。

**为什么不交付二进制。** 它链接 `librknnrt` 和 `librkllmrt`，那是板子自己的 NPU
运行时；而一次发布是精确的 git commit。所以源码随发布走，`scripts/eidolon-tts-build`
在主机上就着它要链接的运行时编译一次。

**为什么有两个基准 .cpp 在这里。** `cosyvoice2_streaming_pipeline.cpp` 把
`cosyvoice2_rkllm_generate_test.cpp` 和 `cosyvoice2_hift_rknn_benchmark.cpp`
以文本方式 include 进两个命名空间（并把它们的 `main` 改名藏掉），
`embedded_rkllm::InitializeRkllm`、`embedded_hift::F0Bucket` 这些都来自那里。
它们是真实依赖，不是遗留物。把可复用的部分抽成正经头文件是应该做的，但那是另一
件事——先让它可发布。

## 服务模式

`--serve` 让它常驻：一行 stdin 一句话，PCM 从一个专用描述符流出。

昂贵的东西只付一次：RKLLM 主体（508 MB）、四张 RKNN 图、HiFT 的分桶、以及它们
的预热——合计约 2.9 秒。每句话仍要付的是音色预计算（约 0.8 秒），因为请求要消费
它留在 encoder 和 Flow cache 里的状态。所以：

| | 每句起一个进程 | 常驻 |
|---|---|---|
| 加载 + 预热 | 2.9 s，每句 | 2.9 s，一次 |
| 音色预计算 | 0.8 s | 0.8 s |
| 首个 PCM | 2.1 s | 2.1 s |
| **调用方看到的首音** | **约 5.8 s** | **约 2.9 s** |

把音色预计算也省掉需要给 encoder / Flow 的 cache 加快照与恢复，那会把首音再压到
2.1 秒左右。还没做。

**stdout 上为什么不是音频。** librkllm 在加载模型时往 stdout 打自己的横幅（实测
567 字节）。库不是我们的，改不了它写哪儿。所以服务模式在加载任何库之前先把
stdout 复制一份留给音频，再让 fd 1 指向 stderr：库照它的习惯写 stdout，写进的是
journal。指标和状态也走 stderr，用 `--- eidolon-tts ... ---` 哨兵括起来，库自己
的噪声就落在括号之外。

**一句失败不带走服务。** 模型还在内存里，拒绝的理由从 stderr 出去，下一句照常。

非服务模式的行为一个字节都没变——板上比对过：同一句话、同样 102 个 token、
4.08 秒音频、TTFT 2121 ms、RTF 0.84。基准数字仍然可复现。
