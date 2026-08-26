# CT-Transformer 中文标点 ONNX

这是 Eidolon 中文 ASR final 文本使用的标点恢复制品。模型固定自 ModelScope commit
`8f239ff78c6267c4d859233e7eb3bbdb68c61824`（2024-09-25），使用量化 ONNX 权重。

模型只接收 ASR 文本，不接收音频。流式 interim 继续返回 Paraformer 原文；Channel 发送
`end_utterance` 后，服务对完整 final 文本执行一次标点恢复，并同时返回 `raw_text`、
`punctuation_ms` 和包含 ASR 加标点的 `total_inference_ms`。这样不会因 partial 标点变化造成
流式文本反复改写。

默认启用。诊断或容量对照时可以通过 `--no-punctuation` 或
`EIDOLON_ASR_PUNCTUATION_ENABLED=0` 关闭。模型约 270 MiB，因此启用后会增加常驻内存和交付空间。

上游模型元数据声明 Apache-2.0；来源、revision 和文件摘要见 `manifest.json`。
