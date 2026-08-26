# Paraformer 中文流式 ASR ONNX 2.0.5

这是 Eidolon 第一版私有化中文流式 ASR 制品。模型固定自 ModelScope tag
`v2.0.5` / commit `ec6a3c64e290b719409e8c06cc2ac504e747c8eb`，使用 INT8 ONNX
encoder 与 decoder。

本制品只负责流式 ASR：不内嵌离线第二遍、VAD、标点、ITN 或说话人模型。语音端点由
`eidolon_channel` 的 VAD 统一拥有；收到 `end_utterance` 后，本模型刷新流式缓存并产生 raw
final，再由同一服务中的独立 CT-Transformer 制品恢复标点。

输入为 16 kHz、单声道、signed little-endian PCM16。模型和运行时代码许可证分别按
`manifest.json` 与仓库 `LICENSING.md` 追踪。
