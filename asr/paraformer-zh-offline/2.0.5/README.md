# Paraformer 中文离线 ASR ONNX 2.0.5

这是 Eidolon 本地 ASR 的 second-pass final 模型。模型固定自 ModelScope tag `v2.0.5`、
commit `7162433de474912574f9d03122263776db5b046a`，使用量化 ONNX 权重。

流式 Paraformer 继续产生低延迟 interim；`eidolon_channel` 发出 `end_utterance` 后，本模型对
内存中的完整 utterance 重新识别，所得文本覆盖流式文本，再交给 CT-Transformer 恢复标点。
它不负责 VAD、EOT 或说话人识别。

输入为 16 kHz、单声道、signed little-endian PCM16。模型和运行时代码许可证分别按
`manifest.json` 与仓库 `LICENSING.md` 追踪。
