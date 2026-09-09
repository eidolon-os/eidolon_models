# CosyVoice2 RK3588 流式合成产物 2026-07-21

这是本机 TTS 的模型包。此前它只以一份手工副本存在于板子的
`/var/lib/eidolon/models/cosyvoice2`——`eidolon-tts.service` 起得来，只是因为
有人放过一次。现在它在这里，一台新板子只靠发布就有声音。

产物固定自 [`Sariel00/cosyvoice2_rknn`](https://huggingface.co/Sariel00/cosyvoice2_rknn)
commit `47e9a3ea6724b0a65f8c433c77281258ac393f5a`（该仓库自称 archive date
2026-07-21）。这是 CosyVoice2-0.5B 针对 RK3588 做「一步 Flow 流式蒸馏」后的
RKNN / RKLLM 转换产物。

**它们是转换输出，但仍然有可指向的出处。** 上游把转换产物、转换脚本
（`onnx/conversion/source/`）和一份覆盖全部 1785 个文件的 `MANIFEST.sha256`
一起发布了。所以本包 26 个文件里有 24 个的 sha256 是**逐字节比对过**上游
manifest 的，`ops/component.toml` 也照同一批 digest 把它们声明成了 artifact。
剩下 2 个是转换运行时生成的 sidecar `manifest.json`，上游没有发布——
`manifest.json` 的 `source.unpublished_files` 把这件事写明了。

转换**不是**在这块板子上做的，也不是 Eidolon 做的：那两个 sidecar 里嵌着生产它们
的那台机器的绝对路径（`/home/serial/project/nextTTS/...`），而板上没有任何转换
工具链。

## 只取需要的那些

26 个文件是 `src/eidolon_models_tts/config.py` 的 `required_paths()` 要的全部，
不多一个。上游还有 HiFT 的 mel74/mel108 桶和三核的 `qwen2_body_w8a8_c3_*`，
只在 `--hift-batch2` 或 steady-flow 下用得到，服务两个开关都不开；teacher-replay
的其余 fixture 对应一个服务明确拒绝的模式。

## 音色

默认 `testwav_prompt3s`，即 `DEFAULT_VOICE`。**上游自己的默认是
`testwav_llm_prompt1s`，本包不带它**——`required_paths()` 只要前者。把
`EIDOLON_TTS_VOICE` 指向别的名字，得先把那个 profile 加进来。

## 规格

| 项 | 值 |
| --- | --- |
| 语言 | 中文 |
| 输出 | 24 kHz、16-bit、单声道、`pcm_s16le` |
| 输入 | 已规范化的 UTF-8 文本 |
| 运行时 | RKLLM 1.3.0、RKNN 2.3.2、RKNPU driver 0.9.8 |
| 硬件 | RK3588，3 个 NPU 核 |
| 引擎 | `src/eidolon_models_tts/cpp`，主机上编译 |
| 许可证 | Apache-2.0（见 `LICENSE`） |

大小 1.5 GB，进 Git LFS。这不是新决定：`asr/*/2.0.5/` 已有 728 MB 的先例。校验值
逐文件记在 `manifest.json`，与 ASR 用的是同一个 `verify_artifacts`。

实测数字（首包、RTF、核分配）在 `HOST-RK3588.md` §2.21，不在这里。
