# ASR / TTS / LLM 验证方案（RK3588）

状态：`plan` · 2026-09-04 · 依据 [HOST-RK3588.md](HOST-RK3588.md) 的实测基线

## 0. 这份方案的三条前提

实测数字改变了选型的约束条件，方案由这三条推出，不是按模型流行度排的：

1. **内存带宽 20 GB/s 是硬天花板**，且 CPU/NPU/GPU 共享。本地 LLM 的 decode 速率 ≈ 带宽 ÷ 权重字节，与 6 TOPS 无关。
2. **NPU 的价值尚未被证实。** A76 主频比 Pi 5 低 6%，但 funasr 2pass 实测两机**大致持平**（`infer` 0.153 vs 0.218，`bench` 0.234 vs 0.181，方向相反）。而唯一实测过的 NPU 流式 ASR 路径（zipformer encoder）**比 ONNX CPU 慢 2.2 倍**且 int8 转换失败。所以原先"RKNN 是这块板唯一立项理由"的判断**已被数据推翻**：目前唯一有官方数据支撑 NPU 胜出的负载是 **LLM（RKLLM）**。
3. **`AUTO` 只用一个 NPU 核**，三核要显式 `rknn_set_core_mask()`。多模型应按核绑定，不是共享排队。
4. **GPU（Mali G610）不可用于计算**——无 libmali、无 OpenCL ICD。不进分配方案。

## 1. 前置条件（不满足别开始）

| # | 前置 | 为什么是阻断项 |
| --- | --- | --- |
| P0-1 | **有线网络** | ops 的 `require_wired_release_upload = true` 会拒绝在无线上做 release 上传；2.4 GHz WiFi 实测抖动 15 ms 且掉线，不能承载交付 |
| P0-2 | ~~一台 x86_64 Linux 转换机~~ **已解除** | `rknn-toolkit2` 2.3.2 有 aarch64 wheel，**转换可以在板子上完成**，已于 2026-09-04 端到端验证。见 §2.0 |
| P0-3 | **自有中文 CER 评测集** | 没有它，"选型"只是"能跑"。当前 BENCHMARK.md 全是时延，一个准确率数字都没有 |
| P0-4 | **端口注册** | ✅ 已解除（2026-09-05）：`eidolon-asr` 默认改为 **8768**。8767 归 `channel_provider`（`eidolon_ops/.../source_assets.py` 与 `eidolon_channel/config/channel-provider.yaml` 都已占用），本服务未部署过、是后来者，所以让位。端口角色 `asr_stream` 已在 `ops/component.toml` 声明，冲突今后由 ops 的角色注册表在装配时拦截。TTS/LLM 的端口等它们真有 unit 时再申请，先占号等于声明不存在的服务 |

P0-2 的替代要求（不是阻断项，只是环境细节）：`rknn-toolkit2` / `rknn-toolkit-lite2` 的 wheel 只到 **cp312**，
而板子系统 Python 是 3.14、仓库 pin 是 3.13。用 `uv venv --python 3.12` 单开一个转换/推理 venv 即可，
不影响仓库的 3.13 收口。另需 `setuptools<81`——`pkg_resources` 在 setuptools 81 起被移除，而 rknn 仍依赖它。

P0-3 的最小可用形态：AISHELL-1 test（干净、可对外引用）+ WenetSpeech test_meeting（远场难例）+ **200 条自有真机录音**（板载 ES8388 录、1 m / 3 m、有回声、有 TTS 泄漏）。指标固定为 CER / 热词召回 / 标点 F1 / EOT-final p50·p95。

## 2. 阶段一：RKNN 工具链 spike —— 第一个目标不是 ASR

### 2.0 已提前验证（2026-09-04）

在板子上用合成模型跑完了整条链路：**PyTorch → ONNX → `.rknn`（int8 量化）→ 真 NPU 执行 → 与 ONNX Runtime 对比**。
两个模型：Conv2d（对照）与 **Conv1d/TDNN（CAMPPlus、ECAPA 的结构代理：Conv1d + dilation + mean pooling + Linear）**。

| 模型 | core_mask | 结果 | cos vs ONNX | 延迟 |
| --- | --- | --- | ---: | ---: |
| TDNN int8 | `CORE_0` | 真 NPU 执行 | **0.999966** | 0.492 ms（2033/s） |
| TDNN int8 | `CORE_0_1_2` | 真 NPU 执行 | 0.999966 | 0.667 ms（1498/s） |
| TDNN fp | `CORE_0` | 真 NPU 执行 | 1.000000 | 0.593 ms（1685/s） |
| TDNN fp | `CORE_0_1_2` | 真 NPU 执行 | 1.000000 | 0.770 ms（1298/s） |

结论：**TDNN 结构在 RKNPU2 上转换干净、int8 量化几乎无精度损失（远高于本阶段 ≥0.99 的门槛）。**
阶段一的技术风险因此大幅下降——剩下的是真实权重、真实校准集和真实 EER，而不是"能不能转"。

**三个必须知道的坑（都是实测踩出来的）：**

1. `rknn.init_runtime()` **不带 `target=` 参数时用的是模拟器**，日志会打 `Target is None, use simulator!`。
   模拟器的精度数字不能当作 NPU 结论。板上真实执行要用 `rknn-toolkit-lite2` 的 `RKNNLite`（或 C API）。
2. `rknn.inference()` 对 4 维输入**默认按 NHWC 解释**。喂 NCHW 会报 shape 错误，需显式 `data_format='nchw'`。
3. **小模型上三核比单核慢。** TDNN 单核 0.492 ms、三核 0.667 ms；而 mobilenet_v1 是单核 2.11 ms、三核 1.19 ms。
   核数收益与模型规模相关，**必须逐模型实测**。对多模型共存反而是好事：小模型各占一核，延迟和并行度同时更优。

### 2.1 真实目标模型

**CAMPPlus（声纹）+ ECAPA-VoxCeleb（pVAD 目标说话人）**

选它们而不是 Paraformer，四条理由都成立（§2.0 的实测已经支持第 1 条）：

1. TDNN/CNN 结构，RKNPU2 对静态 shape 友好，转换成功率最高；
2. 成功即可**从 channel 移除 torch**：venv 1.8 GB → ~400 MB，RSS 省数百 MB，且 RK3588 上少一套要交叉编译的框架；
3. 用最低风险一次性验完 `asr/DESIGN.md` §7 列的五步（工具链 / 量化数据 / runtime ABI / 接口 / 漂移对比），**经验 100% 可复用到 Paraformer**；
4. 失败也只是维持现状，不阻塞主线。

拿 Paraformer 当第一个 RKNN 目标，是把最难的转换和最不确定的工具链绑在一起赌。

**步骤**

1. x86 机装 `rknn-toolkit2`，把 CAMPPlus / ECAPA 的 ONNX 转 `.rknn`（RK3588 target，int8，量化校准集用自有语音）；
2. 产出 `.rknn` + 转换 manifest + sha256，按仓库现有 `manifest.json` 规范落盘；
3. 板子上用 C API 或 `rknn_toolkit_lite2` 跑推理（注意 lite2 只到 **cp312**，仓库 pin 是 3.13 → 走 C API / ctypes，或单开 3.12 venv）；
4. 与 ONNX CPU 版对比 embedding 余弦相似度、延迟、RSS。

**通过门槛**

| 指标 | 门槛 |
| --- | --- |
| embedding 与 ONNX 版余弦相似度 | ≥ 0.99 |
| 说话人验证 EER 漂移 | ≤ 1 个百分点 |
| 单次推理延迟 | ≤ ONNX CPU 版的 50% |
| channel venv 体积 | 1.8 GB → **< 500 MB** |

**失败分支**：若相似度或 EER 不达标，保留 ONNX CPU 路径，但**必须把 ECAPA/CAMPPlus 换成 ONNX Runtime 版**（两者都有 ONNX 导出），单独砍掉 torch——这个收益与 NPU 无关，独立成立。

## 3. 阶段二：ASR

分两条并行的轨，**不要混在一起判定**。

### 轨 A：CPU 正确性基线（1 天，不占日程）

在 RK3588 上跑现有 `./scripts/eidolon-asr test` 全套 + `bench`。目的是架构正确性和回归基线，**不是性能探索**——预期结果就是比 Pi 5 慢 6%。

| 指标 | 门槛 |
| --- | --- |
| 全套测试 | 全绿（Pi 5 是 38 passed） |
| 单路 total RTF（5.55 s 样本） | ≤ 0.25（Pi 5 实测 0.192） |
| 中文关键词 | 与 Mac / Pi 5 一致 |

### 轨 B：runtime 决策 + RKNN spike

**先换 runtime，再谈换模型。** 当前绑死的 `funasr-onnx==0.4.2` 有三个结构性问题：上游基本不维护；把 frontend 状态挂在 model 对象上，迫使 `FunASROnnxBackend` 用一把全局锁把两遍推理串成一条线（`backend.py` 注释自己写了）；**没有 RKNN 通路**。

候选：**sherpa-onnx**。C++ 核、`create_stream()` 每路独立状态（全局锁问题从根上消失）、内置 hotwords、且是目前唯一有活跃 RK3588 RKNN 后端的开源流式 ASR 框架。

> 待核实：sherpa-onnx 的 RKNN 后端实际覆盖哪些模型架构（印象里 streaming zipformer 优先，paraformer 未必）。**这是阶段二的第一个待答问题，不是结论。**

**步骤**

1. 板子上装 sherpa-onnx，用**同一份 paraformer ONNX**（模型先不动）跑通；
2. 对比四项：单路 RTF、4 路并发的 interim 抖动（对照现有全局锁版本）、常驻 RSS、RKNN 后端覆盖面；
3. 决策点：抖动和 RSS 明显赢就换 runtime。这一步的收益（去锁 + 拿到 hotwords + 拿到 RKNN 通路）比任何换模型都大；
4. 若 RKNN 覆盖 paraformer，用阶段一的工具链转换并对比。

**通过门槛**

| 指标 | 门槛 |
| --- | --- |
| 4 路并发 interim 抖动（p95 − p50） | 优于 funasr-onnx 版 |
| 三模型常驻 RSS | **≤ 700 MiB**（当前 Pi 5 是 1011 MiB） |
| CER | 不劣于云端基线 + 2 个百分点 |
| NPU 版（若转成） | decode 时间 ≤ CPU 版 50%，CER 漂移 ≤ 1 pp |

### 轨 C：形态收敛（依赖 P0-3）

1. **默认关闭 offline second pass。** 当前成本是 227 MiB 常驻 + Pi 5 上 EOT-final +230 ms，收益未证明——`asr/BENCHMARK.md` 自己记录了它把"欢迎"改成"欢迎你"。用 CER 表决定保留、还是换 **SenseVoice-Small**（234 M，int8 ~230 MB，自带标点 + ITN，一个模型替掉 offline Paraformer + CT-Transformer 两个，省约 270 MiB）。
2. **加热词接口。** 伴侣名、设备名、家人名现在必错，而这是陪伴场景出现频率最高的词。趁协议还没有消费者改，成本最低。

## 4. 阶段三：TTS

### 4.0 前提修正：门槛应对齐线上实测，且要同口径（2026-09-05）

原方案里「首包 < 300 ms」是拍的。从 `eidolon_channel/benchmark/` 取线上真实数据：

| 指标 | 边界 | 样本 | p50 | p95 |
| --- | --- | ---: | ---: | ---: |
| **`commit_to_tts_first_audio_ms`** | **EOT → 首个音频（真端到端）** | 398 | **2799 ms** | 5248 |
| `llm_started_to_tts_first_audio_ms` | LLM 启动 → 首个音频 | 655 | 2640 | 6728 |
| `brain_first_delta_to_tts_first_audio_ms` | 脑首 token → 首个音频 | 92 | 614 | 1166 |
| `tts_request_to_provider_first_audio_ms` | TTS 请求 → 首音频（**含等文本**） | 621 | 652 | 1129 |
| **`tts_first_text_sent_to_provider_first_audio_ms`** | **文本发出 → 首音频（纯 TTS）** | 657 | **330 ms** | 682 |

**⚠️ 口径很重要。** 本地 `ttft_first_pcm_ms`（文本就绪 → 首个 PCM）的同口径对照是
**`tts_first_text_sent_to_provider_first_audio_ms` = 330 ms**，不是 652 ms（后者含等 LLM 攒文本的时间）。

**修订后的 TTS 门槛：首包对齐云端 330 ms（p95 682 ms）。** 这是有实测依据的数字。

### 4.0.1 端到端影响推算

线上现状 `commit → first audio` p50 **2799 ms**，其中 TTS 仅占 330 ms。

| 方案 | 端到端 | 说明 |
| --- | ---: | --- |
| 现状（全云端） | **2799 ms** | 实测 |
| 仅 TTS 换本地 | 2799 − 330 + 1940 = **4409 ms** | 劣化 1.6 s |
| 全本地 | 250（ASR）+ 165（LLM 首 token）+ ~1040（攒首句 ≈12 tok @11.5 tok/s）+ 1940（TTS）≈ **3400 ms** | 劣化 0.6 s |

**关键反转：本地 ASR（250 ms）与本地 LLM（165 ms 首 token）都优于云端链路的对应贡献；
瓶颈完全集中在 TTS。** 若 TTS 能做到云端级别（330 ms），全本地端到端约 1785 ms，**反而优于现状 2799 ms**。

**所以 TTS 是唯一卡点。** 即便拿到 CosyVoice2 的优化版（TTFT 1070–1362 ms），仍是云端 330 ms 的 3–4 倍。

### 4.0.2 RK3588 上的 TTS 全景（ModelScope + GitHub 调研，2026-09-05）

在 ModelScope「语音合成」类目（2594 个模型）与 GitHub 上按「是否存在 RK3588/RKNN 移植」筛选：

| 模型 | RK3588 移植 | 最佳已知 RTF | 克隆能力 | 产物是否发布 |
| --- | --- | ---: | --- | --- |
| **CosyVoice2** | [Sariel00](https://huggingface.co/Sariel00/cosyvoice2_rknn)（C++ + 一步流蒸馏） | **~1.15**（本板实测） | ✅ 零样本 | ✅ **全部发布** |
| CosyVoice3 | [MasterVVK](https://github.com/MasterVVK/cosyvoice3-rknn-russian)（纯 Python） | 6.6–13.6× | ✅ | ❌ 需自转 |
| **Qwen3-TTS** | [MasterVVK](https://github.com/MasterVVK/qwen3-tts-rknn-russian)（Python） | **5.5×** | ✅ 3 秒克隆 + 音色描述 | ❌ 需自转（talker 要 x86） |
| Kokoro | [marty1885/kokoro-server](https://github.com/marty1885/kokoro-server) ⭐4 | 未知 | ❌ 仅预置多音色 | — |
| IndexTTS-2 / VoxCPM2 / Spark-TTS / Fish-Speech / GPT-SoVITS / MeloTTS | **完全没有** | — | — | — |

**结论：CosyVoice2 是目前 RK3588 上最快的、带零样本克隆的 TTS，且快出约 5 倍。选它是正确的。**

**Qwen3-TTS 的架构教训**（否定了「12 Hz 比 25 Hz 省一半」的直觉）：

```
Text → TextEmbedder(CPU) → Talker RKLLM(NPU) → CodePredictor(15组×5层) → Vocoder(RKNN)
```

12 Hz 的代价是**每帧 16 个码本**。Talker 确实快（30–60 ms/step），但 **Code Predictor 成为新瓶颈
（374–500 ms/step）**，而每 step 只对应 1/12 秒音频（83 ms）——仅此一段 RTF 就 4.5。
**帧率低 ≠ 计算少。**

另注：`cosyvoice3-axera-russian` 达到 RTF 1.26×，但那是 **AX650N** 独立 NPU，不是 RK3588。

### 4.1 CosyVoice 在 RK3588 的既有实现（社区调研，2026-09-05）

不是无人做过——有两个独立的开源实现，都在 RK3588 上实测过：

| 实现 | 平台 | LLM | Flow | HiFT | 引擎 | 实测 |
| --- | --- | --- | --- | --- | --- | --- |
| [`MasterVVK/cosyvoice3-rknn-russian`](https://github.com/MasterVVK/cosyvoice3-rknn-russian) | RK3588 | ONNX FP32 **CPU** 7.5 tok/s | FP16 RKNN，**10 步 ODE** | ONNX **CPU** | 纯 Python | **RTF 6.6×（短）/ 13.6×（长）** |
| [`Sariel00/cosyvoice2_rknn`](https://huggingface.co/Sariel00/cosyvoice2_rknn) | RK3588 | **RKLLM W8A8 NPU** | FP16 RKNN，**1 步蒸馏** | FP16 RKNN | 全 C++ | TTFT 1.07–1.36 s，RTF 0.6–0.8 |

两者相差约 10 倍，差距全在三处：**一步流蒸馏、RKLLM、C++ 引擎**。且 Sariel00 那组是**尚未开源**的优化版，作者原话「就算这样也没有把 TTFT 压进 1s」。

**运行时分工（两个实现一致）：LLM 段走 RKLLM，speech head / flow / HiFT 走 RKNN。**
因此自己转换**必须有 x86**（`rkllm_toolkit` 只有 `linux_x86_64`）；RKNN 那几段已验证板上可转。

MasterVVK 记录了两个反直觉的选择，**都是质量原因不是性能原因**：

- LLM 放 CPU：「CPU FP32 的语音 token 质量明显好于量化后的 NPU」；已知问题里写着 **RKLLM 版会截断/重复**（W8A8 量化所致）
- HiFT 放 CPU：「NPU 量化引入高频伪影」

### 4.2 瓶颈拆解与待验证项

从 MasterVVK 的分段数据（CosyVoice3 是 25 Hz，即每秒音频 25 个语音 token）：

| 阶段 | 实测 | 折算 RTF | 可修性 |
| --- | --- | ---: | --- |
| LLM 7.5 tok/s（CPU FP32） | — | **3.3** | ✅ 换 RKLLM 预期 ~30 tok/s → RTF 0.83（代价：质量） |
| **Flow 5.9 s / 2.1 s 音频（10 步 ODE）** | 5.9 s | **2.8** | ❓ **减 ODE 步数——待实测** |
| HiFT 1.1 s / 2.1 s 音频 | 1.1 s | 0.5 | ✅ 可上 NPU |

**唯一能改变结论的实验：把 `flow_fp16.onnx` 转 RKNN，量单步耗时，推算 1/2/4/10 步的 RTF。**
flow-matching 通常在 4–6 步下质量损失可接受，若成立则不必做蒸馏。

**若该实验失败**，CosyVoice 在这块板上只能做**非实时离线合成**（预生成固定话术），实时对话需要另选模型——但那时才有资格谈轻量方案，且必须同样实测。

## 4.9 阶段三：TTS（原方案）

当前 channel 只有 sensetime / bailian 两个云 TTS（`factory.py`），本地为零。

**候选（按"能上 RK3588 且实时"排）**

1. **MeloTTS-zh**（sherpa-onnx 的 `vits-melo-tts-zh_en`，~160 MB，CPU RTF ~0.1）——最稳最省，音色偏播报；
2. **Kokoro-82M**（含中文）——更小更快，自然度好于 VITS。

**明确排除**：CosyVoice2-0.5B / F5-TTS / IndexTTS2。质量最好，但 LLM + flow-matching 架构在 20 GB/s 带宽下做不到实时首包。**声音克隆需求留在云端。**

**通过门槛（这是 TTS 的真指标，不是 RTF）**

| 指标 | 门槛 |
| --- | --- |
| 首包延迟 | **< 300 ms** |
| barge-in cancel 生效 | **< 50 ms** |
| 整句 RTF | < 0.3 |
| 常驻 RSS | < 400 MiB |

必须按句切分流式吐 PCM 且支持中途取消，否则再快也会让全双工体验退化——channel 已有 `tts/_aggregator.py` 和 `_pool.py` 可对接。

## 5. 阶段四：LLM

**不需要新抽象。** `eidolon_agent/config/settings.example.yaml` 已经是 `openai/<model>` + `api_base` + `fallback_models`。本地 LLM 只要暴露 OpenAI 兼容 `/v1/chat/completions`，接入是改一行配置。任何自定义 LLM 协议都是纯负债。

**带宽反推的容量表（20 GB/s）**

| 模型（int4） | 权重 | 理论上限 | 现实预期（~60%） | 判定 |
| --- | ---: | ---: | ---: | --- |
| Qwen3-1.7B | ~1.0 GB | 20 tok/s | **~12 tok/s** | 交互可用 |
| Qwen3-4B | ~2.4 GB | 8.3 tok/s | **~5 tok/s** | 短应答可用 |
| 7–8B | ~4.5 GB | 4.4 tok/s | ~2.7 tok/s | **不可用** |

**定位：本地 LLM = 降级脑 + 隐私脑，不是主脑。** 主脑仍是 deepseek，用 `fallback_models` 反向配置（默认云、断网切本地）。

**通过门槛**

| 指标 | 1.7B | 4B |
| --- | ---: | ---: |
| decode | ≥ 8 tok/s | ≥ 4 tok/s |
| TTFT（1 k prefill） | < 1.5 s | < 3 s |
| 8 k context KV 占用 | 实测记录 | 实测记录 |

若实测 decode 低于理论上限的 50%，说明实现有问题（量化格式、KV 布局、算子回退），先查实现再换模型。

### 5.1 已完成：Qwen3-1.7B w8a8 实测（2026-09-04）

完整数据见 [HOST-RK3588.md §2.9](HOST-RK3588.md)。对照上面的门槛：

| 指标 | 门槛 | 实测 | 判定 |
| --- | ---: | ---: | --- |
| decode | ≥ 8 tok/s | **12.36**（A76）/ 9.25（A55） | ✅ |
| TTFT（1 k **token** 冷 prompt） | < 1.5 s | **~4.5 s** | ❌ **严重不达标** |
| VmHWM | — | **2336 MB**（预估 1800，需上调） | ⚠️ |

**瓶颈是 prefill 不是 decode**：每 token prefill ≈ 4.5 ms，近似线性。

**因此新增一条必需项（不是优化项）：`RKLLMPromptCacheParam`。**
system prompt + Companion genome 是固定前缀，缓存后只对新增话轮（~40 token）做 prefill：

| 段 | 冷 prompt | prompt cache |
| --- | ---: | ---: |
| EOT → ASR final | 250 ms | 250 ms |
| ASR final → LLM 首 token | ~4500 ms | **~180–280 ms** |
| LLM 首 token → TTS 首包 | ≤250 ms（待测） | ≤250 ms |
| **合计** | **~5 s** ❌ | **~700–780 ms**（当时的预测，见下方更正） |

**修订后的 Gate：带 prompt cache 的 TTFT < 300 ms。** 这是 §7 对话预算能否成立的唯一决定项。

> **已完成（2026-09-05）：达标，且比预期好。** 实测 TTFT **160–170 ms**（同前缀不同话轮，只 prefill 13–16 token）。
> **RKLLM 1.3.0 自带前缀缓存，无需调用任何 API**；`rkllm_load_prompt_cache` 只用于解决冷启动（首轮 190 ms）。
> ⚠️ 两条硬规则：**对话循环里不要调 `rkllm_clear_kv_cache`**（实测会导致全量重算）；**前缀必须逐字节稳定**，易变内容排在其后。
> 详见 [HOST-RK3588.md §2.10](HOST-RK3588.md)。**但对话预算合计不是 665 ms。**
> 那个数字里，"EOT → ASR final" 是分量相加推算的、"LLM 首 token → TTS 首包 ≤250 ms"
> 是未测的臆造值。2026-09-05 联合压测端到端实测：**≈ 4.06 s**
> （292 ASR + 3 bge + 165 LLM 首 token + 1604 LLM 生成首句 + 1998 TTS 首包）。
> 本 Gate（TTFT < 300 ms）依然达标，**失守的是它下游的 TTS**——
> 详见 [HOST-RK3588.md §2.17](HOST-RK3588.md)。

两条必须写进实现的配置：

- `RKLLMInput.enable_thinking = false` —— Qwen3 思考模式会先吐数百 `<think>` token，语音场景不可用
- `enabled_cpus_mask` —— **不要设。** RKLLM 默认自选 A76 cpu4-7，这就是最优；
  2026-09-05 实测强制 `mask=0xff, num=8` 会把 decode 从 12.94 打到 5.52 tok/s。
  另有硬约束 `enabled_cpus_num >= npu_core_num`（本模型为 3），给 2 个 CPU 直接
  `rkllm_init=-1`。与 ASR 的冲突在真实时序下不存在——ASR 的 offline 突发与
  LLM/TTS 天然串行，见 [HOST-RK3588.md §2.17](HOST-RK3588.md)

## 6. 阶段五：联合压测 —— 这才是最终验收

**前四个阶段的单项数字全部作废，除非它们在联合负载下仍成立。**

理由是算出来的：4B 以 5 tok/s decode 就要消耗 `2.4 GB × 5 = 12 GB/s`，约占全机带宽 **60%**。剩下 8 GB/s 要同时喂 ASR、TTS、bge embedding。

**场景**：LLM 持续 decode 的同时，跑一路 ASR + 一路 TTS + 一次 memory recall。

| 指标 | 门槛 |
| --- | --- |
| ASR 单路 total RTF | **< 1.0**（不能失去实时性） |
| TTS 首包 | < 500 ms |
| LLM decode 掉速 | ≤ 30% |
| 全机 RSS | **< 10 GB**（16 GB 下给页缓存和突发留 6 GB） |
| 持续 30 min 后温度 | 记录，观察是否降频 |

**核绑定方案**（依据阶段一/二的实测调整）：

| 资源 | 分配 |
| --- | --- |
| A76 4–6 | CosyVoice2（绑核，rtf 0.926；给到 4–7 反而是 1.015） |
| A76 4–7 | Qwen3 RKLLM —— **不传 mask，它默认自选这四个核** |
| A76（不绑核） | ASR funasr 2pass —— 内核 EAS 会把 offline 突发放上大核 |
| A76 7 + A55 | memory bge（绑 A76，线程数=核数）、channel、控制面、vision |
| NPU 三核 | **无法分区** —— RKLLM 无 core mask API，两个 RKLLM 各抓满 3 核 |

> **2026-09-05 联合压测修正**：本表两次被实测推翻，详见 HOST-RK3588.md §2.17。
> * 「NPU core0/1/2 分别给 ASR/TTS/LLM」**不成立**。`rkllm.h` 没有任何 NPU
>   core 接口，RKLLM 用满 3 核；Qwen3 与 CosyVoice2 并发时双方各掉 46%–64%，
>   NPU 近似串行（§2.15）。**LLM 与 TTS 必须串行，端到端按串行算。**
> * ASR 不该绑到小核。A55 的 eot_final 是 1271 ms，不绑核是 292 ms——
>   差 0.9 秒，且几乎全在 offline 二遍上（§2.16、§2.17 结论 ④）。
> * 静态划核在真实时序下大多是多余的：ASR 的 offline 突发与 LLM/TTS 天然串行，
>   争的是同一簇大核的不同时刻，不是同时刻（§2.17 S6）。

> **2026-09-05 实测修正**：本表原先把 memory bge 放在 A55 0–3，已推翻。
> bge-small-zh INT8（batch 32，seq_len 127）实测：A76×4 = 14.27 ms/doc，
> A55×4 = 56.76 ms/doc，A76 单核 = 51.65 ms/doc，A55 单核 = 261.5 ms/doc
> ——**A76 单核是 A55 单核的 5.06 倍**，主频只差 1.25 倍。A55 复核时
> cpu0 全程 1.8 GHz 满频、45–47 °C，不是降频假象。
>
> 同一批测量还给出第二条：让 ONNX Runtime 用满 8 核（15.81 ms/doc）
> **比只绑 4 个 A76 更慢**（14.27），且多占 4 个核。ORT 均分工作量，
> A55 线程晚约 5 倍完成，整批等最慢的那个。所以 `intra_op_threads`
> 不仅要显式绑核，**还必须把 A55 排除在外**。
>
> 真实工作点比上表的 seq_len 127 短得多：`benchmarks/results/bge-base-pi5/REPORT.md`
> 第 50–51 行记录 turn 粒度 median 42–44 token、p99 112–114。对应
> A76 上 median ≈ 4.8 ms/doc、p99 ≈ 12 ms/doc，单条 recall query 3.1 ms。
> 按这个量级，bge 在 A76×2 上（seq 44 约 8.5 ms/doc）就够用，
> 不必占满 4 个大核——留 2 个给实时路径。

> **2026-09-05 代码修正**：本节原写「当前 `config.py` 的
> `intra_op_threads = min(4, cpu_count)` ……**必须改成显式亲和性**」。
> 该句早于同日联合压测，与上方修正块冲突（表中 ASR 为**不绑核**，
> 绑 A55 的 eot_final 1271 ms vs 不绑核 292 ms），已作废。
>
> 线程数计算已改为 `os.process_cpu_count()`（`config.py` 的
> `_default_intra_op_threads()`），它遵守亲和性掩码。板上实测确认
> （`taskset -c 4,5` 下 `os.cpu_count()` 仍报 8，`os.process_cpu_count()` 报 2）。
> 注意其影响范围：**ASR 按现行方案不绑核，此时两者同为 8，仍夹取到 4，
> 行为与修改前完全一致**（实测 0.226 / 0.255，无回归）；该修正只在进程被
> taskset 绑到少于 4 个核时才生效，防止「绑 2 个核却开 4 个线程」的超订。
>
> 修复效果（HOST-RK3588.md §2.16 ③ 有完整数据）：A76×2 从 **0.708 降到
> 0.281**，2.52 倍。**这条对本节的核分配有新影响：A76×2（0.281）已逼近
> A76×4（0.253），ASR 只用 2 个大核就够，可以腾出 2 个给 LLM/TTS。**
> 但这与上方修正块「ASR 不绑核、静态划核多余」的结论冲突——两者都有实测
> 支撑（不绑核靠 EAS 把 offline 突发放上大核，绑 4–5 则牺牲突发峰值换确定性），
> **需要在混跑场景下另做一次对比才能定，目前维持不绑核。**
> 这个对比现在是一条命令，不必再手写脚本：
>
> ```bash
> scripts/asr-affinity-sweep -r 5 -n "LLM+TTS 并发中" - 4,5
> ```
>
> 空载下已复测：不绑核 0.225、A76x2 0.280（governor=ondemand）。
> **决定（2026-09-05）：维持不绑核，推迟到 OPi 5 Max 实际部署后按真实
> 混跑负载实测再定。** 空载数据不足以支撑改动。
> 决策本身是数据，改 `deploy/cpu-allocation.env` 一行即可，线程数自动跟随。
>
> 「和 channel 抢大核」的顾虑仍未消除，但 09-05 的结论是静态划核在真实
> 时序下大多多余——ASR 的 offline 突发与 LLM/TTS 天然串行。若后续混跑
> 数据推翻这一点，再引入绑核，届时线程数会自动跟随。

## 7. 内存预算（16 GB 已确认）

| 项 | 预算 | 依据 |
| --- | ---: | --- |
| 控制面 11 个服务（不含 channel） | ~1.6 GB | Pi 5 实测外推 |
| channel（pVAD/EOT/声纹） | ~1.5 GB | Pi 5 实测；阶段一成功后可降 |
| ASR | ~0.7 GB | 阶段二门槛 |
| TTS | ~0.4 GB | 阶段三门槛 |
| LLM 4B int4 + 8 k KV | ~3.5 GB | 待实测 |
| vision（若上机） | ~0.3 GB | 估 |
| journald + 页缓存 + 突发 | ≥ 1.5 GB | Pi 5 上 journald 曾达 219 MB |
| **合计** | **≈ 9.5 GB** | 余 ~6.5 GB |

16 GB 够，前提是 ASR 降到 0.7 GB 且 LLM 锁在 4B / context ≤ 8 k。

## 8. 放弃判据（写在前面，避免沉没成本）

| 若出现 | 则 |
| --- | --- |
| 阶段一两个 TDNN 都转不成 | RKNN 通路存疑，**暂停所有 NPU 计划**，只做"去 torch"这一项收益，本地推理维持 CPU |
| Paraformer 与 zipformer 都无法转 RKNN | 本地 ASR 定为 CPU-only，性能等同 Pi 5，这块板的价值退回到"内存大 + NVMe 快" |
| 联合压测下 ASR total RTF ≥ 1.0 | 本地 LLM 与本地 ASR **不可同机共存**，二选一或上更强 Host |
| 4B decode < 3 tok/s | 本地 LLM 降级为 1.7B，只做意图分类与短应答，不做对话 |

## 9. 未验证清单（截至 2026-09-04）

- RKLLM 实际推理（LLM 上 NPU 完全未测）
- Paraformer / sherpa-onnx 的 RKNN 可转换性
- 板载 ES8388 codec 的实际录放链路
- 持续重载下的散热与降频行为
- 三模型在 RK3588 上的常驻 RSS
