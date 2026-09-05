# Host 能力基线

状态：`rk3588-verified` · 实测日期 2026-09-04

本文只记录**实测数字**。推断和计划在 [VERIFICATION.md](VERIFICATION.md)，不在这里。

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

**RK3588 的 A76 主频比 Pi 5 低约 6%，但实测两机在 ASR 上大致持平**（见 §2.6）。主频不是主导因素——内存带宽和核数更重要。**不要用主频推断性能。**

同时：截至 2026-09-04，唯一实测过的 NPU 流式 ASR 路径（zipformer encoder）比 ONNX CPU **慢 2.2 倍**，所以"NPU 是这块板唯一的能力增量"这一说法尚未被证实。

## 2. RK3588 实测明细

### 2.1 系统

```
Armbian_community 26.11.0-trunk (IMAGE_TYPE=nightly, BRANCH=vendor)
Ubuntu 26.04 resolute / GNOME
kernel 6.1.115-vendor-rk35xx
/proc/device-tree/model = RK3588 OPi 5 Max
系统 Python 3.14.4
根分区 ext4 on /dev/nvme0n1p1（SPI 引导 + armbian-install mtd 模式）
```

镜像自带 `Developer Preview Build ... do not use this image in production` 警告，且是 nightly。**不能作为产品底座**：交付前需把用过的 `.img` 与其 sha256 钉成自有 artifact，与 `nats-server` / `livekit-server` / `uv` / `node` 同等对待。

选 vendor 6.1 而非 edge 7.1.9：mainline 内核的 NPU 走上游 `rocket` 驱动，与 Rockchip 的 `librknnrt` / RKLLM 用户态不兼容。

### 2.2 NPU

```
driver          v0.9.8   （/sys/kernel/debug/rknpu/version）
librknnrt       2.3.2 (429f97ae6b@2025-04-09)
DRM 设备        /dev/dri/by-path/platform-fdab0000.npu-{card,render}
```

mobilenet_v1（RK3588 预编译，4.69 MB，1×224×224×3 uint8 → 1001 类），200 次迭代：

| core_mask | avg | min | max | 吞吐 |
| --- | ---: | ---: | ---: | ---: |
| `RKNN_NPU_CORE_AUTO` | 2.11 ms | 1.92 | 2.23 | 475 inf/s |
| `RKNN_NPU_CORE_0` | 2.17 ms | 2.08 | 2.27 | 460 inf/s |
| `RKNN_NPU_CORE_0_1_2` | **1.19 ms** | 1.12 | 1.36 | **839 inf/s** |

三种配置输出完全一致（`top1=905 val=0.3079`），确定性计算。运行中从 debugfs 采到 `NPU load: Core0: 69%`，确认是 NPU 硬件执行而非 CPU 回退。

**`AUTO` 只使用一个 NPU 核。** 要用满三核必须显式 `rknn_set_core_mask()`。这决定了多模型部署形态：单模型最低延迟用 `CORE_0_1_2`；多模型并存应按核绑定，而不是让它们在一个核上排队。

### 2.2b 模型转换与自举（板上完成，无需 x86）

`rknn-toolkit2` 2.3.2 提供 aarch64 wheel（cp36–cp312），**转换工具链可直接在这块板上运行**：

```
uv venv --python 3.12                       # wheel 只到 cp312
uv pip install rknn-toolkit2==2.3.2 "setuptools<81" rknn-toolkit-lite2==2.3.2
```

`setuptools<81` 是必需的：`pkg_resources` 自 setuptools 81 起被移除，而 rknn 仍依赖它。

端到端验证（PyTorch → ONNX → int8 `.rknn` → 真 NPU → 对比 ONNX Runtime），Conv1d/TDNN 结构：

| core_mask | cos vs ONNX Runtime | 延迟 | 吞吐 |
| --- | ---: | ---: | ---: |
| int8 `CORE_0` | **0.999966** | 0.492 ms | 2033/s |
| int8 `CORE_0_1_2` | 0.999966 | 0.667 ms | 1498/s |
| fp `CORE_0` | 1.000000 | 0.593 ms | 1685/s |
| fp `CORE_0_1_2` | 1.000000 | 0.770 ms | 1298/s |

**小模型三核慢于单核**（TDNN 0.492 → 0.667 ms），与 mobilenet_v1 相反（2.11 → 1.19 ms）。核数收益与模型规模相关，逐模型实测。

坑：`rknn.init_runtime()` 不带 `target=` 走**模拟器**（日志 `Target is None, use simulator!`），板上真实执行须用 `RKNNLite` 或 C API；`rknn.inference()` 对 4 维输入默认按 NHWC 解释。

### 2.3 内存带宽（STREAM，3×32M doubles = 768 MB）

| 线程 | Copy | Scale | Add | Triad |
| --- | ---: | ---: | ---: | ---: |
| 8（全核） | 18.98 | 19.63 | 19.76 | 18.73 |
| 4（绑大核 4-7） | 21.21 | 21.36 | 20.85 | 19.88 |
| 1 | 24.00 | 19.63 | 21.12 | 19.72 |

单位 GB/s。**工作值取 20 GB/s。** 单线程 Copy 偏高疑为 memcpy 的非临时存储优化，多线程的 19–21 GB/s 是更可信的持续值。

CPU、NPU、GPU 共享同一条 LPDDR。这个数字是本地 LLM 的硬天花板，与 6 TOPS 无关。

### 2.4 存储与散热

```
NVMe   Fanxiang S500MQ 256GB  →  238.5G，写 2.0 GB/s，读 2.4 GB/s（2GB O_DIRECT）
温度   空闲 39–41°C；STREAM 与 NPU 基准后 41–44°C
thermal zones: soc / bigcore0 / bigcore1 / littlecore / center / gpu / npu
```

轻载无降频风险。**持续重载未测。**

### 2.5 音频

```
card 0  rockchip-hdmi0        HDMI 输出
card 1  rockchip-hdmi1        HDMI 输出
card 2  rockchip,es8388       ES8323 codec，播放 + 录音
```

板载模拟 codec 存在，可在本机做 ASR/TTS 端到端验证，不必依赖外部设备。**实际录放未测。**

## 2.6 funasr 2pass 实测（CPU，2026-09-04）

仓库原样搬到板子（`rsync` 730 MB 权重）+ `uv sync --locked`，未改任何配置。

```
doctor / verify   三个模型 SHA-256 全部 ok，revision 正确
test              38 passed / 10.98 s   （Mac 38/3.70s，Pi 5 38/9.58s）
readyz            backend=onnx-cpu，host_kind="rk3588"   ← detect_host_kind() 正确
常驻 RSS          1036 MiB
温度              空闲 41°C → 跑后 61°C（大核）
```

单路 `infer`（5.547 s 中文样本）：

| 段 | Mac | Pi 5 | **RK3588** |
| --- | ---: | ---: | ---: |
| 流式 decode | 316 ms | 965 ms | **639 ms** |
| offline decode | 91 ms | 238 ms | **204 ms** |
| 标点 | 2.4 ms | 6.3 ms | **3.6 ms** |
| **total RTF** | 0.074 | 0.218 | **0.153** |

文本 `欢迎大家来体验达摩院推出的语音识别模型。` 正确带句号，`final_revised=false`（两遍一致），7 个 interim。

并发 `bench`：

| 并发 | Pi 5 total RTF | **RK3588 total RTF** |
| ---: | ---: | ---: |
| 1 | 0.181 | **0.234** |
| 2 | 0.396 | **0.371** max / 0.196 p50 |
| 4 | 0.449 | **0.388** max / 0.349 p50 |
| 8 | `capacity_timeout` | **`capacity_timeout`** |

**⚠️ 两条路径结论相反**：`infer` 显示 RK3588 快 30%，`bench` 并发 1 显示慢 29%。每个数据点只跑了一次，且 `bench` 取 4 种音频变体里最差一路。**结论只能是"大致持平"，要断言差异必须重复测量取 p95。**

**8 路仍然超时，与 Pi 5 一致**——换硬件没有修好容量模型（2 实时槽 + 6 排队 + 10 s 等待）。这是设计问题不是算力问题。

## 2.7 GPU（Mali G610）：不可用于计算

```
/dev/mali0                ✅ 内核驱动在
libmali                   ❌ 用户态 blob 缺失
/etc/OpenCL/vendors       ❌ 无 ICD 注册
libOpenCL.so.1            有加载器，无 vendor → 找不到设备
clinfo / vulkaninfo       未安装
GPU 频率                   300 MHz（空闲）
```

GNOME 能渲染（mesa/panfrost 提供 GL），但 **OpenCL / Vulkan 计算路径没有打通**。要用需装 libmali blob + 注册 ICD，且 Mali G610 的 OpenCL 在 ML 框架里支持度差。**GPU 不进分配方案。**

## 2.8 LLM 容量（RKLLM 官方实测，RK3588）

来自 `airockchip/rknn-llm` 的 `benchmark.md`（w8a8，seqlen 128，64 new tokens）：

| 模型 | TTFT | tok/s | 内存 |
| --- | ---: | ---: | ---: |
| Qwen3 0.6B | 199 ms | 32.9 | 791 MB |
| Qwen2.5 1.5B | 378 ms | 16.7 | 1689 MB |
| InternLM2 1.8B | 380 ms | 15.5 | 1776 MB |
| Qwen3.5 2B | 777 ms | 13.6 | 2122 MB |
| MiniCPM3 4B | 1408 ms | 5.9 | 4384 MB |

**Qwen3-1.7B 被 1.5B / 1.8B 夹住 → 预期 ~15–16 tok/s、TTFT ~380 ms、~1.8 GB。**
**⚠️ 本机实测低于该预期，见 §2.9。**

注意两点：**RK3588 上 `w8a8` 是官方默认，不是 w4a16**；且这些数字**优于**用 §2.3 的 20 GB/s 反推的结果，说明 **NPU 的内存通路比 CPU STREAM 测到的更高效**——不要用 CPU 带宽推 NPU 上限。

## 2.9 Qwen3-1.7B 实测（RKLLM，2026-09-04）

产物：`t-firefly/qwen3-1.7b-rkllm-rk3588` 的 `qwen3-1.7b_w8a8_rk3588.rkllm`（2397950956 B）。
**第三方产物，量化参数不透明、无 checksum 溯源——用于能力测量，不可作为交付物。**

运行时自报：

```
rkllm-runtime 1.3.0  ·  rknpu driver 0.9.8  ·  toolkit 1.3.0
max_context_limit: 16384  ·  npu_core_num: 3  ·  model_dtype: W8A8
默认 Enabled cpus: [4, 5, 6, 7]        ← RKLLM 自己就绑 A76 大核
```

模型加载 3041 ms（从 NVMe 读 2.3 GB）。`max_new_tokens=64`，实际因 EOS 停在 36–38 tok。

**默认（A76 cpu4-7）：**

| prompt 字数 | prefill tok | prefill ms | TTFT(wall) | decode tok/s | VmHWM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 64 | 286.0 | 290.7 | **12.36** | 2335 MB |
| 512 | 249 | 928.3 | 934.4 | 11.86 | 2336 MB |
| 1024 | 343 | 1513.5 | 1521.3 | 11.50 | 2336 MB |
| 2048 | 672 | 3401.1 | 3412.6 | 9.29 | 2337 MB |

**绑 A55（`enabled_cpus_mask=0x0F`）：**

| prompt 字数 | prefill ms | decode tok/s |
| ---: | ---: | ---: |
| 128 | 409.7 | 9.25 |
| 512 | 1735.2 | 8.53 |
| 1024 | 2833.5 | 7.18 |
| 2048 | 7019.4 | 6.05 |

跑后温度 49°C。

### 三条结论

**① decode 低于官方表，内存高于官方表。** 12.36 tok/s 对 官方 1.5B 的 16.7 / 1.8B 的 15.5；VmHWM **2336 MB** 对 官方 1776 MB。Qwen3 词表 151936，embedding + lm_head 显著更大。**内存预算按 2.34 GB 计，不是 1.8 GB。**

**② prefill 是瓶颈，不是 decode。** 每 token prefill ≈ **4.5 ms**，近似线性。真实 prompt（system + Companion genome + memory recall + 历史，约 1k token）冷启动 TTFT 约 **4.5 s**，比 400 ms 预算差一个数量级。

**③ 因此 `RKLLMPromptCacheParam` 是必需项，不是优化项。**

| 段 | 冷 prompt | prompt cache |
| --- | ---: | ---: |
| EOT → ASR final | 250 ms | 250 ms |
| ASR final → LLM 首 token | ~4500 ms | **~180–280 ms** |
| LLM 首 token → TTS 首包 | ≤250 ms（待测） | ≤250 ms |
| **合计** | **~5 s** ❌ | **~700–780 ms** ✅ |

**④ A76 比 A55 值钱：** decode +34%、prefill +30%。把 LLM 的 CPU 侧挪到 A55 以腾出大核给 ASR/TTS，代价是 LLM 性能 -25~33%，需联合压测定夺。

**⑤ Qwen3 必须关思考模式：** `RKLLMInput.enable_thinking = false`。否则先吐数百 `<think>` token，语音场景不可用。

## 2.10 Prompt 前缀缓存实测（2026-09-05）

同一 3560 字节中文 system prompt（≈740 token，模拟 Companion genome），`max_new_tokens=24`。

| 用例 | prefill tok | prefill ms | TTFT | decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| A1 冷启动（system+话轮） | 751 | 3036.3 | **3045.4** | 11.01 |
| A2 完全相同的 prompt 再来一次 | **0** | 0.0 | **96.7** | 11.24 |
| B2 同前缀 + 不同话轮 | **13** | 153.3 | **162.5** | 10.86 |
| B3 同前缀 + 又一话轮 | **16** | 161.4 | **170.8** | 11.04 |
| B4 同前缀 + 再一话轮 | **15** | 155.8 | **164.9** | 10.97 |
| C1 `clear_kv_cache(keep_system_prompt=1)` 之后 | **752** | 3014.0 | 3023.4 | 11.01 |
| C2 `clear_kv_cache(keep_system_prompt=0)` 之后 | 751 | 2991.0 | 3000.4 | 11.03 |
| D1 存盘缓存（一次性） | 742 | 2961.4 | — | — |
| D2 `rkllm_load_prompt_cache` 之后 | **20** | 180.7 | **189.9** | 10.93 |

### 四条结论

**① RKLLM 1.3.0 自带前缀缓存，无需调用任何 API。** 只要前缀不变，每轮只 prefill 新增话轮的 13–16 token，TTFT 稳定 **160–170 ms**——比冷启动快 18 倍。原先「必须实现 prompt cache」的判断要修正为：**在内存中它本来就有；需要显式做的只是冷启动。**

**② `rkllm_load_prompt_cache` 解决冷启动。** 一次性存盘（2961 ms），服务启动时载入 → 首轮 TTFT 190 ms，不用等 3 秒。

**③ 对话循环里绝对不要调 `rkllm_clear_kv_cache`。** 实测即使 `keep_system_prompt=1` 也会导致 **752 token 全量重算**（3014 ms）。只有切换 Companion（换 genome）时才该清，然后重新载入该 Companion 的缓存。

**④ 架构约束：前缀必须逐字节稳定。** 一旦 system prompt / genome 里混入每轮变化的内容（当前时间、memory recall 结果），前缀断裂 → 退回 3 秒。**易变内容必须排在稳定前缀之后，不得穿插：**

```
[稳定前缀 → 缓存命中]   system + Companion genome
[易变部分 → 每轮 prefill] 当前时间 + memory recall + 对话历史 + 本轮用户话
```

这是对 `eidolon_agent` 组 prompt 方式的硬约束，不是建议。

### 对话延迟预算

> **2026-09-05 更正（§2.17 实测）**：本表原先写作"合计 ~665 ms ✅ 低于 900 ms 门槛"。
> **这个结论是错的，实测约 4.06 s。** 两处来源问题：第一行的 ~250 ms 是把分量
> 相加推算的，不是端到端实测；第三行的"≤250 ms"当时就标着**未测**，而且那个
> 数字是我凭空定的（真实的云端同边界基线是 330 ms，见 VERIFICATION §4.0）。
> 下表用 §2.17 的实测值重写。

| 段 | 实测 | 来源 |
| --- | ---: | --- |
| 说完 → ASR 最终文本 | **292 ms** | §2.17 S6 中位（ASR 不绑核） |
| memory recall 查询编码 | 3 ms | §2.17（Chroma 检索本身**未测**） |
| **ASR 文本 → LLM 首 token（前缀命中）** | **165 ms** ✅ | 本节上表，空载测 |
| LLM 生成够起 TTS 的一句（20 token @ 12.47 tok/s） | 1604 ms | 由 §2.17 decode 速率折算 |
| **LLM 首 token → TTS 首包** | **1998 ms**（原写 ≤250 ms） | §2.17 S6 中位 |
| **合计** | **≈ 4.06 s** ✗ | |

前缀缓存这一段（165 ms）依然成立，本节的四条结论不变。变的是它下游：
TTS 首包 1998 ms 与 LLM 生成首句 1604 ms 合计占了预算的 89%，
把总数从 665 ms 推到 4 秒。详见 §2.17。

## 2.11 CosyVoice2 TTS 分组件实测（2026-09-05）

产物取自 [`Sariel00/cosyvoice2_rknn`](https://huggingface.co/Sariel00/cosyvoice2_rknn)，均为 toolkit 2.3.2 编译的
rk3588 **静态形状** RKNN 模型，与本机 runtime 同版本。用自写 C harness（`rknn_query` + `rknn_run`）计时。

**关键参数：一个 chunk = 25 speech token = 50 mel 帧 = 1.0 秒音频**（token 25 Hz，mel 50 Hz @ 24 kHz）。

| 组件 | 大小 | CORE_0 | CORE_0_1_2 | NPU 占用 | 每秒音频调用 | 折算 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| flow_encoder (chunk25) | 101.8 MB | **91.5 ms** | 93.7 ms | Core0 85% | 1× | 91 ms |
| **flow_estimator（1 步蒸馏）** | 165.8 MB | **327.0 ms** | 308.0 ms | Core0 91% | 1× | **327 ms** |
| hift_f0 (seq50) | 6.3 MB | 1.6 ms | 1.2 ms | 21% | 1× | 2 ms |
| **hift_decoder (mel50)** | 37.0 MB | **269.0 ms** | 277.1 ms | **仅 24%** | 1× | **269 ms** |
| speech_head | 11.3 MB | 1.8 ms | 1.3 ms | 21% | **25×** | 45 ms |
| **NPU 侧合计** | | | | | | **≈ 734 ms → RTF 0.73** |
| + LLM（Qwen2-0.5B w8a8，官方 41.6 tok/s） | 490 MB | | | | 25 token | +601 ms → RTF 0.60 |

**串行首包 ≈ 1334 ms，落在 Sariel00 公布的 1070–1362 ms 区间内——分组件实测独立复现了其整体数字。**

### 三条结论

**① 它的 RKNN 部分只用得动一个 NPU 核。** 三核几乎无收益（estimator 327→308 ms 仅 6%，encoder 和 hift 反而更慢），
NPU load 显示 Core0 80–91% 而 Core1/Core2 仅 10–13%。

> **2026-09-05 修正（§2.15 实测证否）**：本条原文写作「它只用得动一个 NPU 核……
> 另外两核可留给本地 LLM，资源冲突比预估小得多」，**这个推论是错的**。
> 上表只测了 RKNN 算子（flow_encoder / flow_estimator / hift / speech_head）。
> CosyVoice2 还有一个自己的 Qwen2-0.5B **走 RKLLM**（表末那行 +601 ms），
> 而 RKLLM 会抓满 3 个 NPU 核——§2.15 实测 Qwen3 单独跑时 Core0/1/2 各 45%。
> 所以「只用一个核」只对半条流水线成立，被误写成了对整条成立。
> §2.15 的并发实测给出真实答案：Qwen3 与 CosyVoice2 同跑，双方各掉 46%–64%，
> NPU 近似串行。**另外两核并不能留给本地 LLM。**

**② hift_decoder 是可优化的大头。** 269 ms 却只有 **24% NPU 占用**——不是算力瓶颈，是数据搬运
（每次进出各 216 KB）。上游仓库有 `rknn_zero_copy_benchmark.cpp`，说明作者也知道；零拷贝 API 有机会显著改善。

**③ 与云端基线的对照（见 VERIFICATION §4.0）：**

| | 首包 |
| --- | ---: |
| 线上百炼 CosyVoice | **p50 652 ms / p95 1129 ms** |
| 本地 CosyVoice2（本机实测推算） | **≈ 1334 ms** |

本地约为云端 p50 的 2 倍，略高于 p95。**能持续流式（RTF < 1）的前提是 LLM 与 flow/hift 跨核流水**，
否则 0.60 + 0.73 = 1.33 超实时。

## 2.12 CosyVoice2 端到端实测（2026-09-05）

用上游 [`Sariel00/cosyvoice2_rknn`](https://huggingface.co/Sariel00/cosyvoice2_rknn) 的 C++ 引擎
（`cosyvoice2_streaming_pipeline`，本机 cmake 4.2.3 + libpcre2 编译）和全部已发布产物，在本板跑通端到端合成。

文本「你好，今天天气不错，我们出去走走吧。」→ 4.32 s 音频，108 speech token。

| 配置 | TTFT | request_to_done_rtf | steady_pcm_rtf | 断流 | mel_cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| A 基线（首测） | 2230 ms | 1.253 | 0.915 | 3 | 0.999989 |
| A 基线（复测） | 2460 ms | 1.367 | 0.990 | 3 | 0.999989 |
| **B `--precompute-full-prompt`** | **1974 ms** | **1.185** | **0.904** | 3 | 0.999623 |
| C `--hift-batch2` | 2470 ms | 1.438 | 1.075 | 2 | 0.999989 |
| D B+C | 2144 ms | 1.319 | 1.021 | 2 | 0.999623 |
| E D + 分核（head0/enc1/flow0/hift2） | 2030 ms | 1.360 | 1.104 | 2 | 0.999623 |

分段时间线（配置 B）：`first_speech_token 331 ms → first_target_mel 1579 ms → first_pcm 1974 ms → done 5121 ms`。
峰值 RSS ≈ **1.5 GB**。跑后温度 45–46°C。

### 五条结论

**① 最好 TTFT 1974 ms，是云端 p50（652 ms）的 3 倍。** `--precompute-full-prompt` 是唯一有效的开关（−20%）。

**② 稳态跟不上播放——这是致命项。** ⚠️ 上表是 `teacher_replay=1`（回放固定的 108 token fixture），
不代表真实生成。加 `--sample` 后按文本真实合成，三档长度实测：

| 长度 | 文本 token | 语音 token | 音频 | TTFT | 总耗时 | `request_to_done_rtf` | **`steady_pcm_rtf`** | 断流 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 短 | 18 | 103 | 4.12 s | 1940 ms | 5.4 s | 1.311 | **1.056** | 3 |
| 中 | 54 | 280 | 11.20 s | 2098 ms | 13.8 s | 1.229 | **1.127** | 11 |
| 长 | 119 | 594 | 23.76 s | 2427 ms | 27.5 s | 1.157 | **1.093** | 23 |

**它是真流式**（PCM 在 ~2 s 开始吐出并持续），**首包也确实随文本变长被摊薄**（1.311→1.229→1.157）。
但 **`steady_pcm_rtf` 全部 > 1（1.06–1.13）——稳态生成追不上播放**，断流次数随时长线性增长
（约每秒音频断一次：3→11→23）。23.76 s 的音频要 27.5 s 生成，晚 3.7 s 完成、中间断 23 次。

**当前公开产物不具备实时对话能力。**

**③ `--hift-batch2` 是负优化**（TTFT 与 steady RTF 都变差），只把断流从 3 次降到 2 次。

**④ 分核几乎无收益**，印证 §2.11 的「它只用得动一个 NPU 核」。

**⑤ 零拷贝被证伪。** `rknn_zero_copy_benchmark` 对 hift_decoder 测得 **248.9 ms**，对比普通 IO 的 269.0 ms
只快 7.5%——那 250 ms 不是数据搬运瓶颈，§2.11 里 24% 的 NPU 占用读数具有误导性。**我识别的最大优化点不成立。**

**音质无问题**：所有配置 `mel_cosine_vs_host ≥ 0.9996`，NPU 输出忠实于主机参考。

### 与上游公布数据的差距

Sariel00 公布的 TTFT 1070–1362 ms / RTF 0.6–0.8 是**尚未开源的「极致优化版」**（其 README 明示「会在此仓库
心心 like 超过 100 时开源」）。**用当前公开产物在本板只能到 TTFT ≈ 2 s、RTF ≈ 1.19。**

## 2.13 Kokoro v1.1-zh 对照实测（2026-09-05）

`kokoro-int8-multi-lang-v1_1`（sherpa-onnx 官方发布，109 MB int8 ONNX + 51 MB voices，含中文 lexicon
与 date/number/phone-zh FST），经 `sherpa-onnx==1.13.7` aarch64 wheel 在本板 **CPU** 运行。
`num_speakers=103`，采样率 24 kHz。用与 CosyVoice2 完全相同的三段中文文本。

| 档 | 字数 | 音频 | 首包 | 总耗时 | RTF | 块数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 短 | 18 | 3.97 s | 2327 ms | 9534 ms | **2.404** | 3 |
| 中 | 54 | 11.43 s | 2290 ms | 26010 ms | **2.276** | 7 |
| 长 | 119 | 23.93 s | 2297 ms | 51508 ms | **2.153** | 12 |

线程与亲和性扫描（短句）：threads=2 → RTF 2.718；**threads=4 → 2.443（最优）**；threads=8 → 2.549；
threads=4 + `taskset -c 4-7` → 2.441。**不是配置问题。**

### TTS 三方对照（同口径：文本就绪 → 首个音频）

| 方案 | 引擎 | 首包 | RTF | 零样本克隆 | 音色 |
| --- | --- | ---: | ---: | --- | --- |
| **云端 CosyVoice（现状）** | 百炼 | **330 ms** | — | ✅ | — |
| **CosyVoice2 本地** | NPU（RKLLM + RKNN，C++） | **1940 ms** | **1.06–1.13** | ✅ | 无限 |
| Kokoro v1.1-zh 本地 | CPU int8 ONNX | 2290 ms | 2.15–2.40 | ❌ | 103 |

**CosyVoice2 在首包（快 18%）和 RTF（好一倍）两项上都优于 Kokoro，且多出零样本克隆能力。**

**顺带解开一个歧义**：`marty1885/kokoro-server` 的 README 写「decoder on RK3588 NPU at ~2.5× RTF」，
本次 CPU 实测正好 RTF 2.4——说明那句是「**RTF = 2.5**」而非「快 2.5 倍」，即其 NPU 版与 CPU 版性能相当。
该路径无价值。（且该移植基于 Kokoro **v1.0 英文版**，中文需另行适配。）

### 2.14 bge embedding（memory 的编码器，实测 2026-09-05）

跑的就是 Host 上会跑的那一份：`Xenova/bge-small-zh-v1.5@75c43b06` 的
`onnx/model_quantized.onnx`（INT8，24 MB）+ `tokenizer.json`，两个文件的
SHA-256 与 `eidolon_ops/src/eidolon_ops/embedding_model.py` 里钉的值逐字节相同。
编码路径按 `onnx_sentence_embedder.py` 复刻：CLS pooling、不加 query 前缀、
padding + truncation@512、L2 归一化、batch 32、`CPUExecutionProvider`。
ONNX Runtime 1.29.0 / Python 3.13。

**核簇与线程数**（合成中文文档，seq_len 127，512 docs，取 5 次中位数）：

| 绑核 | intra_op | ms/doc | 单条 query p50 | session load |
| --- | ---: | ---: | ---: | ---: |
| A76 4–7 | 4 | **14.27** | 3.14 ms | 100 ms |
| 全 8 核 | auto | 15.81 | 3.44 ms | 117 ms |
| A76 4–7 | 2 | 25.54 | 4.11 ms | 114 ms |
| A76 4–7 | 1 | 51.65 | 6.08 ms | 116 ms |
| A55 0–3 | 4 | 56.76 | 10.11 ms | 276 ms |
| A55 0–3 | 1 | 261.5 | 25.96 ms | 273 ms |

两条结论：

* **A76 单核是 A55 单核的 5.06 倍**（51.65 vs 261.5），而 A76 最高 2256 MHz、
  A55 最高 1800 MHz，只差 1.25 倍。整簇对整簇 3.97 倍。复核时采样 12 次
  cpufreq，cpu0 全程 1800000 kHz 满频、thermal_zone0 45–47 °C，
  **不是降频或过热造成的**。
* **用满 8 核比只用 4 个 A76 慢**（15.81 vs 14.27），还多占 4 个核。
  ORT 在算子内均分工作量，A55 线程晚约 5 倍完成，整批等最慢的那个。
  在这块板上 `intra_op_num_threads` 必须配合亲和性把 A55 排除。

**序列长度扫描**（A76×4，256 docs）：

| seq_len | 23 | 47 | 87 | 167 | 231 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ms/doc | 2.28 | 4.78 | 9.28 | 19.68 | 30.07 |

约 0.128 ms/doc/token，近似线性。单条 query 的 p50 在整个扫描区间恒定
3.10–3.15 ms——它是 batch 1 的短查询，与文档长度无关。

**真实工作点**：`eidolon_memory/benchmarks/results/bge-base-pi5/REPORT.md`
第 50–51 行记录 turn 粒度的 token 分布——locomo/turn median 42 / p99 114 / max 156，
clongeval/turn median 44 / p99 112 / max 327，**0% 触到 512 截断**。
落到上面的曲线上：median ≈ 4.8 ms/doc，p99 ≈ 12 ms/doc。
一次 recall 的查询编码 3.1 ms；一轮对话后写入 20 条 fragment 约 96 ms
（A76×4）。按这个量级 bge 在 A76×2 上就够用（seq 44 约 8.5 ms/doc），
不必占满 4 个大核。

**与 Pi 5 不可直接比较**：`eidolon_memory/config/settings.example.yaml` 第 56–59 行
那张表的 "ms/doc (Pi 5)" 一列里，只有 bge-base-zh 的 32.5 有真实 Pi5 运行记录
（`results/bge-base-pi5/results.json`，`runs/cmteb/ms_per_document = 32.45`）。
bge-small 的 6.5 和 bge-large 的 104.1 在仓库里**没有对应的 Pi5 运行**——
`runs/cmteb/video_bge-{small,base,large}-zh.json` 三份的 provenance 都是
`macOS-26.5.2-arm64`、12 核、batch 64（0.66 / 3.93 / 15.31 ms/doc）。
拿 6.5 去对照本板的数会得出"RK3588 比 Pi5 慢 2.4 倍"的错误结论，
实际差异全部来自模型档位（small vs base）与语料长度（VideoRetrieval 短标题
vs 本测 seq 127）。

**部署形态**：`eidolon_memory/config/settings.yaml` 现为 `provider: http`，
所有 Memory 进程共用常驻的 `eidolon-memory-embedder`（127.0.0.1:8760，
OpenAI 兼容 `/v1`，单请求上限 256 条 / 4 MB），**不是每个进程各持一份权重**。
所以在整机预算里它是一个进程、一份 24 MB 权重。

**未测**：ONNX Runtime 的 RKNPU EP 不存在，bge 目前只能跑 CPU；
是否值得把 bge 转成 RKNN 放到 NPU 上，取决于 §2.12 的 NPU 是否已被
LLM/TTS 占满——见联合压测。

### 2.15 联合压测 · NPU 争用（实测 2026-09-05）

问题：Qwen3-1.7B（RKLLM）与 CosyVoice2（RKLLM + RKNN）能否在 3 核 NPU 上
真正并行——也就是"边生成边播"这条链路是否成立。

两者单独跑、再并发跑，同一台机、同一 seed（`--sample --seed=7
--precompute-full-prompt`），CosyVoice2 两次都生成 4.32 s 音频：

| 指标 | 单独 | 并发 | 变化 |
| --- | ---: | ---: | ---: |
| Qwen3 decode @128 ctx | 12.66 tok/s | 6.67 | **−47%** |
| Qwen3 decode @512 ctx | 12.21 tok/s | 6.54 | **−46%** |
| Qwen3 prefill @128 ctx | 290.0 ms | 389.7 | +34% |
| Qwen3 prefill @512 ctx | 922.3 ms | 2180.1 | **+136%** |
| CosyVoice2 steady RTF | 0.924 | **1.519** | +64% |
| CosyVoice2 TTFT | 2229 ms | 2988 ms | +34% |
| CosyVoice2 underrun | 3 | 4 | |

llmbench 的第三行（1024 ctx，11.57 tok/s）不计入：CosyVoice2 在它之前就结束了，
只有前两行是真并发窗口。

**结论：NPU 近似串行。** 完全串行的预期是各得一半——Qwen3 6.3 tok/s（实测 6.67）、
CosyVoice2 RTF 1.85（实测 1.519）。实测比纯串行好约 22%，那点收益来自
CosyVoice2 的 CPU 阶段与 LLM 的 NPU 阶段错开，不是 NPU 上的并行。

NPU 占用采样（`/sys/kernel/debug/rknpu/load`，每 2 s）：

* Qwen3 单独：Core0/1/2 **各 45%**——RKLLM 会用满 3 个核，不是单核。
* 并发：Core0 78–92%，Core1/2 47–78%，**没有任何一个核到 100%**。
  多出来的是仲裁开销，不是有效并行。
* 全程 loadavg ≤ 2.70，thermal_zone0 48–54 °C，不是 CPU 或散热受限。

**没有办法把两者分到不同 NPU 核上。** `rkllm.h` 全文没有一处 `core`——
公开 API 只有 `enabled_cpus_num` / `enabled_cpus_mask`（CPU 亲和性）。
`librkllmrt.so` 内部有 `used_npu_core` / `MAX_NPU_CORE` / `Illegal core_mask`，
但不导出；运行期环境变量只有 `RKLLM_DUMP_LEVEL`、`RKLLM_LOG_LEVEL`、
`RKNN_LOG_LEVEL`、`RKNN_MIN_TIMEOUT_MS`。`.so` 里的 `rkllm.core_num` 是
**转换期写进 .rkllm 文件的配置键**——要限制 NPU 核数只能在 x86 上重新转换模型。
对比之下 RKNN 侧有 `rknn_set_core_mask`（`RKNN_NPU_CORE_0/1/2/0_1/0_1_2`），
所以 CosyVoice2 的 speech_head / flow_encoder / flow_estimator / hift 可以钉核
（pipeline 已暴露 `--head-core --encoder-core --flow-core --hift-core`），
但 Qwen3 和 CosyVoice2 的 `qwen2_body` 两个 RKLLM 都会各自抓满 3 核。

**对话架构的后果**：`steady_rtf 1.519 > 1` 表示并发时 TTS 追不上实时播放，
会持续欠载出声音空洞。在这块板子上 **LLM 生成与 TTS 合成必须串行**，
端到端时延要按串行计算，不能按"边生成边播"计算。

**未验证**：以 `core_num=1` 或 `2` 重新转换 Qwen3 后，是否能换来
"LLM 慢一些但 TTS 不掉出实时"的更优组合。需要 x86 主机上的 rkllm-toolkit。

### 2.16 联合压测 · ASR 核簇与实时余量（实测 2026-09-05）

funasr 2pass（paraformer-zh streaming + offline + punc，全 int8 ONNX，纯 CPU）。
`eidolon-asr serve` + `eidolon-asr bench`，实时节奏（非 `--burst`），
测试音频 `tests/data/asr_example_zh.wav`（5.55 s）。`total_rtf` 含流式解码、
offline 二遍与标点，是整条 ASR 链路对实时的占用比。

| serve 绑核 | 1 路 total_rtf | 2 路 p50 / p95 | 4 路 p50 / p95 |
| --- | ---: | --- | --- |
| A76 4–7 | **0.255** | 0.209 / 0.395 | 0.348 / 0.385 |
| A55 0–3 | **0.646** | 1.15 / 1.455 ✗ | 1.449 / 1.485 ✗ |
| A76 4–5（2 核） | **0.705** | — | — |

单用户家庭设备的常态是 1 路；2/4 路只用来看余量，不是目标形态。

**① A55 刚好能扛单路 ASR，余量 35%。** 0.646 表示 5.55 s 语音要 3.58 s 推理。
到 2 路就 1.15 超实时。所以「把 ASR 放小核、四个大核全留给 NPU 宿主线程」
在单路下成立，但没有冗余——这 0.646 还是**空载**测的，NPU 跑起来后共享
内存带宽会不会把它推过 1.0，见 §2.17。

**② A76 整簇比 A55 整簇只快 2.53 倍**（0.255 vs 0.646），远小于 §2.14 里
bge 的 3.97 倍。两者都是 int8 ONNX，差异来自算子构成不同，
**不能用 bge 的核簇比例去推 ASR 的**。

**③ 绑核必须和线程数一起调，否则适得其反。** A76×2 的 0.705 竟然比
A55×4 的 0.646 还差。原因是当时 `src/eidolon_models_asr/config.py` 用的是：

```python
intra_op_threads: int = max(1, min(4, os.cpu_count() or 1))
```

`os.cpu_count()` **不看 CPU 亲和性**——taskset 到 2 个核上它照样返回 8、
取 4 个线程，塞进 2 个核里超订。

**已修复**：现在按 `os.process_cpu_count()` 取数（`config.py` 的
`_default_intra_op_threads()`）。Python 3.13 里它在有 `sched_getaffinity`
的平台上就是 `len(os.sched_getaffinity(0))`，遵守 taskset 设的亲和性；
macOS 无该接口时退回 `os.cpu_count()`，与原行为一致。夹取上限 4 和
`EIDOLON_ASR_THREADS` 覆盖都不变，仍可用后者手工指定。
**上表 A76×2 的 0.705 是修复前测的，需要重测。**

### 2.17 联合压测 · CPU/NPU 分配方案（实测 2026-09-05）

在 §2.15（NPU 争用）、§2.16（ASR 核簇）、§2.14（bge 核簇）之上，把五个负载
放在一起找可用的分配。测了 5 组静态方案，外加一组按真实对话时序的验证。

#### 静态方案对照（ASR 持续循环 = 最坏情况）

| 方案 | ASR | LLM/TTS 宿主 | bge | ASR eot_final | LLM @512 | TTS rtf | bge ms/doc |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| S1 | A55 0–3 | A76 4–7 | A76 6–7 | 1295 ms | 12.82 | 0.951 | 8.86 |
| S2 | A76 6–7 | A76 4–5 | A55 0–3 | 1089 ms | **拒绝启动** | 1.251 ✗ | 24.16 |
| **S3** | A55 0–3 | **A76 4–6** | A76 7 | 1271 ms | **13.08** | **0.926** | 16.21 |
| S4 | A76 4–7 | A76 4–6 | A55 0–3 | **331 ms** | 9.40 | 1.142 ✗ | 20.28 |
| S5 | 不绑 | 不绑（mask 0xff）| 不绑 | 354 ms | 5.52 | 1.031 ✗ | **5.05** |

空载基线：ASR@A76 0.245 rtf / eot_final 329–443 ms；ASR@A55 0.646 / 1195–1297 ms；
LLM 12.21 tok/s；TTS rtf 0.924。

#### 五条实测结论

**① RKLLM 本身就是 big.LITTLE 感知的，不要给它传 mask。** 不传参时它自己选
`Enabled cpus: [4, 5, 6, 7]`，跑出 12.94 / 12.41 tok/s。S5 里我显式传
`mask=0xff, num=8` 反而掉到 6.32 / 5.52——A55 上的线程拖慢整体，
和 §2.14 里 bge 用满 8 核变慢是同一个掉队效应。**S5 的 LLM 数字不代表"默认行为"，
它代表"错误地强制用全部 8 核"。**

**② RKLLM 硬性要求 `enabled_cpus_num >= npu_core_num`。** 给 2 个 CPU 时直接
`rkllm_init=-1`，报 `E rkllm: The number of enabled CPUs must be greater than or
equal to the number of NPU cores.`。本模型烧的是 `npu_core_num: 3`，所以
**Qwen3 至少占 3 个 CPU 核**。这是 S2 那格空白的真实原因。

**③ 三个大核和四个大核几乎一样，多给的那个是浪费。**
LLM：A76×3 = 12.46/12.20/11.52 vs A76×4 = 12.66/12.21/11.70。
TTS：绑 4–6（三核）rtf **0.926**，绑 4–7（四核）rtf 1.015–1.036。
**少给一个核反而更稳**——第四个核只带来线程迁移和争抢。所以 cpu7 应当留给
bge 与控制面，不要划给 NPU 运行时。

**④ ASR 放小核的代价是用户能直接感觉到的 0.9 秒，不是吞吐。**
eot_final（说完到出最终文本）A55×4 空载 1195–1297 ms，A76×4 空载 329–443 ms，
差距几乎全在 offline 二遍上（774–852 ms vs 222–338 ms）。S3 共存下 A55 是
1271 ms，与空载相同——**慢的是小核本身，不是负载**。

**⑤ 但把 ASR 搬上大核会把 TTS 顶出实时。** S4 拿回了 940 ms 的 ASR 延迟，
代价是 LLM −28%（9.40 tok/s）且 TTS rtf 1.142。**播放中断音比开口前多等 0.9 秒更糟**，
所以 S4 在静态方案里不可取。根子是只有 4 个 A76，而 RKLLM 要 ≥3 个。

#### 真实时序验证（S6）：静态方案的前提不成立

上面 5 组都让 ASR 持续循环，这**不是真实占空比**——真实一轮对话里，助手说话时
ASR 是闲的，ASR 的 offline 突发与 LLM/TTS 天然串行，不需要各占一套核，
只需要在不同时刻拿到同一簇大核。

按真实时序重测三轮（ASR 不绑核 / bge 绑 A76 / RKLLM 不传 mask / TTS 绑 A76 4–7）：

| 阶段 | 轮 1 | 轮 2 | 轮 3 | 中位 |
| --- | ---: | ---: | ---: | ---: |
| ASR eot_final | 278 | 401 | 292 | **292 ms** |
| └ offline 二遍 | 201 | 320 | 197 | 201 ms |
| bge query 编码 | 3.29 | 3.20 | 3.19 | **3.20 ms** |
| LLM prefill @512（冷） | 876 | 902 | 897 | 897 ms |
| LLM decode | 12.47 | 12.59 | 12.17 | **12.47 tok/s** |
| TTS TTFT | 2009 | 1860 | 1998 | **1998 ms** |
| TTS steady rtf | 1.036 | 1.015 | 0.980 | 1.015 |
| bge 写入 10 条 fragment | 51 | 51 | 51 | **51 ms** |

**ASR 不绑核在真实时序下是 292 ms**，比 S4/S5 的 331/354 更好——那两个数里含了
我让 ASR 连续循环造成的自我争用。内核 EAS 会把 offline 那一下突发放到大核上，
不需要显式绑核。

#### 推荐分配

| 组件 | 分配 | 依据 |
| --- | --- | --- |
| Qwen3 (RKLLM) | **不传 mask，用默认** | 默认即 A76 4–7；强制 8 核掉一半（结论 ①） |
| CosyVoice2 (TTS) | 绑 **A76 4–6** | 4–6 得 rtf 0.926，4–7 得 1.015（结论 ③） |
| bge embedding | 绑 **A76**，线程数=核数 | ORT 不感知 big.LITTLE（§2.14） |
| ASR funasr 2pass | **不绑核** | 真实时序下 292 ms；绑 A55 要 1271 ms（结论 ④） |
| 控制面 / vision | A55 0–3 | 剩余 |

**内存**：三轮跑完 `used = 1965 MB / 15959 MB`。注意这是 TTS 进程退出后采的，
不是峰值。LLM 与 TTS 串行，峰值约为 常驻（ASR+bge+控制面）+ max(LLM VmHWM
2115 MB, TTS RSS 1.50 GB) ≈ **4 GB**，16 GB 余量充足。**峰值未直接测量。**

#### 端到端首音预算（各段实测，合成为一轮）

| 段 | 实测 | 来源 |
| --- | ---: | --- |
| 说完 → ASR 最终文本 | 292 ms | S6 中位 |
| memory recall 查询编码 | 3 ms | S6（**Chroma 检索本身未测**） |
| ASR 文本 → LLM 首 token（前缀命中） | 165 ms | §2.10（空载测，未在 S6 复测） |
| LLM 生成够起 TTS 的一句（20 token @ 12.47） | 1604 ms | 由 S6 decode 速率折算 |
| TTS 首包 | 1998 ms | S6 中位 |
| **合计** | **≈ 4.06 s** | |

**这与 §2.10 那张预算表的 665 ms 差 6 倍。** 那张表的第一行是分量相加推算的、
第三行（"LLM 首 token → TTS 首包 ≤250 ms"）当时标着"未测"且是我凭空定的数，
实测是 1998 ms。§2.10 已就地更正。

首音 4 秒的两个大头是 TTS 首包（1998 ms，49%）与 LLM 生成首句（1604 ms，40%）。
TTS 首包里 `first_target_mel` 就占 1474–1639 ms——它要先攒够 25 个 speech token
才能过 flow，而 RKNN 模型是 chunk25 静态形状，改小 chunk 要重新导出。

**并且 TTS 稳态 rtf 中位 1.015 > 1**：即便串行、即便独占大核，CosyVoice2 也刚好
追不上实时播放，三轮各有 3 次 underrun。这不是争用造成的（空载也是 0.924，
余量仅 8%），是模型本身在这块板上的能力边界。

## 3. 已知缺陷

| 缺陷 | 证据 | 影响 |
| --- | --- | --- |
| 两个 PCIe 控制器 link fail | `rk-pcie fe190000/fe170000: PCIe Link Fail, LTSSM 0x3` → `failed to initialize host` | 不影响 NVMe（另一控制器正常） |
| 有线口无链路 | `enP3p49s0` driver `r8169`，`carrier=0` | 当时无网线。**ops 的 `require_wired_release_upload = true` 会拒绝在无线上做 release 上传，所以有线是硬要求** |
| 2.4 GHz WiFi 抖动 | RTT `min/avg/max/mdev = 2.8/11.7/42.0/15.2` ms；SSH 建连 1.15–1.38 s；期间掉线一次 | 交互和大文件传输不可靠，不能用于交付链路 |
| NPU region 重复注册 | `RKNPU fdab0000.npu: can't request region for resource` | 未影响推理（上述基准通过）；记录备查 |

## 4. Python 与发行版解耦（实测）

板子系统 Python 是 3.14.4，仓库 pin 是 `>=3.13,<3.14`。在板子上跑 `uv sync`：

```
Downloading cpython-3.13.13-linux-aarch64-gnu (27.8MiB)
Using CPython 3.13.13
home = /root/.local/share/uv/python/cpython-3.13-linux-aarch64-gnu/bin
version_info = 3.13.13
```

**uv（0.11.15，与 ops 钉的版本一致）自动获取了托管解释器**，且与开发 Mac 上的 3.13.13 同版本——比 Pi 5 现状（系统 3.13.5 对 Mac 托管 3.13.13）更一致。

结论：**发行版不决定 Python 版本。** 在线安装路径零改动即可用。把 `cpython` 加入 ops 的 foundation artifact 表只对**离线 release bundle** 是必需项，不是阻断项。

## 5. 复现

NPU 基准（`librknnrt.so`、`rknn_api.h`、`mobilenet_v1.rknn` 取自 `airockchip/rknn-toolkit2`）：

```bash
cp librknnrt.so /usr/lib/ && ldconfig
gcc -O2 -o t t.c -I. -lrknnrt && ./t
```

内存带宽：

```bash
gcc -O2 -fopenmp -o stream stream.c
OMP_NUM_THREADS=8 ./stream
OMP_NUM_THREADS=4 taskset -c 4-7 ./stream
```

NVMe：

```bash
dd if=/dev/zero of=/var/tmp/t bs=1M count=2048 oflag=direct
sync; echo 3 > /proc/sys/vm/drop_caches
dd if=/var/tmp/t of=/dev/null bs=1M iflag=direct
```
