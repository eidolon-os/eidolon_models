# Orange Pi 5 Max（RK3588）实测档案

状态：`rk3588-verified` · 首测 2026-09-04 · 联合压测 2026-09-05

本文只记录**这一块板子**的实测数字。其他 Host 的对照在
[HARDWARE.md](HARDWARE.md)，验证方案与判废条件在 [VERIFICATION.md](VERIFICATION.md)。
推断与计划不写在这里。

**硬件**：RK3588，4×A76 @ 2256 MHz + 4×A55 @ 1800 MHz，16 GB LPDDR，
NVMe 256 GB，RKNPU2 三核（driver v0.9.8 / librknnrt 2.3.2）。

---

## 0. 当前结论速查

**这块板子能跑什么**（全部实测，链路细节见对应小节）：

| 负载 | 计算单元 | 实测 | 小节 |
| --- | --- | --- | --- |
| ASR funasr 2pass | CPU (ONNX int8) | 单路 rtf 0.20–0.26，说完到出文本 **292 ms** | §2.6 §2.16 §2.17 |
| LLM Qwen3-1.7B | NPU (RKLLM w8a8) | decode **12.5 tok/s**，前缀命中 TTFT **165 ms** | §2.9 §2.10 |
| TTS CosyVoice2 | NPU (RKLLM **两核** + RKNN 钉核 2) | 首包 **~2.0 s**，稳态 rtf **0.80–0.94**；27 s 回复**零断音** ✓ | **§2.21** |
| bge-small-zh | CPU (ONNX int8) | 查询 **3.2 ms**，写 10 条 fragment 51 ms | §2.14 |

**CPU/NPU 分配**（§2.17 实测得出）：

| 组件 | 分配 | 理由 |
| --- | --- | --- |
| Qwen3 (RKLLM) | **不传 cpu mask** | 默认自选 A76 4–7；强制 8 核掉一半 |
| CosyVoice2 | `--cpu-mask=0xE0`（A76 5–7）+ **NPU 核 2** | 上游配置：两核 RKLLM 占 NPU 0–1，encoder/flow/hift 钉 NPU 2（§2.21） |
| bge | 绑 **A76**，线程数=核数 | ORT 不感知 big.LITTLE |
| ASR | **不绑核** | 内核 EAS 会把 offline 突发放上大核 |
| 控制面 / vision | A55 0–3 | 剩余 |

**三条硬约束**：

1. **NPU 只能部分分区。** `rkllm.h` 没有 core 接口，但**核数在模型编译时烧定**
   （c2 占两核、c3 占三核）；RKNN 侧 `rknn_set_core_mask` 一直可用。TTS 内部据此
   分核后 rtf 从 >1 降到 0.80–0.94（§2.21）。但**跨模型仍然无空核可分**：
   Qwen3 烧的是三核，与 TTS 共存时 LLM 掉 90%（12.45 → 1.15–1.33 tok/s），
   而流式需要 3.7–5.3 tok/s。**本地 LLM 与本地 TTS 不能在这块板上同轮流式共存**（§2.22）。
2. **RKLLM 要求 `enabled_cpus_num >= npu_core_num`**（本模型为 3），给 2 个 CPU 直接
   `rkllm_init=-1`（§2.17 结论 ②）。
3. **A76 单核是 A55 单核的 5.06 倍**（bge int8 GEMM），主频只差 1.25 倍。
   把计算放上 A55 是实打实的数倍代价（§2.14）。

**当前最大的两个问题**：

* **端到端首音 ≈ 4.06 s**（§2.17 末），其中 TTS 首包 1998 ms 占 49%、
  LLM 生成首句 1604 ms 占 40%。**⚠️ 需按 §2.21 的新 TTS 配置重测**——TTS 首包
  基本不变，但 NPU 核占用变了，与本地 LLM 的争用条件随之改变。
* ~~CosyVoice2 稳态 rtf > 1~~ **已作废（§2.21）**：那是我用了三核 RKLLM 且没做 NPU
  核分配。改成上游配置（两核 RKLLM + encoder/flow/hift 钉 NPU 核 2）后
  **rtf 0.80–0.94，27 秒回复零断音，缓冲全程不降**。**本地 TTS 可行。**
  代价是 TTFT 涨约 260 ms。

---

## 2. 实测明细

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

> ⚠️ **2026-09-06 作废（见 §2.21）**：本节的 CosyVoice2 数字测于**我自己配错的配置**——
> 用了三核 RKLLM（`..._c3_...`）且没有做 NPU 核分配。上游基准用的是**两核 RKLLM
> 加上把 encoder/flow/hift 显式钉到 NPU 核 2**，两者必须一起。改对之后
> steady rtf 从 1.02–1.14 变成 **0.80–0.94**，27 秒回复零断音。
> **本节涉及 rtf、underrun、实时性、可用长度的一切结论均已作废。**


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
但 **`steady_pcm_rtf` 全部 > 1（1.06–1.13）——稳态生成追不上播放**。23.76 s 的音频要 27.5 s 生成，
晚 3.7 s 完成。

> **2026-09-06 更正（§2.20）**：上面"断流次数 3→11→23，约每秒音频断一次"是**误读**。
> 那一列是 `underrun_count`，源码里它数的是 `deadline_margin_ms < 0`，即"这一块相对于
> 前面已产出音频播完的时刻迟到了"——对 rtf 略大于 1 的产出者来说按构造就是几乎每块都中，
> 所以它必然≈音频秒数，**不是听得见的断音次数**。
> 听得见断音的条件是 `buffer_after_ms < 0`（= 迟到量超过一整块 1 s 音频）。
> 按这个口径复测，4 s 回复是 **0 次断音**（缓冲尚余 +400 ms），
> 而 24 s 回复是 **15–18 次**。**短回复其实是干净的，这一条此前被自己的读数掩盖了。**

**当前公开产物不具备任意长度的实时对话能力**，可用区间见 §2.20。

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

> ⚠️ **2026-09-06 部分作废（见 §2.21）**：本节"NPU 近似串行"的结论成立**只是因为
> 两侧都在抓 3 个核**——TTS 那边用的是三核 RKLLM。换成两核 RKLLM 并把 RKNN 钉到
> 核 2 之后，TTS 内部就不再串行了。**"NPU 无法分区"这句话对 RKLLM 成立
> （`rkllm.h` 确实没有核接口，但模型编译时就烧进了核数），对 RKNN 不成立
> （`rknn_set_core_mask` 可用）。跨模型的争用需要按新配置重测。**


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
| A76 4–5（2 核） | ~~0.705~~ → **0.281** | — | — |

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
**2026-09-05 重测已完成（同板、同代码、同音频，唯一变量是线程数）**：

| 绑核 | intra_op_threads | total_rtf（p50，三次） | 中位 |
| --- | ---: | --- | ---: |
| A76 4–5 | 4（强制，复现旧行为） | 0.711 / 0.679 / 0.708 | **0.708** |
| A76 4–5 | 2（修复后自动取值） | 0.285 / 0.281 / 0.277 | **0.281** |

旧行为复现出 0.708，与原记录 0.705 吻合；改用 `os.process_cpu_count()`
后 **2.52 倍提速**。超订确认为唯一主因。

无回归复测（这些配置本就不超订，线程数不变）：
A76 4–7 = 0.251 / 0.255（原 0.255）；A55 0–3 = 0.691 / 0.671（原 0.646）；
不绑核 = 0.226 / 0.255。

**衍生结论：A76×2（0.281）已逼近 A76×4（0.253），只用一半大核。**
若后续需要给 LLM/TTS 腾大核，ASR 绑 4–5 是有实测支撑的选项——
但这与 §核分配表「ASR 不绑核」的现行方案冲突，需另行决策，见
`VERIFICATION.md` §6。

#### ④ CPU governor 是所有 RTF 数据的隐藏变量（实测 2026-09-05）

板子出厂是 `ondemand`。A76 有 11 个频点，**408 MHz 到 2256 MHz，最高是最低
的 5.5 倍**（A55 是 408–1800，4.4 倍）。ondemand 按周期采样利用率决定升降频，
空闲掉到 408，负载来了要等下一个采样周期才爬上去。

ASR 跑动时每 100 ms 采一次 A76 频率，120 个样本：82 次满频 2256、32 次
600、6 次 408——**约 32% 的时间在 1/4 频以下**。offline 二遍是 ~300 ms 的
短突发，相当一部分就跑在爬频途中。

同场次对照（`scripts/asr-affinity-sweep -r 2`，其余条件相同）：

| 绑核 | `performance` | `ondemand` | performance 快 |
| --- | ---: | ---: | ---: |
| 不绑核 | **0.186** | 0.231 | 19% |
| 4–7 (A76×4) | 0.197 | 0.252 | **22%** |
| 4,5 (A76×2) | 0.258 | 0.281 | 8% |
| 0–3 (A55×4) | 0.675 | 0.714 | 5% |

**大核损失最大（19–22%），A55 只损失 5%**——A55 频率范围小，且绑满后基本
一直在高负载，不太掉下来。performance 下的**波动也明显收窄**（0.188/0.184
对 ondemand 的 0.267/0.237）。

**排序没有翻转。** 曾担心 ondemand 按利用率决策会系统性偏袒某种核配置、
使对比不可信；机制上成立，但实测两种 governor 下顺序都是
不绑核 < A76×4 < A76×2 < A55×4。**A76×2 vs 不绑核的结论不是 governor 假象。**

两条实践：

* **比较核配置时用 `performance`**，把频率钉死，差异才归因于核分配。
* **但 SLA 数字必须在生产实际用的 governor 下测**——爬频延迟在生产里真实
  存在，performance 下的 `eot_final` 会偏乐观。是否把生产也锁 performance
  是独立决定：要权衡功耗与温度（家用常驻无风扇设备），且 `schedutil`
  反应比 ondemand 快又仍省电，很可能是更合适的折中。**尚未实测。**

上表所有历史 RTF 数据都是 ondemand 下取的，跨场次比较时请记得这一点。

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

> **2026-09-06 更正（§2.20 [E]）**：TTS 这一半**未能复现**。各跑 5 次后
> 4–6 中位 0.993、4–7 中位 1.032，区间大幅重叠，运行间方差 ±7%——**两者不可区分**，
> 0.926 是分布的一次幸运抽样。"少给一个核反而更稳"**降级为未证实**。
> cpu7 留给 bge 与控制面的建议仍然成立，但理由是那边需要它，不是 TTS 用它更差。
> LLM 那一半（三核≈四核）不受影响。

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

> **2026-09-06 补充（T8，同场次内部对比）**：线程池修复（`os.process_cpu_count`）
> 之后，ASR@A76×2 从 0.705 更正为 0.281，于是"把 ASR 钉在两个大核、腾出另外两个"
> 成为一个此前因坏数据而被排除的候选。在 ASR 流式常驻（设备一直在听）的条件下
> 重测三种配置——绝对值受 §2.19 的调速器问题影响不可跨场次比较，但同场次排序可用：
>
> | ASR 位置 | LLM @512 | TTS steady_rtf |
> | --- | ---: | ---: |
> | A76 4–5（两个大核） | 6.65 | 1.312 |
> | 不绑核 | 6.53 | 1.260 |
> | **A55 0–3** | **9.71** | **1.256** |
>
> **把 ASR 钉在两个大核没有好处，是三者中最差的**：RKLLM 默认就占 A76 4–7，
> 钉在 4–5 等于和它正面重叠。这回答了 §2.16 遗留的那个疑问——不必等到部署后。
> 注意这三行都比 §2.17 正文的数字差，那是调速器状态不同，不是配置变差。

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
追不上实时播放。这不是争用造成的，是模型本身在这块板上的能力边界。

> **2026-09-06 更正（§2.20）**：括号里"空载也是 0.924，余量仅 8%"已撤回——
> 复现不出来（§2.20 [E]）。真实分布是中位 ~1.01、区间 0.976–1.094，**跨越 1.0 而非
> 稳定留有 8% 余量**。"三轮各有 3 次 underrun"也是误读，那是迟到块计数而非断音
> 次数（§2.20 [B]）；4 s 回复的真断音是 0 次。可用边界见 §2.20 [C]。

### 2.18 CosyVoice2 优化探底（实测 2026-09-05）

> ⚠️ **2026-09-06 作废（见 §2.21）**：本节的 CosyVoice2 数字测于**我自己配错的配置**——
> 用了三核 RKLLM（`..._c3_...`）且没有做 NPU 核分配。上游基准用的是**两核 RKLLM
> 加上把 encoder/flow/hift 显式钉到 NPU 核 2**，两者必须一起。改对之后
> steady rtf 从 1.02–1.14 变成 **0.80–0.94**，27 秒回复零断音。
> **本节涉及 rtf、underrun、实时性、可用长度的一切结论均已作废。**


在决定"是否先优化再整机部署"之前，把现有二进制里三个从未测过的开关和
回复长度的影响都测掉。同一 seed、同一文本、同绑核（A76 4–6）。

**[A] 三个开关全部让情况变差：**

| 配置 | TTFT | first_mel | steady rtf | underrun | 缓冲最低点 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基线 | 2059 ms | 1721 ms | **1.045** | 3 | +328 ms |
| `--hift-batch2` | 2228 | 1896 | 1.168 | 2 | +92 |
| `--serialize-downstream-npu` | 2147 | 1798 | 1.212 | 4 | **−229** |
| 两个都开 | 2015 | 1686 | 1.064 | 2 | +356 |

**现有二进制里没有剩余的免费收益。** `--serialize-downstream-npu` 甚至让缓冲见底。

**[B] 回复长度的影响：不成立，被运行间方差淹没。**

| `--max-tokens` | 实际 token | 音频 | steady rtf | 缓冲最低点 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 100 | 4.00 s | 1.005 | +572 ms |
| 200 | 130 | 5.20 s | 1.055 | +370 |
| 400 | 130 | 5.20 s | 1.054 | +378 |
| 800 | 130 | 5.20 s | 1.125 | +150 |
| 800 + `--hift-batch2` | 130 | 5.20 s | 1.097 | +195 |

200/400/800 三档都在 130 token 处采到 EOS，**输出完全相同**，rtf 却在
1.054–1.125 之间。所以**运行间方差约 ±7%**，此前"RTF 随回复变长而恶化"的
说法（比较的是文本不同的两次运行）**不成立，已撤回**。

**成立的是缓冲欠账随时长线性累积**：欠账 = (rtf − 1) × 时长。rtf 1.05 时
5 s 回复欠 0.25 s（缓冲仍余 +150~+572 ms），24 s 回复欠 1.2 s
（§2.12 那次 23.76 s 的运行实测 `minimum_buffer_after_ms = −1447`）。
**短回复扛得住，长回复会真的断流。**

**剩余的优化空间都要重新导出模型：**

| 项 | 现状 | 可能性 |
| --- | --- | --- |
| TTFT 1.9–2.1 s | `first_mel` 占 1470–1896 ms | **首块变小**（现为 chunk25 静态形状）是最大杠杆，估可省 350–400 ms；只降 TTFT，不降 RTF |
| flow_estimator 327 ms/s | NPU 占用 **91%** | 真·算力瓶颈，且已是 1 步蒸馏。只能换模型 |
| hift_decoder 269 ms/s | NPU 占用仅 24% | 数据搬运瓶颈，但零拷贝实测只快 7.5%（§2.12） |
| 自带 LLM 601 ms/s | 占总量 45% | Qwen2-0.5B w8a8，换更小的语音 LM 才动得了 |

NPU 侧 734 ms + 自带 LLM 601 ms = 理论 RTF 1.33，实测约 1.05——
**流水线已经吃掉了约 20% 的重叠收益**，靠调度再挤的空间不大。

**结论：现有产物下 CosyVoice2 就卡在实时线上，没有任何配置能改变这一点。**
剩下的杠杆全部需要重新导出模型，而其中最大的一个（首块变小）只改善首包、
不改善 RTF。

> **补充（§2.20）**：本节测量是在满频下做的（数值与 §2.20 的 performance 一列吻合），
> 所以 §2.19 发现的调速器杠杆**在这里早已拉满**，不是本节的遗漏。

### 2.19 CPU 调速器对 NPU 负载同样成立，幅度更大（实测 2026-09-06）

§2.17 那批测量之后板子重启过。重启后空载复测 Qwen3-1.7B，得到 **9.47 / 9.30 / 8.81 tok/s**，
而 §2.9 与 §2.17 记录的是 12.94 / 12.41 / 11.52。两次复现，差 27%。
在解释清楚之前，当天基于这块板的联合测量（T8）全部作废——本节是排查过程与结论。

**排除的假设**（都实测过，不是推断）：

| 假设 | 实测 | 结论 |
| --- | --- | --- |
| 有残留进程抢资源 | `ps` 无、NPU 0%、load 主要来自 GNOME 会话 | 否 |
| 过热降频 | 44–51 °C 全程 | 否 |
| NPU 掉频 | `fdab0000.npu` 恒在 1 GHz（上限） | 否 |
| CPU 最大频率被改 | 1800 / 2256 MHz，未变 | 否 |
| 运行时或模型被换 | `librkllmrt.so` 与 `.rkllm` 时间戳、大小、md5 均未变 | 否 |
| **DDR 掉到最低频** | `dmc` 停在 534 MHz（上限 2400） | **否，见下** |
| **CPU 调速器** | 见下表 | **是** |

**DDR 频率不影响它。** 把 `dmc` governor 设成 `performance`（534 → 2400 MHz）后重测：
9.56 / 9.28，与 534 MHz 下的 9.46 / 9.33 没有区别。而且整个 LLM 运行期间
`dmc_ondemand` 根本没有升频——**Qwen3-1.7B 的 decode 不受 DDR 频率约束**。
这一条软化了 §2.3 里"内存带宽是本地 LLM 的真瓶颈"的说法：534 MHz 的 DDR
已经够喂 9.3 tok/s。（`decode ≈ 带宽/权重字节` 作为上界仍然成立，只是这块板上
没有触到那个上界。）

**是 CPU 调速器。** 同一次会话内背靠背，空载：

| 条件 | decode @128 | @512 |
| --- | ---: | ---: |
| ondemand（板子默认） | 9.84 | 9.32 |
| **performance** | **12.98** | **12.53** |
| ondemand + 4 个满载占核进程 | 8.92 | 8.27 |

`performance` 下正好回到此前记录的 12.94 / 12.41。**ondemand 对 LLM decode 的
代价是 32%**，比 §2.16 结论④里 ASR 上的 19–22% 更大。运行中采样 A76 频率，大部分时间
在 1200 MHz 而非 2256——decode 是 NPU 密集的，CPU 利用率不足以让 ondemand 升频，
但 RKLLM 的宿主线程仍然吃这个亏。

第三行否掉了一个顺手的解释：并不是"有并发负载把大核顶住所以更快"，满载占核
让它更差。

**仍未解释**：2026-09-05 的测量是在 ondemand 下做的，却得到 12.66–13.74。
唯一说得通的是那些 llmbench 都紧接在其他负载之后运行、大核仍在爬坡状态，
但这没有被验证，**不当作结论**。

**后果**：

* §2.9、§2.15、§2.17 的**绝对值**都带有未记录的调速器状态，不能当作 SLA。
* 同一次会话内的**排序**仍然可用——这与 §2.16 结论④在 ASR 上的结论一致（换调速器
  不改变排序）。所以 §2.17 的分配方案结论不受影响，它比较的是同场次的几个配置。
* **要不要出厂就用 performance，是一个功耗与散热决策**，不是性能决策：
  这是一台无风扇常开设备。`schedutil` 是未测的中间选项。

### 2.20 本地 TTS 可行性定论（实测 2026-09-06）

> ⚠️ **2026-09-06 作废（见 §2.21）**：本节的 CosyVoice2 数字测于**我自己配错的配置**——
> 用了三核 RKLLM（`..._c3_...`）且没有做 NPU 核分配。上游基准用的是**两核 RKLLM
> 加上把 encoder/flow/hift 显式钉到 NPU 核 2**，两者必须一起。改对之后
> steady rtf 从 1.02–1.14 变成 **0.80–0.94**，27 秒回复零断音。
> **本节涉及 rtf、underrun、实时性、可用长度的一切结论均已作废。**


§2.19 发现调速器对 NPU 负载有 32% 影响，而 §2.11/2.12/2.18 全部测于其前。本节
先补掉这个变量，再回答真正的产品问题：**本地 TTS 到底能不能用，边界在哪。**

#### [A] 调速器：对 TTS 同样成立，但杠杆早已拉满

同场次交替三轮，同 seed、同绑核（A76 4–6）、同文本：

| governor | A76 频率 | TTFT | steady rtf | 缓冲最低点 |
| --- | ---: | ---: | ---: | ---: |
| **performance** | 2256 MHz | 1804–1886 ms | **1.020–1.074** | **+237 ~ +426 ms** |
| ondemand | 1184–1555 MHz | 2351–2524 ms | **1.299–1.339** | **−494 ~ −593 ms** |

`ondemand` 的代价是 rtf **+26%**、TTFT **+30%**，且缓冲直接转负。

**但 §2.18 的数字（rtf 1.045 / 缓冲 +328）落在 performance 这一列**，说明那批测量本来
就在满频下。**调速器不是被漏掉的杠杆，它已经拉满了，而 rtf 仍然 > 1。**
出厂用 performance 由 foundation profile 保证；本次实测满频满载温度 44–50 °C，
风扇 pwm 为 0（未转），**性能档在这块板上是热免费的**。

#### [B] 先修一个读数错误：`underrun_count` 不是断音次数

源码 `cosyvoice2_streaming_pipeline.cpp:1607`：

```cpp
event.deadline_margin_ms = deadline - event.ready_ms;
event.underrun = event.deadline_margin_ms < 0.0;      // 这一块迟到了
event.buffer_after_ms = event.deadline_margin_ms + 本块时长;
```

`deadline` 是"前面已产出的音频播完的时刻"。所以 `underrun` 数的是**每一块有没有迟到**，
对 rtf 略大于 1 的产出者按构造几乎每块都中——它必然 ≈ 音频秒数，
**这就是 §2.12 那个"约每秒断一次"的来源，是误读**。

**一块 = 1 s 音频，所以迟到 600 ms 仍余 +400 ms 缓冲。听得见断音的条件是
`buffer_after_ms < 0`。** 下面全部按这个口径计数。

#### [C] 可用边界：一轮 ≤ ~7 秒音频

performance 满频，真实文本经文本前端合成（`--max-context=2048` 必须给，默认太小），
每档 3 个 seed：

| 字数 | 音频 | steady rtf | **真断音块** | 缓冲最低点 | 判定 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 18 | 4.08–4.20 s | 1.024–1.143 | **0 / 0 / 0** | +185 ~ +565 ms | 干净 |
| 26 | 4.84–5.60 s | 1.042–1.086 | **0 / 0 / 0** | +307 ~ +428 ms | 干净 |
| 39 | 7.36–8.44 s | 1.069–1.086 | **0 / 1 / 0** | −53 ~ +67 ms | 临界 |
| 50 | 9.08–10.36 s | 1.063–1.139 | **5 / 1 / 4** | −411 ~ −34 ms | 断音 |
| 121 | 23.56–24.32 s | 1.114–1.128 | **15 / 18** | −2171 ~ −2378 ms | 崩 |

**≤26 字（约 5 s 音频）稳定零断音；39 字（~7.5 s）临界；50 字起必然断音。**
欠账 = (rtf−1) × 时长，初始缓冲只有约一块（1 s）且被 TTFT 吃掉一部分，
所以崩溃点落在 7–9 s 而不是理论上的 14 s。

#### [D] 分句不能规避，实测更糟

把 121 字那段切成 5 个短句逐句合成（进程内计时，不含 3 s 预热）：

| 句 | 字数 | 音频 | 产出耗时 | 该句 steady rtf | 真断音 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 18 | 4.16 s | 5.60 s | 1.116 | 0 |
| 2 | 11 | 2.24 s | 3.21 s | 0.917 | 0 |
| 3 | 12 | 3.00 s | 3.97 s | 0.940 | 0 |
| 4 | 9 | 1.64 s | 2.54 s | 0.739 | 0 |
| 5 | 11 | 2.12 s | 3.08 s | 0.978 | 0 |
| **合计** | 61 | **13.16 s** | **18.39 s** | **实效 1.397** | 0 |

对照单次长文本：24.32 s 音频 / ~27.4 s 产出 = **1.127**。

**分句把实效 rtf 从 1.127 推到 1.397。** 每句要付 ~1.9 s 首包，5 句就是 9.5 s 死时间，
占了 18.39 s 的一半以上。单句的 `steady rtf` 反而 < 1（0.74–0.98）是因为 2–3 块的
窗口太短、且 precompute 已在 TTFT 里付过，那个数不能外推。

**流水化也救不了**：TTFT 那 1.9 s 几乎全是 NPU 忙时间（precompute 780 ms + RKLLM 出首
token + flow encoder/estimator + hift 首块），不是空闲。§2.15 已实测 NPU 近似串行，
**每秒音频的 NPU 工作量是不变量，调度改不了它。**

#### [E] 顺带纠正一个自己的数字：0.926 未能复现

§2.17 结论③记的是"绑 4–6（三核）rtf **0.926**，绑 4–7（四核）1.015–1.036，少给一个核
反而更稳"。今天在满频下把这两种绑核各跑 5 次（`--sample`，同 fixture prefill）：

| 绑核 | n | 中位 | 均值 | 区间 |
| --- | ---: | ---: | ---: | ---: |
| A76 4–6 | 5 | 0.993 | 1.022 | 0.976–1.094 |
| A76 4–7 | 5 | 1.032 | 1.032 | 0.995–1.084 |

**两者不可区分**：中位差 4%，区间大幅重叠，而运行间方差本就 ±7%（§2.18 [B]）。
另外把 `teacher_replay` 与 `--sample` 各跑一遍也不解释它——8 次运行全在 1.014–1.087。

**真正的结论是：这块板上 CosyVoice2 的 steady rtf 分布跨越 1.0**（n=10，中位 ~1.01，
区间 0.976–1.094）。§2.17 那个 0.926、以及 §0 曾写的"空载 0.924，余量 8%"，
**是这个分布的一次幸运抽样，不是三核的收益，已降级。**

这不改变 [C] 的边界结论——[C] 数的是真断音，而偶尔一次 rtf < 1 不足以抵消
长回复上累积的欠账：只要有一次抽到 1.09，那一轮就断。

#### 定论

**当前公开产物的 CosyVoice2，在这块板上不能支撑任意长度的语音回复。**

| 结论 | 依据 |
| --- | --- |
| 干净可用区间：**一轮回复 ≤ ~7 s 音频（≤ ~39 字，稳妥取 ≤26 字）** | [C] |
| 超出即断音，欠账随时长线性累积 | [C]，24 s 回复欠 2.2–2.4 s |
| 分句 / 流水**不能**规避 | [D]，实测反而更差 |
| 调速器杠杆已拉满，无热代价 | [A] |
| 现有二进制无剩余开关 | §2.18 [A]，三个开关全是负优化 |

**唯一还没试过的 rtf 杠杆是重新导出模型**，而 §2.11 的分解指出方向：自带的
Qwen2-0.5B 占总 NPU 时间 **45%**（601 ms/s 音频），flow_estimator 已是 1 步蒸馏
且 NPU 占用 91%（真算力瓶颈），hift 零拷贝已证伪（只快 7.5%）。
**换更小的语音 LM 是唯一动得了 rtf 的地方**；§2.18 提到的"首块变小"只改首包。

产品侧的取舍（不在本文件决定）：约束回复长度 / 混合云端 / 换 TTS 模型 / 重导出。
Kokoro 已证否（§2.13，rtf 2.15–2.40，且无零样本克隆）。

### 2.21 本地 TTS 其实可行——此前是我配错了（实测 2026-09-06）

**结论先说：CosyVoice2 在这块板上 steady rtf 0.80–0.94，27 秒回复零断音，缓冲全程
不下降。本地 TTS 可行。§2.11/2.12/2.15/2.17/2.18/2.20 里所有关于 rtf 与实时性的
结论都作废，原因是我用错了配置。**

#### 怎么发现的

上游 README 公布的是**已发布产物**的数字（不是那份"心心过 100 才开源"的优化版）：

| 音频 | TTFT | Streaming RTF |
| ---: | ---: | ---: |
| 5.92 s | 2.255 s | **0.746** |
| 9.64 s | 2.090 s | 0.846 |
| 24.28 s | 2.334 s | 0.913 |
| 75.11 s | 2.165 s | 0.918 |

我此前记的"0.6–0.8 是未开源版"是错的。而我实测 1.02–1.14，**TTFT 却比他还快**——
首包更快而稳态更慢，说明我每一块多干了活，不是启动多花了时间。这个信号指向配置。

读他的 `rknn/scripts/run_cosyvoice2_streaming_rk3588.sh`，差异是：

| 项 | 上游 | 我 |
| --- | --- | --- |
| RKLLM | **`qwen2_body_w8a8_c2_ctx2048.rkllm`（两核）** | `..._c3_...`（三核） |
| `--head-core` | **0** | 未传 |
| `--encoder-core` / `--flow-core` / `--hift-core` | **全 = 2** | 未传 |
| `--cpu-mask` | 0xE0 | 未传（用 taskset） |
| `--threads` | 4 | 默认 |
| `--token-queue` / `--mel-queue` | 64 / 2 | 默认 |
| 音色 | `voices/testwav_llm_prompt1s` | `testwav_prompt3s` |
| `--precompute-full-prompt` | 不传 | 一直传 |

#### 机制：两核模型和核分配必须一起改

`ParseCore` 取的是核编号，所以上游是把 **encoder + flow + hift（91 + 327 + 269
= 687 ms/秒音频，§2.11）全钉在 NPU 核 2**，而两核 RKLLM 只占核 0–1，**永远不碰核 2**。
speech head 留在核 0，它只有 1.8 ms × 25 次。

这也解释了 §2.11 那条"RKNN 部分只用得动一个 NPU 核"——在正确配置下**那是资产
不是缺陷**：RKNN 要 1 核、RKLLM 要 2 核，正好凑满三核。

**两者必须一起改。** 我先单独换了 c2（核分配仍是默认），结果**更差**：

| 配置 | n | rtf 中位 | 区间 |
| --- | ---: | ---: | ---: |
| c3 + taskset 4-6（我原配置） | 5 | 1.010 | 0.984–1.067 |
| **c2，核分配默认** | 5 | **1.145** | 1.126–1.236 |
| **c2，核分配默认，绑 5-7** | 5 | **1.380** | 1.326–1.422 |

只换模型等于只吃了 LLM 变慢的代价（两核 decode 比三核慢）而没拿到并行。

#### 结果

同一段文本、同音色、同为 5.92 s 音频，seed 1–5：

| 配置 | rtf 中位 | 区间 | TTFT | 最低缓冲 |
| --- | ---: | ---: | ---: | ---: |
| 我: c3 + precompute + ts 4-6 | 1.077 | 1.042–1.120 | 1731 ms | +156 ms |
| **上游: c2 + 核分配 + 0xE0** | **0.856** | 0.828–0.874 | 1970 ms | **+840 ms** |
| **上游 + precompute** | **0.840** | 0.754–0.906 | 1994 ms | **+840 ms** |

区间不重叠。**rtf 降 22%，代价是 TTFT 涨 ~260 ms**（两核 prefill 慢）。
对连续说话，rtf 比 TTFT 重要得多。

长度扫描（上游配置 + precompute，每档 3 个 seed）：

| 字数 | 音频 | rtf | **真断音** | 最低缓冲 | 温度 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 18 | 5.76–5.96 s | 0.804–0.823 | **0** | **+840 ms** | 53–56 °C |
| 50 | 10.56–11.80 s | 0.872–0.894 | **0** | **+840 ms** | 56–57 °C |
| 121 | 23.00–27.28 s | 0.905–0.937 | **0** | **+840 ms** | 59–61 °C |

**缓冲全程停在 +840 ms（= 一整块），一次都没有下降**——产出比播放快，欠账根本不累积。
与上游公布值吻合（他 24.28 s 报 0.913，我 23–27 s 得 0.905–0.937）。

**不是靠产出垃圾换来的**：121 字那段两种配置的输出，RMS 4567 vs 4669、同峰值、
近静音占比 29.7% vs 31.3%，声学统计一致。

持续负载下 61 °C（风扇在位但 pwm=0，未转）。

#### 作废了什么

| 此前结论 | 出处 | 现状 |
| --- | --- | --- |
| steady rtf > 1，追不上播放 | §0 §2.12 §2.17 §2.18 §2.20 | **作废**，实测 0.80–0.94 |
| 可用上限一轮 ≤ ~7 s 音频 | §2.20 [C] | **作废**，27 s 零断音 |
| 分句更糟（实效 1.397） | §2.20 [D] | **作废**，前提是 rtf > 1；现在不需要分句 |
| "现有产物就卡在实时线上，没有配置能改变" | §2.18 | **作废**，正是配置问题 |
| NPU 近似串行 | §2.15 | **部分作废**，那是三核模型的性质 |
| 端到端首音 4.06 s | §2.17 | **需重测**，TTS 首包从 1998 → ~1990 ms 但 LLM 侧争用条件变了 |
| "0.6–0.8 是未开源的优化版" | §2.11 | **改正**，那是已发布产物的公布值 |

**教训**：§2.18 我写了"现有二进制里没有剩余的免费收益"，那句话的依据是我试了
三个开关。真正的收益不在开关里，在**我从未读过上游的运行脚本**——我自己拼了
一套命令行，然后把它的性能当成了模型的性能边界。

### 2.22 LLM × TTS 共存重测（实测 2026-09-06）

§2.21 改对 TTS 配置后，§2.15 那次共存测量的前提变了（那次两侧都在抓三核），
重测。结论：**争用从对称变成单向——TTS 几乎不受影响，本地 LLM 被压到不可用。
本地 LLM 与本地 TTS 在这块板上不能同轮流式共存，差 2.8–4.6 倍。**

#### 实测

TTS 用 §2.21 的上游配置（`--cpu-mask=0xE0 --cpu-count=3`，NPU 核 2 承担
encoder/flow/hift），文本 121 字 → 23 s 音频；Qwen3-1.7B 用 llmbench 默认自选 A76 4–7。
先跑 6 秒让 LLM 进入 decode，再启动 TTS。三轮：

| 轮次 | TTS 单跑 rtf | TTS 共存 rtf | 真断音 | 缓冲最低 | Qwen3 ctx512 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.942 | 0.945 | 0 | 840 ms | **1.23** tok/s |
| 2 | 0.919 | 0.939 | 0 | 840 ms | **1.33** tok/s |
| 3 | 0.925 | 0.966 | 0 | 804 ms | **1.15** tok/s |

Qwen3 单跑基线 12.45 tok/s（同场次实测，与 §2.9 记录的 12.41 吻合）。
**TTS 代价 +2~4% 且仍零断音；LLM 代价 −90%。**

共存期间 NPU 逐核占用：Core0 均值 56%、Core1 56%、Core2 53%（峰值 82–85%）。
三个核都在忙，但吞吐几乎全归 TTS。

#### 争的是 NPU，不是 CPU

给 TTS 只两个 CPU（`--cpu-mask=0xC0 --cpu-count=2`，注意不给 `--cpu-count`
会因默认 3 与两核 mask 不匹配而 `rkllm_init=-1`），再分别试 CPU 完全不重叠与部分重叠：

| 场景 | TTS rtf | 真断音 | 缓冲最低 | Qwen3 ctx512 |
| --- | ---: | ---: | ---: | ---: |
| TTS 单跑（2 CPU） | 0.944 | 0 | +840 ms | 8.66（LLM 在 3,4,5 单跑） |
| 共存 · CPU **完全不重叠**（TTS 6-7 / LLM 3,4,5） | **1.120** | **16** | **−2232 ms** | 3.77 |
| 共存 · CPU 部分重叠（TTS 6-7 / LLM 默认 4-7） | 1.123 | 18 | −2534 ms | 4.60 |

**完全不重叠与部分重叠结果一样**（1.120 vs 1.123，断音 16 vs 18）——所以瓶颈是 NPU。
另外可见：TTS 拿 3 个 CPU 时共存毫发无损，只拿 2 个时自己也崩。**谁拿到的 CPU 多谁赢，
但两者都要 NPU，总量不够。**

无干净分区可言：Qwen3 烧的是 `npu_core_num: 3`，TTS 的 c2 占 NPU 0–1、RKNN 钉 NPU 2，
三个核已被占满，没有空核可分。

#### 对流式对话意味着什么

从 §2.21 的实测反推 TTS 的文本消耗速率：121 字 → 23.0 s 音频 = **每秒音频 5.26 字**。
按 Qwen3 中文 0.7–1.0 text token/字：

| | 需要 | 实际 |
| --- | ---: | ---: |
| TTS 说话时的文本消耗 | **3.7–5.3 tok/s** | — |
| LLM 单跑 | — | 12.45 tok/s（3 倍余量 ✓） |
| LLM 与 TTS 共存 | — | **1.15–1.33 tok/s** |

**缺口 2.8–4.6 倍。** 「LLM 生成下一句、TTS 同时说上一句」这个流式模式在这块板上
不成立——不是差一点，是差 3 倍以上，TTS 会立刻等文本。

改成"先整段生成完再说"也不行：50 字回复约 40 token，单跑 12.45 tok/s 要 3.2 s，
加 ASR 250 ms 与 TTS 首包 2.1 s，首音约 5.6 s。

#### 可行的组合（架构决策，不在本文件定）

| 方案 | NPU 占用 | 状态 |
| --- | --- | --- |
| **本地 TTS + 云端 LLM** | TTS 独占 | ✓ 实测可用，rtf 0.80–0.94 零断音。也是当前产品配置 |
| 本地 LLM + 云端 TTS | LLM 独占 | ✓ LLM 12.45 tok/s |
| **两者都本地，但分两台 host** | 各自独占 | ✓ 符合 Eidolon OS 任意组合架构，且是完全本地化 |
| 两者都在这块板上同轮流式 | 抢满 | ✗ 差 2.8–4.6 倍 |

ASR 是纯 CPU（ONNX int8），与 TTS 不争 NPU；ASR + TTS + 云端 LLM 这条链路
**尚未端到端测过**，是下一项。

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
