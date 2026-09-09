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
| LLM Qwen3-1.7B | **CPU A55×4 (Q4_0/llama.cpp)** | decode **6.07** tok/s（与 TTS 共存 5.85，门槛 3.43）✓ | **§2.23** |
| 同上，若独占 A76 | CPU A76×4 (Q4_0) | decode **18.94** tok/s——比 NPU 快 52%，但会打断 TTS | §2.23 |
| 同上，NPU 方案 | NPU (RKLLM w8a8) | decode 12.5 tok/s；与 TTS 共存掉到 1.15–1.33 ✗ | §2.9 §2.22 |
| TTS CosyVoice2 | NPU (RKLLM **两核** + RKNN 钉核 2) | 首包 **~2.0 s**，稳态 rtf **0.80–0.94**；27 s 回复**零断音** ✓；常驻服务调用方首音稳态 **2.0–2.2 s** | **§2.21** §2.26 |
| bge-small-zh | CPU (ONNX int8) | 查询 **3.2 ms**，写 10 条 fragment 51 ms | §2.14 |

**CPU/NPU 分配**（§2.17 实测得出）：

| 组件 | 分配 | 理由 |
| --- | --- | --- |
| Qwen3 (RKLLM) | **不传 cpu mask** | 默认自选 A76 4–7；强制 8 核掉一半 |
| CosyVoice2 TTS | `--cpu-mask=0xE0`（A76 5–7）+ **NPU 全部** | 上游配置：两核 RKLLM 占 NPU 0–1，encoder/flow/hift 钉 NPU 2（§2.21） |
| **聊天 LLM** | **A55 0–3，llama.cpp Q4_0，不碰 NPU** | 放 A76 会打断 TTS，即使只给一核（§2.23） |
| bge | 绑 **A76**，线程数=核数 | ORT 不感知 big.LITTLE |
| ASR | **不绑核** | 内核 EAS 会把 offline 突发放上大核 |
| 控制面 / vision | A55 0–3 | 剩余 |

**三条硬约束**：

1. **NPU 只能部分分区，所以聊天 LLM 不放 NPU。** `rkllm.h` 没有 core 接口，但
   **核数在模型编译时烧定**（c2 占两核、c3 占三核）；RKNN 侧 `rknn_set_core_mask`
   一直可用。TTS 内部据此分核后 rtf 从 >1 降到 0.80–0.94（§2.21）。但**跨模型无空核
   可分**：两个 RKLLM 都从 core 0 开始占，Qwen3 与 TTS 共存时 LLM 掉 90%（§2.22）。
   **解法是把聊天 LLM 整个搬到 CPU 小核**：Qwen3-1.7B Q4_0 / llama.cpp 在 A55×4 上
   5.85 tok/s（门槛 3.43），TTS 零断音。**单机全本地成立**（§2.23）。
   注意必须是 A55——留在 A76 上即使只给一个核也会把 TTS 打断。
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

### 2.23 单机全本地成立：聊天 LLM 下 CPU 小核，TTS 独占 NPU（实测 2026-09-06）

§2.22 的结论是"本地 LLM 与本地 TTS 不能同轮流式共存"，前提是**两者都在 NPU 上**。
把聊天 LLM 整个搬出 NPU 之后这个前提消失了。**单机全本地成立。**

#### 分配方案

| 资源 | 归谁 | 实测 |
| --- | --- | --- |
| **NPU 三核 + A76 5–7** | CosyVoice2 TTS（c2 RKLLM 占 NPU 0–1，RKNN 钉 NPU 2） | rtf **0.932–0.945**，**零断音**，缓冲全程 840 ms |
| **A55 0–3** | 聊天 LLM：Qwen3-1.7B **Q4_0 / llama.cpp** | 单跑 6.07–6.10，共存 **5.74–5.85 tok/s** |
| A76 4 | ASR / bge / 控制面 | ⚠️ **没有依据**，§2.24 实测后撤回：一个核装不下 ASR + pVAD + EOT + bge + 十几个服务 |

**门槛是实测的，不是估的**：121 字文本经 Qwen3 分词是 **79 token**（1.53 字/token），
对应 23.0 s 音频，所以「LLM 供得上 TTS 说」的门槛 = **3.43 tok/s**。
共存实测 5.77 → **余量 1.68 倍**。

#### 意外发现一：CPU 上跑这个模型比 NPU 快

| 绑核 | 线程 | prefill | decode |
| --- | ---: | ---: | ---: |
| A76 4–7 | 4 | 93.6 | **18.94** tok/s |
| A76 4–6 | 3 | 65.9 | 18.20 |
| A76 4–5 | 2 | 46.3 | 14.54 |
| 全 8 核 | 8 | 84.4 | 13.76 |
| A55 0–3 | 4 | — | 6.07 |
| A55 0–1 | 2 | — | 3.23 |

**A76 四核 18.94 tok/s，比 NPU 上 RKLLM w8a8 的 12.45 快 52%。** 原因是 decode 受内存
带宽约束而 Q4_0 是 4 bit——权重字节只有 W8A8 的一半。代价是量化更狠，质量差异未评估。

（llama.cpp 从源码在板上构建，cmake 4.2.3 + gcc 15.2，`-DGGML_NATIVE=ON`。
A76 有 `asimddp` 无 `i8mm`，dotprod 是 Q4_0 的主路径。）

#### 意外发现二：必须放 A55，不是"放到 NPU 之外"就够

把 LLM 移出 NPU 但留在 A76 上，**TTS 照样断**：

| LLM 配置 | LLM 单跑 | LLM 共存 | TTS |
| --- | ---: | ---: | --- |
| A76 ×1（t1） | 8.15 | 7.39 | rtf 1.099，**断音 12**，缓冲 −1843 ms ✗ |
| A76 ×2（t2） | 14.78 | 9.62 | rtf 1.182，**断音 21**，缓冲 −3906 ms ✗ |
| A76 4,5 / TTS 在 6,7（CPU 完全不重叠） | 14.33 | 9.99–10.15 | rtf 1.143–1.166，断音 18–21 ✗ |
| **A55 ×4（t4）** | 6.07 | 5.85 | **rtf 0.945，断音 0**，缓冲 +791 ms ✓ |
| **A55 ×2（t2）** | 3.23 | 3.24 | **rtf 0.875，断音 0**，缓冲 +840 ms ✓ |

**A76 单核只有 8.15 tok/s 却仍然打断 TTS，A55 四核 6.07 tok/s 完全不打断。**
流量只差 34%，解释不了 12 次断音与 0 次的差别——**是核簇，不是速率**。
TTS 自己的 CPU 线程在 A76 5–7，同簇的 LLM 与它争 A76 的 L2/L3 与该簇到互连的通路；
A55 簇走自己的路径。

**DDR 频率不是杠杆**（又一次）：`dmc` 从 534 MHz 提到 2400 MHz，A76 上那两组数字
逐项不变（1.154/1.173 → 1.166/1.143，断音 21/17 → 21/18）。这与 §2.19 在 NPU LLM 上
的发现一致——瓶颈不是 DDR 吞吐。

#### 边界与未验

* A55 ×2 只有 3.23 tok/s，**低于 3.43 门槛**——四个小核都要给它。
* 余量 1.68 倍是按"回复 121 字 / 23 s 音频"这一档测的。语速更快的音色或更密的文本
  会抬高门槛，需要按真实音色重算。
* Q4_0 对中文对话质量的影响**未评估**。NPU 那边是 W8A8。
* ASR（纯 CPU）与 bge 还挤在 A76 4 上，**这条链路没有端到端测过**。§2.24 把 channel
  的两个模型加进来后 TTS 已经从 0.92 涨到 0.99–1.01，**余量基本用完**，而 ASR、nats、
  livekit、agent、各 API、bge、GNOME 都还没算。**"单机全本地"在完整系统下未验证。**
* 未测 30 分钟以上持续负载的温度（本节单轮 121 字后 59–61 °C）。

### 2.24 把 Eidolon OS 自己算进去：余量几乎被吃光，且"每个 unit 绑核"是错的（实测 2026-09-06）

§2.23 只考虑了三个模型。这一节把 Eidolon OS 自身算进来，结论是**全本地仍然成立，
但余量从 8% 降到 0–1%，而且系统还有一大半没计入**。同时推翻了我自己上一步想做的
"给每个 unit 加 CPUAffinity"。

#### Eidolon OS 空载开销

当前在跑的 9 个服务合计约 **4% CPU / 649 MB RSS**（eidolond 1.4%、hub 1.5%，其余 ≤0.4%）。
板上另有 **9 个 GNOME 进程**、共 281 个进程。**所有 eidolon 服务的亲和性都是 `0-7`——
`deploy/systemd/*.service` 里没有任何一个设了 `CPUAffinity` / `CPUWeight` / `Nice`。**

#### channel 的三个模型：首次实测

| 模型 | 大小 | 调用节奏 | 1 线程 | 2 线程 | 4 线程 |
| --- | ---: | --- | ---: | ---: | ---: |
| **pVAD** | 3.8 MB | **每 10 ms 一帧** | **1.671 ms** | 1.395 | 1.234 |
| **EOT 中文 q8** seq=16 | 169.9 MB | 每次轮次判断 | 31.3 ms | 19.0 | 13.1 |
| 同上 seq=32 | | | **58.4 ms** | 34.0 | 21.4 |
| 同上 seq=64 | | | **125.9 ms** | 67.2 | 38.4 |
| 同上 seq=128 | | | 262.1 ms | 138.2 | 73.6 |

* **pVAD 单线程 1.671 ms / 10 ms 帧 = 持续占一个 A76 核的 16.7%**，整轮不间断。
  多线程反而更费总 CPU（1.234 ms × 4 核），**VAD 应当单线程**。
* **EOT 随序列线性增长**，seq=32 单线程 58 ms。若每 300 ms 判一次，是一个核的 19%；
  seq=64 时 42%。
* CAMPPlus（28 MB，ModelScope `.bin`）**未测**——它每段话一次，不是热路径。

#### 真实一轮：TTS + 聊天 LLM + channel 负载

TTS 用 §2.21 配置（NPU + A76 5–7），聊天 LLM 在 A55 0–3，channel 负载按真实占空比
复现（pVAD 100 Hz + EOT 2 Hz，单线程，用板上实际的 onnx 与 onnxruntime 1.26）。

| channel 绑核 | TTS rtf | 真断音 | 缓冲最低 | LLM | VAD 完成帧 |
| --- | ---: | ---: | ---: | ---: | --- |
| **不绑核**（产品现状） | **0.988–1.005** | **0** ✓ | 492–840 ms | 5.77 | 4500/4500 |
| 绑 A76 4 | 1.138–1.143 | **17** ✗ | −2885 ms | 5.76 | 4500/4500 |
| 绑 A76 4–5 | 1.033 | **3** ✗ | −217 ms | 5.71 | 4500/4500 |
| 绑 A55+A76 4（0–4） | 1.111 | **13** ✗ | −2214 ms | 5.74 | 4500/4500 |
| 绑 A55 0–3 | 0.921 | 0 | 840 ms | 5.47 | **1465/4500** ✗ |

五轮不绑核全部零断音（rtf 0.981/0.988/0.989/0.996/1.005）。持续负载后 **64.7 °C**。

#### 三条结论

**① "每个 unit 声明自己的核簇"是错的。** 每一种把 channel 钉在大核上的分配都把 TTS
打断（3–17 次），唯独**不绑核不会**——内核 EAS 把这个 100 Hz 的轻负载摊薄到八个核上并
很快让核空闲，而钉住会把它压成同簇的持续干扰。**正确的规则窄得多：只钉 TTS
（`--cpu-mask=0xE0`）和聊天 LLM（A55），其余一律不绑。**

**② channel 不能放小核。** 绑 A55 时 VAD 只跑完 4500 帧里的 1465——丢了 2/3。
A55 单核约 8.4 ms/帧，追不上 10 ms 的帧率。**打断与轮次检测会直接坏掉**，
而 TTS 的那一栏（0.921 / 零断音）会让人误以为这个配置是好的。

**③ 余量几乎没了。** TTS 从单跑 **0.92** 涨到 **0.988–1.005**，缓冲从 840 ms 掉到
最低 492 ms。23 秒回复仍然零断音，但**已经贴在 1.0 上**。

#### 还没计入的（这是当前最大的未知）

上表只加了 channel 的两个模型。真实一轮还有：**ASR funasr 2pass**（CPU 密集，
§2.6 单路 rtf 0.20–0.26）、nats、livekit-server（WebRTC/SFU）、agent、
kernel/hub/data/data-workspace API、memory embedder（bge）、eidolond 每 5 秒对账、
CAMPPlus，以及 9 个 GNOME 进程。**在 TTS 已经贴在 1.0 的情况下，这些都还没被算进去。**

所以 §2.23 那句"A76 4 给 ASR / bge / 控制面"**是没有依据的**——一个 A76 核装不下
ASR 加 pVAD 加 EOT 加 bge 加十几个服务。**"单机全本地"在完整系统下是否成立，
目前没有测过，不能当作结论。**

产品侧还有一条：**出厂镜像不该跑 GNOME。**

### 2.25 把 ASR 加进来：最坏一轮仍然过关，但余量只剩 257–703 ms（实测 2026-09-06）

§2.24 只加了 channel 的两个模型。这一节把 ASR 也加进来，测**最坏的那一轮**：
TTS 正在说话时用户反复打断——pVAD 逐帧、EOT 判轮次、ASR 二遍突发、聊天 LLM 同时在生成。

#### ASR 两个 pass 的实测成本（首次）

板上模型：streaming `model_quant.onnx` 166 MB + `decoder_quant.onnx` 72 MB，
offline `model_quant.onnx` **227 MB**，标点 ct-transformer **270 MB**。

| 模型 | 输入 | 1 线程 | 2 线程 | 4 线程 |
| --- | --- | ---: | ---: | ---: |
| **offline 二遍** | 2 s 语音（33 帧） | 170.5 ms | 100.3 | 73.5 |
| | **3 s 语音（50 帧）** | **238.8 ms** | **138.1** | **95.2** |
| | 5 s 语音（83 帧） | 383.5 ms | 238.3 | 141.9 |
| **标点 punc** | 20 token | 2.3 ms | 2.2 | 1.6 |

* 二遍 3 s 语音单线程 239 ms、四线程 95 ms——**与 §2.16 记的 eot_final 292 ms
  （其中二遍 222–338）对得上**，是同一件事的两个测法。
* **标点意外地便宜**：270 MB 的模型只要 1.6–4.2 ms，不是热点。
* streaming encoder 有 15 组 cache 张量，手工构造输入易错，**本节未测**。

#### 最坏一轮

TTS（NPU + A76 5–7）+ 聊天 LLM（A55 0–3）+ channel 负载（pVAD 100 Hz、EOT 2 Hz，
单线程，不绑核）+ ASR 二遍每 3 秒一次（模拟反复打断），45 秒窗口：

| 二遍线程 | TTS rtf | 真断音 | 缓冲最低 | LLM | 二遍中位/最大 | VAD 帧迟到 |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 4 | 1.014 | **0** ✓ | **257 ms** | 5.80 | 219 / 419 ms | 1419/4500 |
| **2** | 1.004 | **0** ✓ | **525 ms** | 5.78 | **210 / 291 ms** | 1338/4500 |
| 1 | 0.997 | **0** ✓ | 703 ms | 5.79 | 268 / 309 ms | 1471/4500 |

#### 三条结论

**① 最坏一轮过关，但余量只剩 257–703 ms。** TTS 三档都是零断音，可缓冲最低点从
单跑的 840 ms 掉到 257–703 ms。**这是 23 秒回复的数字**；更长的回复会把它耗完。

**② ASR 二遍应当只给 1–2 线程，不是 4。** 单独测时四线程 95 ms 明显优于单线程
239 ms，但**在真实负载下四线程退化到 219 ms**——优势完全消失，却把 TTS 的缓冲从
703 ms 压到 257 ms。**2 线程是最佳点**（TTS 缓冲 525 ms，二遍中位 210 ms）。
这条只有在满载下测才看得见，单独测会得出相反的配置。

**③ VAD 有 30% 的帧迟到超过 5 ms**（1338–1471 / 4500），比 §2.24 的 20% 更差。
其中一部分是复现脚本 `sleep(0.0005)` 的粒度，但趋势是负载抬高了 VAD 的抖动，
**而打断响应就依赖这个**。真实 channel worker 上要重测。

#### 仍然没算进去的

nats、**livekit-server（WebRTC/SFU，真正在搬 RTP 音频包，可能是剩下最大的一项）**、
agent、kernel/hub/data/data-workspace 四个 API、memory 的 bge（§2.14：查询 3.1 ms）、
eidolond 每 5 秒对账、CAMPPlus、以及 9 个 GNOME 进程。

另外发现两个部署侧的事实：

* **`eidolon_models` 组件没有装到板上，`eidolon-asr.service` 不存在**——尽管
  `config/eidolon-rk3588.toml` 声明了 `capabilities.provides = ["rknpu2", "local_asr"]`，
  而 `eidolon_ops.config.CAPABILITY_SOURCES["local_asr"] = ("eidolon_models",)` 也在。
  板上 release.json 的组件列表里确实没有它。**capability 到发布的链路那次没生效。**
* `/root/eidolon_models/.venv` 的解释器软链断了（指向已被我清理掉的
  `/root/.local/share/uv/python/cpython-3.13-*`）。本节改用板上 channel 组件的
  onnxruntime 1.26 直接跑 ONNX，绕开了它。

### 2.26 音色预计算改成快照，稳态首音 2.9 s → 2.0–2.2 s（实测 2026-09-09）

常驻的 `eidolon-tts.service` 模型只加载一次，但每句话还要付一次**音色预计算**
（约 0.8 s）：把 prompt 的前 28 个 token 喂给 encoder 和 Flow，让状态留在两者的
RKNN cache 里。它每句重做，只因为请求会把 cache 用掉，而当时唯一的复位手段是
`Reset()`——清零。

现在两个类都能导出/导入 cache：第一句跑完预计算后存快照，之后每句写回去。

| | 基线 | 快照 |
| --- | ---: | ---: |
| `profile_precompute_ms` 第一句 | 776 / 780 / 852 | 806 / 864 |
| `profile_precompute_ms` 之后每句 | 764–855 | **5.5 / 5.8 / 6.2 / 12.9** |
| `ttft_first_pcm_ms` | 2018–2223 | 1997–2225 |
| **调用方看到的首音（稳态）** | **约 2.9 s** | **约 2.0–2.2 s** |
| `underrun_count`（2×3 句合计） | 5 | **5** |
| `steady_pcm_rtf` | 0.34–0.87 | 0.41–0.84 |

三句全部 `status=PASS`、`saw_eos=1`，三句的 `audio_seconds` 两边完全一致
（4.12 / 3.68 / 1.84）。**断音次数没变多**，`minimum_buffer_after_ms` 两边都是 840。

#### 音频有没有变

* **非服务模式逐字节相同。** 固定 `--seed` 时该二进制本身可复现（同一版跑两遍
  wav / mel / token 三个文件全等），加快照前后也全等。
* **更强的一遍：** 临时改一版，在预计算之后先 `Reset()` 清空、再从快照恢复，输出
  仍与基线逐字节相同——恢复重建的就是预计算留下的那份状态，不是靠残留状态蒙混。
* **服务模式第一句逐字节相同，第二句起不同。** 原因不是脏状态，是预计算**并非
  只依赖音色**：encoder 那半只吃 prompt token，是纯音色的；Flow 那半还吃两块噪声
  （`flow_noise.Next(0)` / `Next(1)`），而取噪声的生成器跨句共享、故意不复位。
  基线每句预计算的噪声都不一样，快照等于把 prompt 段噪声钉在第一句那次抽取上。

量化过这一件事：只改 prompt 段噪声、其余全固定时，mel 余弦 **0.999975**、MAE 0.034
（mel std 3.23，约 1%）；波形对数谱余弦 0.994、MAE **0.417**。而服务模式基线与新
实现第二句的差异是对数谱余弦 0.989、MAE **0.428**——同一量级，说明差异全部来自那
一次噪声抽取。作为尺度参照，两句**不同文本**的对数谱余弦是 0.72。基线本来每句就在
这个范围内抖，快照只是把它钉住。

那两次 `Next()` 仍然要取走再丢掉：生成器跨句共享，少抽两次会让后面每一块的噪声
错位，那才是真的把音频改了。

#### 为什么快照拿得到

cache 全都住在 `rknn_create_mem` 给的**用户侧**缓冲里——`virt_addr` 上的字节就是
状态本身，`rknn_mem_sync` 只做 cache 维护，不是驱动侧的隐藏状态。`StreamingFlow`
早就有 `ExportState`/`ImportState`（steady flow 的交接在用），这次给
`StreamingEncoder` 补上同样的一对。encoder 的 cache 每次 `Run` 之后输入输出对调
（ping-pong），且那几块是 NPU 写的，所以导出前要 `RKNN_MEMORY_SYNC_FROM_DEVICE`
先失效缓存行；按下标写回则与对调到哪一侧无关。

#### 顺带记下的既有行为（不是这次改的）

服务模式每句的 prefill 都是 `keep_history = true`，rkllm 的 KV 历史跨句累积。所以
同一句话连说三遍，token 数是 103 / 75 / 88 而不是三次相同——固定 `--seed` 时这是
可复现的，两个二进制也一致。它与本节的 cache 无关，但会影响任何"连说 N 句"的比对。


### 2.27 服务一直没传 `--max-context`；补上之后暴露了 serve 模式不清 KV cache（实测 2026-09-09）

**两个发现，第二个推翻了第一个的修法。**

#### ① 服务从未传 `--max-context`，所以长句一直被截断

`cosyvoice2_streaming_pipeline.cpp:1137` 的默认值是 288，而模型是
`qwen2_body_w8a8_c2_ctx2048`。`engine.py` 的参数向量里没有这一项。引擎的预算推导
（同文件 1351–1388）：

```
min_tokens = 文本token × 2
available  = max_context − (音色 prefill + 文本token)
max_tokens = min(文本token × 20, available)
max_tokens ≤ min_tokens → 直接抛异常
```

于是天花板是**字数**而不是模型上下文（3 s 音色，prompt prefill 111）：

| 文本 | 文本token | prefill | available | max_tokens | min_tokens | 后果 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 14 字 | ~15 | 126 | 162 | 162 | 30 | 够用 |
| 44 字 | ~47 | 158 | 130 | **130** | 94 | 被夹住，**8/8 次全部精确卡 6.04 s** |
| 80 字 | 83 | **174** | 114 | 114 | **166** | 抛异常，**0 字节音频** |

**§2.20 [C] 早就写了「`--max-context=2048` 必须给，默认太小」**——手工基准传了，
服务没传。所以 §2.21 那些「121 字 23–27 s 零断音」对服务从不成立：服务根本生成不出
那个长度。**此前所有 TTS 的 rtf 都测在被截断的输出上。**

非服务模式单变量验证（同一段 80 字，只改这一个参数）：

```
--max-context=288   error: --max-tokens must exceed --min-tokens,0 产物
--max-context=2048  prefill=174 generated=422 audio=16.88s,810 KB,逐字正确
```

#### ② 但服务模式下把它提到 2048 更糟：跨句 KV cache 没清

补上 `--max-context=2048` 发布到板上，结果全面变坏。同一句话冷启动后连续 6 次：

| 句序 | `max_context=288` | `max_context=2048` |
| ---: | --- | --- |
| 1–3 | 完整 | 完整 |
| 4 | 完整 | **坏**：「大妈妈妈之前也这样子，忽来过，今天天气不错，我们出去走走吧。」 |
| 5 | 完整 | **坏**：音色 prompt 文本 + 目标文本重复三遍 |
| 6 | 完整 | **坏** |
| 合计 | **6/6** | **3/6** |

崩坏内容是**音色 prompt 自己的文本（「爸爸妈妈之前也这样子胡来过。」）加上前几句的
目标文本**，而且是**累积**出现的（前三句好）。

**这个行为 §2.26 末尾已经记下来了**，只是当时看不出它致命：

> 服务模式每句的 prefill 都是 `keep_history = true`，rkllm 的 KV 历史跨句累积。

288 小到残留还没来得及起作用就被挤出去，所以它当时只表现为「同一句话连说三遍
token 数是 103 / 75 / 88」这种无害的抖动。2048 等于把残留的容身之处给了出来。

**修点是一行，位置也定了**：`cosyvoice2_streaming_pipeline.cpp:2071`——每句的
**首次** prefill 传的是 `keep_history=true`，该是 `false`。同文件 2106 和 2121 那两处
`true` 是对的：它们是同一句内的单 token 续推，句内必须保持历史。

改它要连带重验 §2.26 的快照结论（那一节的 token 数差异正是这个累积造成的），所以
留给引擎那条线做，不在本次改动范围内。

同时 rtf 从 0.9 涨到 **1.5–1.6**，迟到块中位从 1–5 涨到 **12**。

**所以那个「默认太小」的值一直在掩盖一个跨句 bug，而不只是它自己的问题。**
提高上下文必须和「每句清 RKLLM KV cache」一起做——形状与 §2.21 的「两核模型和
NPU 核分配必须一起改」完全相同。单独做前者，是用「长句被截断」换「长句被污染」，
后者更坏：截断至少还是这句话的开头，污染是另一句话。

止血靠的是这个值可被环境覆盖：一个 drop-in 设回 288、重启一次就对了，没有回滚发布。

#### ③ 清掉 KV 之后，2048 才是对的（实测同日）

`rkllm.h`（vendored，1.3.0）里就有现成的 API，板上 `nm -D /usr/lib/librkllmrt.so`
确认 `T rkllm_clear_kv_cache`、`T rkllm_get_kv_cache_size` 都有实现。serve 循环每句
开头全清一次（两个 nullptr；范围形式要求 `keep_history==0` 且生成被回调暂停，这里
两个条件都不成立），清完立刻用 `rkllm_get_kv_cache_size` 自证。

**没有改 `keep_history`**：它的语义是「这一次调用是否保留历史」，而句内的续推
（同文件 2106 / 2121）恰恰需要保留，把首次 prefill 改成 0 就得赌「调用前清」还是
「本次不写入」这个边界。显式全清没有这个歧义。

同一句话冷启动后连续 8 次：

| | 完整 | `audio_seconds` | steady rtf |
| --- | ---: | --- | ---: |
| 288 + 清 KV | **8/8** | 全部 4.08 s | 0.82–1.34 |
| 2048 + 清 KV | **8/8** | 全部 4.08 s | **0.836–0.862** |

两轮 `kv_cache_not_empty_after_clear` 都是 **0 次**。

**顺带修掉了 §2.26 末尾记的那条「既有行为」。** 那里写「同一句话连说三遍，token 数
是 103 / 75 / 88 而不是三次相同……会影响任何『连说 N 句』的比对」——那个抖动就是 KV
跨句累积造成的。清掉之后 8 句的 `audio_seconds` 全部精确相同，**服务模式变成可复现
的了**，这对以后所有基准都是收益。

各长度档（每档 4 次，3 s 音色）：

| 文本 | 288 + 清 KV | **2048 + 清 KV** | 音频 | rtf 中位 | 迟到块中位 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 14 字 | 4/4 | 4/4 | 4.08 s | 0.845 | 0 |
| 20 字 | 4/4 | 4/4 | 5.00 s | 0.785 | 1 |
| 44 字 | **0/4**（截断 6.04 s） | **4/4** | 9.80 s | 0.828 | 1 |
| 80 字 | **0/4**（0 音频） | **4/4** | 16.88 s | 0.860 | 0 |
| 126 字 | **0/4**（0 音频） | 见下 | 26.20 s | 0.896 | 2 |

**rtf 中位全部 < 0.9，长文本第一次真正跑在实时线以内。**

上表测于板子空闲、且用 drop-in 覆盖上下文。默认值改成 2048 之后又在**代码默认路径
加真实负载**下验收了一遍（release `rk3588-20260909T095848`，紧跟 cutover 跑，其余
服务刚重启完还在忙）：

| 文本 | 完整 | 音频 | rtf 中位 | rtf 区间 | 迟到块 | TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 14 字 | 4/4 | 4.08 s | 0.936 | 0.924–0.981 | 3 | 2145 |
| 20 字 | 4/4 | 5.00 s | 0.867 | 0.848–0.880 | 3 | 2138 |
| 44 字 | 4/4 | 9.80 s | 0.913 | 0.907–0.922 | 4 | 2234 |
| 80 字 | 4/4 | 16.88 s | 0.937 | 0.933–0.947 | 2 | 2300 |
| 126 字 | 判据不足* | 26.20 s | **0.970** | 0.969–0.972 | 2 | 2417 |

\* 见下面 ④；这一档由人耳对照全文确认完整。

跨句连续 8 句 8/8、`audio_seconds` 全部 4.08 s、`kv_cache_not_empty_after_clear` 0 次、
`status=FAIL` 0 次、`NRestarts=0`、22 units active / 0 failed。

**但 rtf 整体上移了，而且 126 字档到了 0.970——几乎没有余量。** 对照空闲时的
0.785–0.896，说明**这个工作点对负载很敏感**。两个推论：

* §2.24 / §2.25 的整机余量（最坏一轮只剩 257–703 ms）**是在 TTS 被截断的前提下测的**
  ——那时长回复说到一半就结束，TTS 对 NPU 的占用被系统性低估。现在 44 字要说 9.8 s、
  126 字要说 26.2 s，占用是此前的数倍，**那些余量数字需要重测**。
* 全链路再叠上 ASR 与 LLM 之后，长回复很可能掉出实时。这不是新的退步，是第一次能看见
  真实数字。

#### ④ 一个测量陷阱：ASR 客户端只取第一个 final

126 字那档初测只有 1/4，我一度要把它记成 TTS 吞字。不是：把音频切成两半分别送 ASR，
前半和后半各自完整，整段转写去标点后与 122 字原文**逐字相同**。

原因在判据这一侧——我的客户端在**第一个** `is_final` 就 `break`，而 2pass 流式对长
音频会分多段给 final，于是只读到了开头。**取第一段等于把「说全了」误判成「吞字」。**
累积全部 final 段之后才是真数字。

教训与 §2.20 [B] 的 `underruns` 是同一类：先怀疑判据，再怀疑被测的东西。

**但修掉客户端之后 126 字仍然判 0/4，所以问题不止在客户端——记在这里，留给 ASR
那条线，不在本次范围内：**

* **整段送 ASR 转写 26 秒音频，结果不稳定。** 同一个 wav 文件、同一套客户端逻辑，
  一次给出全部 122 字，另一次只到「合成可以」就停。不是 TTS 每次说的不一样：
  `audio_seconds` 四次都是 26.20 s，rtf 区间 0.901–0.908。
* 我尝试「累积所有 final 段」时加了 `close_stream`，**反而更糟**：在 `end_utterance`
  之后立刻发关闭，ASR 来不及吐完后续的 final。
* 分段送（约 10 秒一片）稳定，但**按固定秒数硬切会在接缝处造出伪影**：切点落在字
  中间，转写反而比原文多出 1–2 个字（「三个**是**」「**理由**匹配」「**旅由**匹配」，
  原文是「三个神经网络核心」「再由流匹配模型」）。四次 `audio_seconds` 全部 26.20 s、
  rtf 0.903–0.921，主体内容全对，所以多出来的字是判据造的，不是 TTS 说的。

**净结论：26 秒这个长度上，我没有可靠的自动判据。** 整段不稳、硬切有接缝伪影，
要判就得按静音切分或者用人耳。所以 126 字档只能说「音频长度与语速正常、主体内容
三种判法都对」，不能给出「逐字完整」的结论。

要查的是 ASR 侧的东西：`end_utterance` 之后 2pass 的 offline 修订对长 utterance
会吐几段 final、会不会被 `max_utterance_seconds`（60 s）之外的什么东西提前截断、
以及客户端应当读到什么条件为止。**这是 ASR 的问题，与本节的 TTS 结论无关**——44 字
和 80 字档的音频只有 9.8 s 和 16.9 s，整段转写在那个长度上是稳的，两轮独立测量
都是 4/4。

#### ③ 新增一个此前没测过的维度：文本完整性

`underruns` **不是断音次数**，是迟到块计数（§2.20 [B] 已辨明，我这次又误读了一遍）。
而「字有没有说全」此前**从未测过**：§2.20/§2.21 全部在测缓冲余量和 rtf，它们回答不了
这个问题。

判据用的是**板上自己的 ASR**：把合成音频重采样到 16 kHz 送回 `eidolon-asr`，转写去标点
后与原文逐字比较。人耳抽样核对过一坏一好两个样本，与判定一致。这个手法便宜且客观，
值得固化成基准脚本。

用它测出来的现状（288，3 s 音色，每档 8 次）：

| 文本 | 完整率 |
| ---: | ---: |
| 14 字 | 7/8 |
| 20 字 | 7/8 |
| 44 字 | **0/8**（全部截断在 6.04 s） |
| 80 字 | **0/8**（全部 0 音频） |

**短句可用，44 字起不可用。** 服务的 `MAX_TEXT_CHARACTERS` 是 400，也就是说它接受的
长度远超它能正确合成的长度——这个上限该怎么定，等 serve 循环修好后再测。

#### 顺带否掉一个岔路

`voices/testwav_llm_prompt1s` 比 `testwav_prompt3s` 少吞字，曾被我当成音色档位的
质量差异。不是：它的 prefill 是 54 而非 111，在 288 的预算里多出 57 token ≈ 2.3 秒
可生成语音。纯算术。两者是**同一段 `test.wav` 的不同 prompt 长度**，说话人 embedding
（`flow/spks.f32.bin`）是同一份，1 s 那份的 manifest 自己写着「沿用三秒参考音频生成的
Flow Prompt 和说话人状态」。TTFT 上两档无可测差异（两轮 A/B 方向相反，都在噪声内）。
所以不必为它入库第二个音色档位。


### 2.28 「欠供」是个误名；真实的可用边界在 80 字与 126 字之间（实测 2026-09-09）

手机端一次真实对话之后，Channel 日志里出现 `[local_tts] the Host underran 4.0 times
on 5.08s of audio`，我据此报了"TTS 全程欠供"。**那是错的，而且是 §2.20 [B] 已经辨明过
的同一个误读第三次发生。**

`underrun_count` 由引擎算出（`cosyvoice2_streaming_pipeline.cpp:1788`）：

```
deadline = 首块就绪时刻 + 之前累积音频的时长
underrun = (deadline − 这一块就绪时刻) < 0
```

`deadline` 从**首块就绪时刻**起算，隐含"播放器拿到首块立刻开播、零缓冲"。所以它数的是
"有多少块晚于自己的截止时刻到达"——对一个 rtf 接近 1 的生产者，几乎每块都算，它跟踪的
是**音频长度**而不是听众体验。实测印证得很干脆：16.88 s 音频对应 `late_chunks = 16`。

**能不能听到，看的是缓冲最低点 `minimum_buffer_after_ms`，小于 0 才是真断音。**
这个量此前**没有被服务转发出去**，所以"有没有断音"在板上根本不可判定。补上转发（并把
`underruns` 改名为 `late_chunks`）之后测出来：

| 文本 | 音频 | **缓冲最低点** | late_chunks | steady rtf | TTFT | 判定 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 14 字 | 4.08 s | **+762.9 ms** | 3 | 0.966–0.999 | 2.2 s | 未断 |
| 20 字 | 5.00 s | **+736.6 ms** | 3–4 | 0.900–0.937 | 2.1 s | 未断 |
| 44 字 | 9.80 s | **+446.7 ms** | 8–9 | 0.972–0.990 | 2.2 s | 未断 |
| 80 字 | 16.88 s | **+208.9 ms** | 16 | **1.002–1.019** | 2.3 s | 未断，余量已薄 |
| 126 字 | 26.20 s | **−246 ~ −372 ms** | 26 | **1.036–1.041** | 2.5 s | **真断音** |

（每档 4 次，3 s 音色，服务模式，代码默认 `max_context=2048`，板子空闲 load 0.27。）

**根因是 rtf 跨过 1。** 80 字档 1.002–1.019、126 字档 1.036–1.041，稳态一旦超实时，欠账
就持续累积：缓冲以约 **43 ms / 秒音频** 的速率被吃掉，26 秒时穿透零线。按 4.08→16.88 s
的斜率外推 26.2 s 应得约 −194 ms，实测 −246~−372 ms，方向与量级都对（实测更差）。

**可用边界：约 80 字 / 17 秒音频。**

#### 产品路径上这条边界目前被挡住了——但不是有意设计的

Channel 的 `local_tts.hard_max_chars = 60`，所以单次送进 TTS 的文本 ≤60 字 ≈ 12 秒音频，
落在 446 ms 余量那一档，**段内不会断音**。这是个巧合式的安全：实测 80 字只剩 209 ms，
60 字才是有余量的点。这个数应当从"恰好安全"变成"声明为安全边界"。

#### 风险因此转移到段之间

长回复被切成多段，每段 TTFT 约 2.5 s，而 `preemptive_tts: false`——预合成是关的。若播完
一段才开始合成下一段，段间就是 2.5 秒静默。§2.20 [D] 量过分句的代价：5 句付 9.5 s 死时间、
实效 rtf 1.397。60 字段有 12 秒音频，远大于 2.5 s TTFT，**有充足的流水重叠空间**，所以
这是配置问题而不是能力问题。

按代价排序的手段：段间流水（开 `preemptive_tts`）→ 启播缓冲 ≥400 ms（上游 README 建议值，
正好覆盖 126 字的 −372 ms 缺口，代价是首音 2.5→2.9 s）→ 把 `hard_max_chars=60` 固化成显式
契约 → 引擎吞吐（§2.11 ② 的 hift 零拷贝，是唯一能把 rtf 压回 1 以下的，但在前三项已挡住
边界时优先级最低）。

#### 一个尚未解释的矛盾

§2.21 记录 **121 字 / 23–27 s / rtf 0.905–0.937 / 零断音**；本节 126 字得 rtf 1.036–1.041
且真断音，**差约 12%**。同一块板、几乎同样的长度。

已知差异变量有三个，而不是一个：**音色**（§2.21 用 1 s 音色 prefill 54，本节用 3 s 音色
prefill 111）、**模式**（手工非服务 vs 服务常驻）、**引擎版本**（§2.21 测于 2026-09-06，
那之后加了音色预计算快照 §2.26 和每句清 KV §2.27）。在隔离出单个变量之前，这 12% 不能
归因于其中任何一个。

#### 音色变量已排除（A-B-A 实测同日）

用 **A-B-A**（3 s → 1 s → 3 s）而不是简单 A/B：切音色必须重启服务，而重启与时间推移会带来
热与负载漂移，只有两次 A 一致，中间的对比才有效。同一段 126 字文本、同一引擎、服务模式、
每段 5 次：

| | `minimum_buffer_ms` 中位 | 区间 | steady rtf 中位 | `audio_seconds` |
| --- | ---: | ---: | ---: | ---: |
| A 3 s 第一次 | −286.5 | −418.9 ~ −148.2 | 1.0362 | 26.20 |
| **B 1 s** | **−632.7** | −709.4 ~ −311.7 | 1.0374 | **28.68** |
| A 3 s 第二次 | −375.2 | −496.6 ~ −257.1 | 1.0417 | 26.20 |

**结论：1 s 音色不是解法，反而恶化约 330 ms。** 两次 A 的中位差 88.7 ms（小于组内 σ≈100，
但方向与漂移一致），而音色效应 330 ms 远大于它，所以结论成立。

**机制是音频长度而不是吞吐**：两档 rtf 几乎相同（1.0374 vs 1.0390），说明音色对稳态吞吐
没有影响——prefill 少 57 token 在 2048 上下文下不再是瓶颈。恶化来自 1 s 音色把同一段文本
说成 **28.68 s**（比 3 s 音色长 9.5%）；rtf > 1 时欠账 ≈ (rtf−1) × 时长，多出的 2.48 秒
就多出约 90 ms 欠账。

**所以那 12% 不是音色造成的**，剩下模式与引擎版本两个变量。

顺带两件：每档内部 5 次的 `audio_seconds` **完全一致**（26.20 / 28.68），又一次验证了 §2.27
清 KV 带来的可复现性；15/15 次 126 字全部 `minimum_buffer_ms < 0`，独立复现了本节的断音结论。

#### 连续多轮耐久：产品路径（60 字）在真实连续使用下守得住（实测同日）

先说一条**对上一节的更正**。我曾从 126 字那轮看到「第二次 A 比第一次差 88 ms、板子
43 → 63 °C」，据此写下"连续长回复会自我恶化"。**那个结论下早了**：本轮把温度同样推到
61 °C，缓冲余量却完全不降。所以那 88 ms 更可能是组内噪声（σ≈100 ms），**不是热效应**。

真正该测的是产品实际走的路径。Channel 的 `hard_max_chars = 60`，所以单次送进 TTS 的
永远是 ≤60 字，而不是 126 字。用 60 字文本（13.20 s 音频）、背靠背无间隙合成、三阶段：

| 阶段 | 轮数 | `minimum_buffer_ms` 中位 | 区间 | rtf | 温度 | load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 冷启基线 | 3 | 554.8 | 444 ~ 561 | 1.005–1.015 | 45 → 53 °C | 0.44 → 1.68 |
| **连续施压** | **20** | **504** | **338 ~ 649** | 1.000–1.027 | 53 → **61 °C** | 2.28 → 3.94 |
| 冷却 180 s 后 | 3 | 517.9 | 500 ~ 544 | 1.007–1.011 | 47 → 55 °C | 0.41 → 1.55 |

**三阶段中位一致（555 / 504 / 518 ms），无单调趋势，无跨零。** 最低点 337.6 ms 出现在施压
第一轮（过渡瞬态），之后回升并稳定。冷却阶段与基线一致，进一步确认没有热致累积。
20 轮后 `NRestarts=0`、`readyz=200`。

设置了冷却阶段是因为它能分开两种完全不同的结论：恢复 = 热致可恢复（解法是散热降载），
不恢复 = 与温度无关的累积效应（更严重）。**只跑前两段是分不出来的。**

**声明：这是最坏情况。** 背靠背合成没有间隙，而真实对话中间有用户说话与 LLM 生成的时间，
板子有机会散热。所以真实使用应当好于此表。

#### 这批数据对那四项手段的意义：三项可以不做

* **启播缓冲 ≥400 ms —— 不需要现在做。** 最坏余量 337 ms 已经够，而加 400 ms 缓冲要把首音
  从 2.3 s 推到 2.7 s，是净损失。
* **hift 零拷贝 —— 更不需要。** rtf 虽在 1.00–1.03，但 60 字这个长度上欠账攒不到伤人的量。
* **段间流水（`preemptive_tts`）—— 仍值得做，但优先级下降。** 它解决的是段间静默，属于
  **首音延迟**问题，不是断音问题。
* **把 `hard_max_chars = 60` 固化成显式契约 —— 反而变成最重要的一项。** 它现在是唯一挡住
  126 字那条 −372 ms 边界的东西，却只是个默认值：没有任何测试或契约说明"超过它会断音"。
  一次无心的调大就会把 §2.28 开头那个边界直接暴露给用户。

## 3. 已知缺陷

| 缺陷 | 证据 | 影响 |
| --- | --- | --- |
| 两个 PCIe 控制器 link fail | `rk-pcie fe190000/fe170000: PCIe Link Fail, LTSSM 0x3` → `failed to initialize host` | 不影响 NVMe（另一控制器正常） |
| 有线口无链路 | `enP3p49s0` driver `r8169`，`carrier=0` | 当时无网线。**ops 的 `require_wired_release_upload = true` 会拒绝在无线上做 release 上传，所以有线是硬要求** |
| 2.4 GHz WiFi 抖动 | RTT `min/avg/max/mdev = 2.8/11.7/42.0/15.2` ms；SSH 建连 1.15–1.38 s；期间掉线一次 | 交互和大文件传输不可靠，不能用于交付链路 |
| NPU region 重复注册 | `RKNPU fdab0000.npu: can't request region for resource` | 未影响推理（上述基准通过）；记录备查 |
| ASR 整段转写长音频不稳定 | 同一 26 s wav、同一客户端，一次给全 122 字、一次只到一半；`audio_seconds` 四次相同故非 TTS 侧抖动。分段（~10 s）送则稳定。详见 §2.27 ④ | **只影响用 ASR 做判据的测量**，不影响 ASR 自己的产品用途（对话是短 utterance）。待查：`end_utterance` 后 2pass 修订吐几段 final、客户端该读到什么条件为止 |

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
