# CAM++ 原生导出 ONNX 与公共 ONNX 对比

日期：2026-10-28。研究范围：从当前固定 PyTorch checkpoint 实际导出 ONNX，比较数值、已有模板兼容性、匿名策略及 Mac CPU 开销。未变更 Host、Person、Memory、Mobile 或线上模型资产。

## 文件与边界

- `export.py`：复用已安装 ModelScope CAMPPlus，strict 加载原权重；导出 batch=1、动态时间维的 FP32 ONNX，opset 18。`legacy` 为默认传统导出对照；`semantic` 为有明确尾段平均语义的导出；`dynamo` 留作可复现的失败路径。
- `compare.py`：复用上一轮的公开语料、样本切分和评估器。相同特征对照与实际前处理对照分别报告；仅在原生校准人上确定阈值，所有导出版本冻结使用这些阈值。
- `benchmark.py`：每个后端在独立进程中顺序运行，ORT 进程不导入 PyTorch；记录 864 次提取的各时长分位数与进程 RSS 高水位。
- `test_export.py`：406 个池化输入长度 × 常量/随机两种输入，核验导出规则的分母语义。与原有评估测试一共 24 项通过。
- `package_results.py`：验证导出与固定策略检查通过，汇总独立性能数据和脚本/源文件摘要；不发布音频或向量。
- `results.json`：无向量、无音频的机器可读汇总和来源摘要。原始向量与生成的 ONNX 保留在工作区 `.eidolon/campplus-onnx-export-20261028/`，不作为生产模型发布。

## 导出选择

固定原生权重 SHA256：`3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8`。

PyTorch 2.10 + onnxscript 0.7.2 的 dynamo 路径在动态 segment pooling / slice 后的形状约束处失败，未生成可运行模型；错误为 `GuardOnDataDependentSymNode`，发生在 CAMLayer 的 linear1 前。不能以静态长度导出成功替代动态长度验收。

传统 exporter 将 PyTorch `avg_pool1d` 的 `ceil_mode`、`count_include_pad` 直接传入 ONNX。在本模型 **padding=0、kernel=stride=100、ceil_mode=True** 的调用中，PyTorch 尾段除数为有效元素数量；本次 ORT 中 `count_include_pad=1` 的行为不等价。`semantic` 通过 exporter 的算子翻译扩展点，将这个已核验的调用映射为 `count_include_pad=0`。遇到其他池化参数会报错，避免把此模型特例扩散成通用算子实现。

没有修改上游 Python 源码、模型类、权重或公共 ONNX。产物直接由原生 checkpoint 导出；不是修改公共 ONNX 后重命名。图结构检查要求 52 个池化节点全部使用已核验规则。该路径依赖 deprecated 的传统 exporter，固定环境可复现；后续工具链升级必须重新执行全部验收。

ONNX 输入为 float32 `[1,T,80]`，核验范围 `30 <= T <= 3000`，输出 `[1,192]` 未归一化向量。图本身不执行该范围检查；接入端需校验。前端为 16 kHz 单声道浮点波形、80 维 Kaldi FBank、按时间均值归一化；输出另做 L2。没有验证动态 batch、FP16、INT8、CoreML、RKNN 或 ARM CPU 性能。

## 对比协议

- 同一台 Apple M3 Pro（36 GiB、12 核），CPU 单推理线程，普通调度；服务未停止。旧实验硬件信息在汇总中保留。
- 复用固定 AISHELL-1 revision 与 144 份独立文件，12 人；0.6/1/1.5/2/3 秒及全长，共 864 个片段。先核验每份文件 SHA256。不是私人产品录音。
- 每个片段：原生 + 传统自导出/公共/语义正确自导出（完全相同的 torchaudio 特征）；另测公共/自导出的 kaldi-native-fbank 实际链路。总计 6 条输出路径、5,184 份向量。
- 34 个合成特征长度覆盖范围端点及内部池化边界（输入约每 200 帧一个窗口，跨过 TDNN stride=2）；报告它们与真实音频分离。这些合成输入用于数值验证，不能算识别准确率样本。
- 数值验收门槛预先写入脚本：每条自导出向量与原生 cosine >=0.999999、L2 归一化后最大分量差 <=1e-4；两种前端及全部边界输入均须满足。
- 6 人校准 / 6 人测试；每人 3 份参考、9 份探针。每档 2 个已知 + 4 个未知，共 54 个探针。不同模型不重新调阈值。
- 比较用新后端重建参考库，以及保留原生参考库两个条件的逐条身份决定；报告 864×864 同后端和跨后端相似度矩阵最大差。
- 复用已有匿名模拟：5 种顺序、3 份证据晋升、最少 1.5 秒才允许建候选；这里只比较每种顺序的统计输出，没有实现生产身份生命周期，也未记录逐事件候选归属轨迹。
- 性能是另一轮独立、串行进程，每档 n=144；含特征、网络及 L2，排除 IO/VAD/排队/Person/Memory。RSS 包含运行库，不能当作设备部署最低内存。ORT 不加载 PyTorch，因此与上一轮全部脚本顶层导入 Torch 的 RSS 不可直接相减归因为模型优化。

## 复现

从工作区根目录执行；先按上一轮 `anonymous-20261028/README.md` 准备固定语料和依赖。新增导出依赖为 onnx 1.23.2、onnxscript 0.7.2、onnx-ir 1.1.0，全部安装到隔离 `deps`；未升级 Host venv。

```bash
research_python="$PWD/eidolon_channel/.venv/bin/python"
corpus="$PWD/.eidolon/anonymous-speaker-eval-20261028"
trial="$PWD/.eidolon/campplus-onnx-export-20261028"
scripts="$PWD/eidolon_models/evals/speaker-verification/campplus-export-20261028"
native="$PWD/.eidolon/mac-product/state/models/campplus-zh-16k-7959bae7/campplus_cn_common.bin"
public="$corpus/weights/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
export PYTHONPATH="$corpus/deps:$PWD/eidolon_models/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# 可选：保留 dynamic exporter 失败报告，退出码非零是本次已记录现象。
"$research_python" "$scripts/export.py" "$native" "$trial/weights/campplus-dynamo.onnx" --exporter dynamo

"$research_python" "$scripts/export.py" "$native" "$trial/weights/campplus-legacy.onnx" --exporter legacy
"$research_python" "$scripts/export.py" "$native" "$trial/weights/campplus-semantic.onnx" --exporter semantic
"$research_python" "$scripts/compare.py" "$corpus" "$trial" "$native"
"$research_python" "$scripts/benchmark.py" "$corpus" "$native" "$trial/benchmark-native.json" --backend torch
"$research_python" "$scripts/benchmark.py" "$corpus" "$public" "$trial/benchmark-public.json" --backend ort
"$research_python" "$scripts/benchmark.py" "$corpus" "$trial/weights/campplus-semantic.onnx" "$trial/benchmark-exported.json" --backend ort
"$research_python" -m pytest -q "$scripts/test_export.py" \
  eidolon_models/evals/speaker-verification/anonymous-20261028/test_analysis.py \
  eidolon_models/tests/test_speaker_evaluation.py > "$trial/tests.log" 2>&1

"$research_python" "$scripts/package_results.py" "$corpus" "$trial" "$scripts/results.json"
```

输出拒绝覆盖；新一轮使用新的 `trial` 目录。比较脚本保存本地向量用于审查，发布汇总不得包含它们。模型来源、语料限制沿用上一轮：公开中文朗读、可能存在预训练重合、非跨设备跨日，不可据此承诺家庭对话误认率。

参考：[PyTorch 导出扩展](https://docs.pytorch.org/docs/2.10/onnx.html)、[AvgPool1d](https://docs.pytorch.org/docs/2.10/generated/torch.nn.AvgPool1d.html)、[ONNX AveragePool](https://onnx.ai/onnx/operators/onnx__AveragePool.html)。具体行为以本次固定版本实测为准。
