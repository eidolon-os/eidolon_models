# 匿名身份声纹模型 Mac 初筛

日期：2026-10-28。结果：[results.json](results.json)。产品解释与决策见工作区 `docs/跨系统/匿名身份声纹模型Mac初测-20261028.md`；功能方案见 `docs/跨系统/匿名用户与声纹生命周期实施计划-20261028.md`。

这是离线实验，不是生产身份服务。没有修改现有服务、数据库、模型或手机应用；没有读取产品用户录音。Models 仓只保存脚本、公开样本标识/摘要和汇总，不提交音频、向量、下载权重或上游源码副本。

## 冻结设置

- Mac：Apple M3 Pro，12 核，36 GiB RAM，macOS 26.5.2 arm64。CPU 单推理线程，五组配置依次运行，普通调度；没有停止现有后台服务。sandbox 不允许 nice 降低优先级，已记录此事实。
- Python 3.13；torch/torchaudio 2.10.0、onnxruntime 1.26.0、numpy 2.4.4、soundfile 0.14.0、modelscope 1.37.1、scipy 1.17.1。隔离增加 kaldi-native-fbank 1.22.4；诊断使用 onnx 1.23.2 / ml-dtypes 0.6.0。
- 数据：官方 HF `AISHELL/AISHELL-1`，revision `bbe295d530192a4cd41644b711c9aecd087df653`。12 位说话人各取归档顺序中前 12 个长度 >=3.2 秒的 16 kHz 单声道文件；共 144 份，约 22.86 MB，来自 train 子集。
- 校准说话人：S0002、S0003、S0004、S0005、S0010、S0011；测试：S0012、S0013、S0018、S0019、S0020、S0021。每人前 3 份作为注册参考，其余 9 份为探针。没有将同一文件裁剪后同时用于注册和探测。
- 每份录音取中心 0.6/1/1.5/2/3 秒和完整长度；每配置 864 次提取，总计 4,320 次，全部成功。短档长度是裁剪音频长度，**不是 VAD 测得的有效语音长度**。
- 每档阈值仅用校准说话人计算：注册匹配阈值高于校准负例最大分数；候选归并阈值高于独立文件负例最大分数；新建阈值为 `min(join-0.05, positive_pair_p05)`；分差 0.05。此规则是保守实验策略，不能输出为正式线上阈值。每档有 270 个注册负例比较、1,215 个候选负例比较，但它们共享人/录音，不独立。
- 开集测试：2 个注册人、4 个未注册人；每档 18 个已知探针和 36 个未知探针。报告计数，不声称低 FAR。
- 匿名模拟：同一 54 个探针，以固定 0–4 五个种子改变输入顺序；先匹配两个注册人和动态候选，至少 1.5 秒才能建候选，3 个去重证据晋升；候选模板最多 5 个，最多 20 候选。录音 ID 只用于去重与计数，真实标签只用于评分，不参与匹配。
- 注册模板取 3 个单位向量的归一化平均；候选加入须满足 centroid 分数、所有已保存证据的一致性及 top1/top2 分差。此模拟不代表生产拟合/状态机已实施。
- 压力探针：每配置 12 个 10 dB 白噪声叠加、12 个等能量双人混合、1 个数字静音、1 个纯白噪声，共 130 次。噪声开集使用 2 人 gallery；混合/静音/纯噪声数值诊断使用 6 人参考 gallery。仅诊断 encoder；没有 VAD/重叠检测器，不能把其接受计数当作产品授权失败率。
- 时间包含特征、encoder、向量校验/归一化，不包含 VAD、音频读取、排队、服务传输、Person/Memory 操作；每长度 n=144，报告本次观测 p50/p95，不是重复压测或响应 SLA。
- 峰值 RSS 为进程累计高水位，包含已导入的 Python/torch/ORT 库，不等于权重内存或最小部署内存。加载时间不含解释器和顶层依赖导入，也不是整套服务冷启动。

## 模型与来源

1. CAM++ 原生：读取现有固定权重，复用已安装 ModelScope `CAMPPlus` 与其 Kaldi FBank80/均值处理，未调用在线产品 API。
2. CAM++ / ERes2NetV2 ONNX：`csukuangfj/speaker-embedding-models` revision `0743f301363dec56491a490f6d6cbc9d67f9a3bf`；遵循模型 metadata 和 sherpa 官方 FBank 前处理。
3. ReDimNet2：官方代码 commit `c5bbe0b76e37df698c403f8844e41304ceab6307`；release v1.0.0 的 B3 `vb2+vox2+cnc2_v0-lm` 与 B2 `vox2-lm`。审阅固定源码后本地构造模型，`weights_only=True` 加载；没有执行浮动 torch.hub。

所有实际文件 SHA256 与代码来源在 results.json 中。原作者/导出者许可互相独立：ReDimNet2 MIT；当前 CAM++ 归档卡 Apache-2.0；AISHELL HF 元数据标 Apache-2.0，正文同时说明学术用途，本次仅作内部研究、未重新分发语料。

## 复现

从工作区根目录执行。`research_python` 指向已具备上述依赖的研究 Python；新增依赖放在独立 target 目录，不能升级正在运行的 Host 环境。

```bash
research_python="$PWD/eidolon_channel/.venv/bin/python"
research_root="$PWD/.eidolon/anonymous-speaker-eval-20261028"
research_scripts="$PWD/eidolon_models/evals/speaker-verification/anonymous-20261028"
export PYTHONPATH="$research_root/deps:$PWD/eidolon_models/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

"$research_python" "$research_scripts/prepare_public_assets.py" "$research_root"
for model in campplus_native campplus_onnx eres2netv2_onnx redimnet2_b3 redimnet2_b2; do
  "$research_python" "$research_scripts/extract.py" "$research_root" "$model" \
    --native-weights "$PWD/.eidolon/mac-product/state/models/campplus-zh-16k-7959bae7/campplus_cn_common.bin"
done
"$research_python" "$research_scripts/analyze.py" "$research_root" --output "$research_root/summary.json"
for model in campplus_native campplus_onnx eres2netv2_onnx redimnet2_b3 redimnet2_b2; do
  "$research_python" "$research_scripts/controls.py" "$research_root" "$model" \
    --native-weights "$PWD/.eidolon/mac-product/state/models/campplus-zh-16k-7959bae7/campplus_cn_common.bin"
done
"$research_python" "$research_scripts/diagnose_onnx.py" "$research_root"
"$research_python" -m pytest -q "$research_scripts/test_analysis.py" eidolon_models/tests/test_speaker_evaluation.py
```

输出默认拒绝覆盖，第二次实验应使用新目录。下载器只读每个语料归档的开头来提取所需文件，避免整套下载；记录固定语料 revision、归档成员路径和每份文件 SHA256，但没有下载并校验完整归档。模型下载会核对上游提供的摘要。

`results.json` 是 summary.json、controls/*.json、onnx-pooling-diagnostic.json、assets.json 的无向量汇总，增加硬件与脚本摘要；原始结果保留在工作区隔离目录。发布汇总前重新核对脚本摘要，不能把修改后的程序冒充已执行版本。

## 重要限制与下一步

- 这是中文干净朗读，不是手机、远场、跨日、多人自然会话；采集 session 未知。模型训练数据可能包含 AISHELL，不能声称严格训练外泛化。
- 同源裁剪和五次顺序复用不是新增独立样本；144 份不能支持低误认率承诺。
- 当前 CAM++ 仍是下一阶段基线。ERes2NetV2 保留为质量增强候选；两款 ReDimNet2 的当前校准不能可靠排除陌生人，不能据参数量直接晋升为产品首选。
- 公共 CAM++ ONNX 与原生路径不能视作等价：本次发现 AveragePool 在 ceil 尾部计入 padding 的差异。diagnose_onnx.py 只在内存更改 count_include_pad 做因果定位，**没有导出或发布修补后的模型**；未来应采用可追溯的正确导出和全范围数值回归。
- 声纹 encoder 对静音、噪声也能输出向量。自动建档需要独立质量检查；重叠检测必须另行验证。不能用成功提取代替有效证据。
- 没有部署匿名池、没有创建真实 Person/Realm、没有修改等待预算；异步状态机、Mobile 页面和真实多人体验仍按计划实施。

来源：[AISHELL 官方模型库](https://huggingface.co/datasets/AISHELL/AISHELL-1)、[ReDimNet2](https://github.com/PalabraAI/redimnet2)、[ONNX 导出产物](https://huggingface.co/csukuangfj/speaker-embedding-models)、[sherpa 前处理示例](https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/3dspeaker/test-onnx.py)。
