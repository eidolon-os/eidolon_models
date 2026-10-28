# 现有 Channel 真实质量提取路径离线验收

直接运行 `ReferenceSpeakerExtractor`，使用原 FireRed pVAD（未提供目标说话人条件）、本地 CAM++ 权重与原 ModelScope pipeline，不替换 provider/VAD，不连接服务。公开数据复用已固定摘要的 AISHELL 12 人，每人第一份完整录音；无用户录音、无向量或音频写入结果。

结果：12 条中 7 条提取成功、5 条因 speech_too_short 拒绝。拒绝样本的录音长度分别为 5.999、3.349、3.520、3.227、5.916 秒，VAD 有效语音为 950、640、640、1490、380 ms。没有逐帧人工真值，因此不能直接称为五次误拒，也不能把模型阈值降低来凑齐通过。这个结果说明不能拿完整录音长度替代 VAD 有效语音验收。

真实 warm_up 耗时约 7.67 秒，包含模型清单摘要校验、VAD 和 embedding 预热，不含进程启动和顶层 Python 导入。暖态 12 次处理约 259–1070 ms，包括质量与通过时的 embedding；拒绝项不运行 embedding。单进程 CPU、OMP/BLAS 线程限制为 1，VAD 原实现单线程；非多轮压测、无并发，不包含 HTTP/Data/gRPC/Memory，不能据此承诺整链 P95。没有改动旁路 deadline，也没有开启自动候选发现。

复现（工作区根目录，已有 Channel venv 与本地权重）：

```bash
HF_HUB_OFFLINE=1 MODELSCOPE_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 eidolon_channel/.venv/bin/python eidolon_models/evals/speaker-verification/runtime-quality-20261029/evaluate.py \
 --assets .eidolon/anonymous-speaker-eval-20261028/assets.json \
 --model .eidolon/mac-product/state/models/campplus-zh-16k-7959bae7 \
 --output /private/tmp/person-runtime-quality-repeat.json
```

results.json 记录逐样本质量/耗时、脚本摘要和模型身份；provenance.json 补充本次 Channel HEAD、VAD 权重、模型清单和输入清单摘要。输出拒绝覆盖。当前公开干净朗读数据可能属于训练集，不是手机远场/多人录音；既有质量输出仍没有可靠 overlap_detected。本验证仅证明实际原提取路径执行及其当前拒绝/延迟表现，不能替代自动识别准确率或完整设备联调。
