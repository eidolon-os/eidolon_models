# 短话轮重叠检测离线筛查

结论：本次 Pyannote segmentation-3.0 ONNX 初筛不足以作为自动建档的无重叠放行依据，未接入运行链路。不能将未检出重叠等同于单人语音。

复用 [Sherpa 官方公开发布包](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2) 中的 FP32 model.onnx。包内 LICENSE 为 MIT / Copyright 2022 CNRS；[原模型说明](https://huggingface.co/pyannote/segmentation-3.0)、[官方使用文档](https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/models.html)。没有接受原 Hugging Face 门控条款、安装新依赖、执行下载源码或改动运行模型。权重只在临时目录；Git 仅保存本脚本和无音频结果。

复用 anonymous-20261028 已下载并固定摘要的公开 AISHELL 样本，每人第一份，共 12 人。中心裁剪 1.5 / 3 秒；相邻不同人等 RMS 混合与第二人低 10 dB 混合，共 48 个混合样本，另有 24 个单人及 2 个静音控制。整体峰值缩放防数字削波；未人为增加噪声或修改标签。

按上游 powerset 映射取 argmax，类别 4–6 为双人重叠；短样本右补零到模型 10 秒窗口，只计入感受野完整落在原音频内的帧，禁止把补零计入“无重叠”时长。CPU 单线程、预热一次，计时覆盖模型调用，不含加载和读盘；模型、脚本、输入清单 SHA256、ORT 版本及逐样本测量见 results.json。

| 输入 | n | 至少一帧重叠 | 累计重叠 ≥100 ms |
|---|---:|---:|---:|
| 单人 1.5 s | 12 | 1 | 0 |
| 单人 3 s | 12 | 0 | 0 |
| 等能量双人 1.5 s | 12 | 10 | 10 |
| 等能量双人 3 s | 12 | 9 | 9 |
| 第二人低 10 dB，1.5 s | 12 | 11 | 10 |
| 第二人低 10 dB，3 s | 12 | 9 | 9 |

7/48 个构造混合完全未检出重叠。100 ms 只是报告用观察点，**不是发布阈值**；提高该门槛反而新增一个漏检。各组推理 P95 约 35–36 ms，不含 CAM++、VAD、队列和传输，不能称为全链延迟。

限制：音频级混合标签不是逐帧人工真值；两轨中原本静音可能不重叠。只做短片段筛查，不提供帧级召回率；相邻混合复用说话人/录音，不是独立样本。AISHELL train 可能与模型训练重叠；不是手机远场、真实双人、跨日或训练外验证。结果不足以校准生产阈值，也不代表此模型所有用法都不可行。下一步应比较足够上下文、软概率/不确定性拒绝策略和独立真实多人标注集，且在现有异步提取预算内验证；缺少证据时仍保持 unknown，不输出可信 overlap_detected=0。

从工作区根目录复现（输出存在时拒绝覆盖）：

```bash
eidolon_channel/.venv/bin/python eidolon_models/evals/speaker-verification/overlap-20261029/evaluate.py \
  --assets .eidolon/anonymous-speaker-eval-20261028/assets.json \
  --model /private/tmp/eidolon-person-overlap-eval/sherpa-onnx-pyannote-segmentation-3-0/model.onnx \
  --output /private/tmp/overlap-repeat.json
```

## 概率拒绝诊断（第二轮）

新增 `--probability-analysis`，结果独立保存在 probability-results.json；第一轮 results.json 对应 `04d070a` 版本脚本，保持原样。本轮校验模型输出指数化后为归一化概率，逐帧求双人三类概率之和，以有效帧最大值作拒绝分数，报告预设 0.01/0.02/0.05/0.1/0.2/0.3/0.5 网格。小于阈值只称“未被本规则拒绝”，不能称为质量准入成功；静音也必须由既有 VAD 等独立条件拒绝。

将前六位与后六位说话人分组，组内循环构造混合，禁止混合跨组。每组 12 个单人片段和 24 个混合片段；同一录音仍被多次使用。字段 calibration/heldout 仅描述本轮说话人分组；这些人此前已经参与硬分类诊断，**不是全新未接触的确认测试集**，也没有排除训练数据重合。

| 最大概率拒绝阈值 | 前组混合未拒绝 /24 | 前组单人未拒绝 /12 | 后组混合未拒绝 /24 | 后组单人未拒绝 /12 |
|---|---:|---:|---:|---:|
| 0.01 | 0 | 0 | 0 | 2 |
| 0.02 | 0 | 6 | 2 | 5 |
| 0.05 | 0 | 7 | 3 | 7 |
| 0.10 | 1 | 10 | 3 | 9 |
| 0.20 | 3 | 12 | 4 | 10 |

当前简单概率拒绝规则没有证明可发布：0.05 的前组零漏过未能迁移到后组；0.01 虽拒绝本次全部混合，但同时拒绝 22/24 单人片段。不能用这一小样本上零漏过作为可靠性承诺，也不能为维持使用率自动放宽阈值。没有输出线上策略或生成可信 overlap_detected=0。

复现使用上方命令增加 `--probability-analysis` 并指定新的输出文件。下一步应使用足够连续上下文及独立真实多人标注数据验证，避免只在同一组短片段上反复调参。
