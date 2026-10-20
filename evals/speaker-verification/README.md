# 手机注册 → 实际设备对话的声纹评测

状态：离线验证工具已实现；没有真实跨设备对照集、冻结门槛或生产准入许可。当前权重仍由 Channel 历史发布清单交付，资产迁移尚未完成。

这套工具不连接房间、不注册用户、不改 Companion 策略，也不要求 BOX-3 注册。注册样本必须来自手机。实际设备测试沿既有对话输入取得；测试原音频的采集与保留须向参与者说明，不开启常态录音。

## 采集清单

把获得同意的 WAV 与 `captures.json` 放在独立私有目录，不提交个人音频、embedding 或真实姓名到 Git。WAV 必须是单声道 PCM16、16 kHz、非空且不超过 30 秒；不会把 48 kHz 音频静默标成 16 kHz。需要重采样时应在已测试的输入适配层完成并记录配置。

示例结构（路径为待提供的真实文件，不是已存在数据）：

```json
{
  "schema_version": 1,
  "samples": [
    {
      "id": "a-enroll-1", "speaker": "person-a", "device": "mobile",
      "capture_session": "session-mobile-enroll-a", "partition": "enrollment",
      "path": "a-enroll-1.wav"
    },
    {
      "id": "a-calibration-1", "speaker": "person-a", "device": "box3",
      "capture_session": "session-box-calibration-a", "partition": "calibration",
      "path": "a-calibration-1.wav"
    },
    {
      "id": "b-test-1", "speaker": "person-b", "device": "box3",
      "capture_session": "session-box-test-b", "partition": "test",
      "path": "b-test-1.wav"
    }
  ],
  "trials": [
    {"id": "positive-calibration", "probe_id": "a-calibration-1", "enrollment_ids": ["a-enroll-1"]},
    {"id": "negative-test", "probe_id": "b-test-1", "enrollment_ids": ["a-enroll-1"]}
  ]
}
```

此最小例子故意不完整，不能报告完整准确率。实际校准和测试都需要本人及异人样本，还要有手机测试对照、不同距离/环境、短句及人数矩阵。`speaker` 使用假名；`capture_session` 是数据集全局标识，不能各设备都从 “1” 计数。

同一录音的不同片段属于同一采集会话，不能跨 enrollment/calibration/test；工具还用规范 PCM 的 SHA256 阻止换 WAV 头伪装成不同录音。元数据真实性仍需采集流程保证，工具不能从向量自动证明人员、会话或环境标注真实。

## 提取

在 `eidolon_channel` 运行：

```bash
uv run python -m benchmark.speaker_dataset /private/path/captures.json \
  --model-dir /absolute/path/to/campplus-model \
  --output /private/path/embeddings.json
```

使用模型官方前处理，输入连续 PCM，不额外拼接 VAD 碎片。逐文件验证模型 manifest 的摘要；不自动下载权重。输出中模型 ID、权重摘要、前处理版本固定，向量只能在同版本数据集内比较。

每条保留时长、RMS、峰值、直流偏移、削波比例、耗时和状态；不伪造有效语音时长。当前参考路径不执行神经 VAD，“ok”只表示数值提取成功，不表示已证明有单一说话人。数字静音和不足 1.5 秒分别标记为质量不足；该长度仅是实验记录规则，不能当作已校准的产品准入门槛。

输入路径不能越过清单目录。失败样本保留，无效/短音频不能悄悄从统计中消失；输出用 0600 创建且拒绝覆盖已有文件。

## 评分

在 `eidolon_models` 运行：

```bash
uv run eidolon-speaker-eval /private/path/embeddings.json \
  --partition calibration --threshold 0.3 0.4 0.5 0.6 0.7 \
  --output /private/path/calibration-report.json
```

这些数值只是探索曲线的示例，不是推荐阈值。确定候选方案后，冻结阈值，再单独执行 test；测试命令只接受一个阈值：

```bash
uv run eidolon-speaker-eval /private/path/embeddings.json \
  --partition test --threshold 0.6 \
  --output /private/path/test-report.json
```

评分基线为“每人注册向量归一化均值 → 对允许人列表取最大余弦”。报告按设备、允许人数拆分，并列出：

- FAR 及有分数负例的 FAR，避免大量提取失败掩盖识别质量。
- 端到端 FRR：本人因质量不足/服务失败被拒绝也算失败。
- 有分数正例的 FRR、质量不足数、提取失败率。
- 不同测试录音、人、采集会话的数量；不把同一录音配很多列表当作独立试验。

缺失正例/负例时对应指标为 null，不是 0。报告不包含向量或原音频，固定 `production_ready=false`，不产生运行时策略。置信区间暂不计算：真实独立性和足够样本量未建立前，不发布低 FAR 承诺。

## 当前验证

- 单测覆盖数据泄漏、无效向量、缺失负例、短音频计入失败、相关试验数量及多样本模板聚合。
- Channel 测试覆盖连续 PCM 不被拼接、格式/路径校验、数字静音与短音频、模型失败脱敏，以及有界音频证据的缺帧/溢出/换源。
- 数值回归：本地合成语音 8.191 秒，官方 WAV 与连续 PCM 输出 192 维向量，最大绝对差 0；不是真人准确率测试。
- Pi5 只读使用已有手机试说 7.920 秒，官方连续音频提取冷请求 15,342.54 ms；五次热请求 505.68、537.12、543.66、562.10、550.40 ms，重复向量最大绝对差 0。未导出个人音频或向量；无并发压测，不能视为 P95 或上线验收。

下一步需要独立的手机测试、实际设备本人/异人测试证据；当前仅有一份已保存手机试说，不能据此选择阈值或恢复门禁。
