# c10 生产切换记录（2026-10-09）

**OPI5 Max 生产已切换为 c10 `e8254243`，RK3588 NPU FP16。** 正式release为 `rk3588-laya-c10-20261009-1`；生产端口仍是127.0.0.1:8771。2026-10-09 11:21:45（北京时间）启动日志确认新权重载入，模型加载约4.2秒。

## 发布范围与验收

通过现有Ops prepare/dry-run → resume/activate，采用reversible模式。没有数据库迁移，没有重新训练，没有更换Host身份或Owner授权。发布前配置/Engine/NPU测试46项通过。

| 组件 | 本次生产提交 |
|---|---|
| Models | `882fc1f` |
| Agent | `fc9de67` |
| SDK | `acf5ce6` |
| Hub | `83cf4bd` |
| Channel | `ba5f8cf` |

Kernel/Data/Admin/Memory保持原生产提交，九仓完整矩阵见 [matrix.json](matrix.json)。已与板上实际 [release.json](deployed-release.json) 逐项核对一致；工作区未提交的其他功能没有入包。

配套生效的行为：Laya优先解释，默认最近3轮上下文，向LLM交接候选证据，相对风速按当前状态增减，命令FIFO及分阶段耗时记录。

- Ops激活结果：`activated / applied`。
- release doctor：`healthy`。
- App-ready：`app_ready`，全部检查为true，包括Channel worker、LiveKit连接、Hub接入和参与模型就绪。
- 安装的Agent命令代码SHA与固定提交一致；配置确认`interpreter=laya`、`context_turns=3`，SDK含`fan_speed.step`。
- 三次生产纯推理探针均返回c10、`features.question_state=true`、无截断，耗时约455/989/400ms。这些请求没有接入执行器，未操作家电。

原始证据：[prepare](prepare.json)、[activation](activation.json)、[生产探针](smoke.json)。脚本 [smoke.py](smoke.py) 只读取配置及调用模型推理，不执行设备。

## 回滚

前一生产release为 `rk3588-memory-policy-20261008-2`，精确来源见 [previous-release.json](previous-release.json)。c4制品路径为 `/var/lib/eidolon/models/laya-smart-home-c4-rknn-7b695ba8`，权重继续保留。

Ops在成功发布后自动回收了原release代码目录；reversible模式只保证切换事务期间的回退，并不覆盖成功后的默认清理策略。为保持后续可回滚，本次额外按原九仓提交重新prepare原release，不激活它。当前c10不受影响。重新prepare结果为`dry_run`，只读回滚计划为`rollback_planned`；最终已核对旧代码目录、c4制品都存在，生产仍为c10。见 [回滚包准备](rollback-prepare.json)、[只读回滚计划](rollback-plan.json)、[最终状态](final-state.txt)。没有实际执行回滚。后续发布仍可能触发清理，应重新核对回滚包可用性。

本次代码快照：`/var/lib/eidolon/deployments/rk3588-laya-c10-20261009-1-11e5b7a2c8e341398c19fb6cad832e08`。

本次Host配置快照：`/var/lib/eidolon/deployments/rk3588-laya-c10-20261009-1-host-98835fa5be9f4a78a9ff408bbaf98e67`。

Ops只读回滚计划入口（不带`--apply`）：

```sh
./eidolon rk3588 rollback --release-id rk3588-laya-c10-20261009-1 \
  --snapshot /var/lib/eidolon/deployments/rk3588-laya-c10-20261009-1-11e5b7a2c8e341398c19fb6cad832e08
```

## 已知边界

1秒解释预算保持不变。隔离NPU测试中长上下文最慢请求30次有1次超过1秒；生产纯推理同例约989ms。此类情况继续交LLM兜底，没有通过放宽deadline掩盖。详细测量见 [NPU验证](../c10-npu-20261009/README.md)。

本次完成正式发布、服务就绪和生产推理验收，**没有冒充完成Korvo真实口述或真实LLM/家电端到端验收**。下一次对Korvo说话将使用新生产版本，仍需结合实际对话日志评价体验。
