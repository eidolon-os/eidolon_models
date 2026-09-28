# opi5max Laya NPU 验证 — 2026-09-29

已复用 Models 的 `RknnBackend`、板上 r14 RKNN 产物和 RKNN Toolkit Lite2 2.3.2，运行现有 HTTP 服务；没有重写后端、转换模型或修改训练。

- revision：`45f3dedb`；FP16；RK3588 三核，默认 placement，128/256/384/512 长度档。
- 在线 `eidolon-laya.service` 仍是 ONNX CPU FP32；NPU 使用独立 loopback 18772 测试端口。
- 测试依赖：现有 Python 3.12 RKNN 实验环境增加 pydantic；正式服务的 Python 3.13 环境未改变。正式部署需由 Models/Ops 管理兼容的 RKNN 环境，不能引用 `/root` 的实验目录。
- Mac 经 SSH 隧道使用现有 Agent `LayaInterpreter`，真实 18 设备 + 4 场景候选；没有控制任何设备。
- 同四个输入完成两轮各 24 次请求。保存第二轮原始结果：p50 **359.75 ms**，nearest-rank p95 **616.1 ms**，最小329.9、最大634.0 ms；24/24 < 800 ms。这是少量重复输入下的单客户端结果，不是并发/长稳态性能保证。
- 同四条 CPU 基线各一次为2364–2968 ms；不同采样次数，只可视为明确的速度改善，不构成完整基准。
- 第二轮全部提案与对应 CPU 基线一致，置信概率略有FP16数值差异。关闭主灯的歧义和调亮20%的低动作概率仍需要兜底，NPU不会改善语义准确率。
- 推理期间 `/sys/kernel/debug/rknpu/load` 采样：Core0 18–76%、Core1 18–77%、Core2 15–39%，确认实际使用NPU。
- 长度边界：819-token 单问返回 HTTP400，`sequence ... exceeds the largest NPU bucket 512`。服务模型配置max_len1024不等于导出图支持1024；不得静默截断后执行。扩大模型输入或接入新上下文任务须重新验证档位。

复现：使用 `eidolon_models_laya.cli serve`，backend=rknn，模型目录含同revision的manifest、tokenizer/config及npu产物；通过现有 Agent 适配器调用 `/v1/systemone`。本次没有改变服务默认配置。实验服务和隧道在完成测试后关闭。

结论：当前家居短输入在opi5max NPU上已有使用价值，可以推进架构实现；上线接管还需要发布产物/运行环境收口、真实负载验证。当前r14的上下文任务仍维持LLM处理，不能由性能通过推导出新语义能力通过。

## Python 3.13 / C Runtime 验证

使用正式服务的 CPython 3.13，经薄 ctypes 适配调用相同 RKNN Runtime 2.3.2；保留既有分桶、三核调度、CPU embedding/scorer 和 HTTP 协议。SDK 与 pydantic 在临时目录/既有环境中提供；未修改在线 venv。

- `native-npu-proposals.json`：同一组 24 次请求，p50 335.85 ms、nearest-rank p95 430.0 ms、最大500.5 ms；24/24 < 800 ms。
- 24/24 的 proposal 与相应 CPU 基线一致；本测试未执行设备动作。
- Runtime SHA256：`d31fc19c85b85f6091b2bd0f6af9d962d5264a4e410bfb536402ec92bac738e8`。
- Models 发布契约固定 RKNN 模型及 Runtime 文件；Ops 沿用原有产物传输与校验流程。两种格式暂同时保留，支持显式回退；有 local_laya 的 Host 会携带两份格式。launcher 由既有 rknpu2 能力声明选择 NPU，加载失败不暗中回退 CPU。
- 服务有效 max_len 受 NPU 最大档位512约束；通用接口按既有协议报告截断，Agent 对任何截断结果弃权，避免基于不完整输入控制设备。

## 正式服务用户验证与负载边界

候选发布 `rk3588-home-npu-20260929-2` 的正式 systemd 服务由 eidolon 用户启动，模型与 Runtime 只读 `/var/lib/eidolon/models` 中的固定产物。max_len512，三核负载采样可见。

- 串行24次：无错误，p50 453.7ms，p95 562.6ms，最大620.7ms，24/24低于800ms。
- 双并发12次：无协议错误，但p50 819.65ms、p95/最大952ms，只有3/12低于800ms。单请求已利用三核，HTTP工作线程串行，额外排队无益于交互预算。
- 因此 NPU launcher 默认 max_pending=1，复用现有503忙碌拒绝和 Agent 的LLM兜底，不放宽800ms预算；显式环境配置仍可覆盖。
- 本候选整机发布因 Data 既有迁移入口未被启动流程调用而健康失败，已自动回滚。NPU验证成功不等于整机发布成功，后续修复结果另记。

## 最终发布与准入验证

`rk3588-home-npu-20260929-3` 已成功激活并通过 Ops doctor / app_ready 门禁。Data 通过自己的 Alembic 入口升级；使用已有 forward-only 屏障，保留升级前 Data 与Owner Authority备份。Laya由eidolon用户运行，backend=rknn，max_len512，max_pending1。

最终24次串行请求：服务端p50 **406.6ms**、p95 **505.2ms**、最大529.3ms；跨机观察p50 **422.85ms**、p95 **1601.1ms**、最大2508.1ms，22/24低于800ms。两次异常的服务端耗时分别529.3ms和437.6ms，额外等待发生在服务端计时之外；不能仅凭此断定是网络，也不能宣称所有跨机请求达标。该轮与发布收尾重叠，结果如实保留。

六组双并发：6次被接纳（p50 416.5ms、最大466.3ms），6次立即503 busy（p50 15.45ms、最大23.1ms）。503经既有Agent适配器映射为UNAVAILABLE，进入既有LLM兜底；此测试没有测量LLM完成时间。800ms预算未放宽，模型阈值未降低。

Mac实际8771调用入口已临时转到正式NPU实例完成验证，随后恢复本地Laya（torch/MPS）；全部本轮测试隧道已关闭。远端正式服务持续使用NPU。该记录包含性能边界，不是长稳态或整体对话质量验收。
