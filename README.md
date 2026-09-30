# eidolon_models

Eidolon OS 私有化部署模型仓库，用于统一管理、版本化和交付 ASR、LLM、TTS 等运行时模型资产。

当前已包含中文流式/离线 Paraformer INT8 ONNX 与 CT-Transformer 标点模型，以及供 macOS
arm64、树莓派 5 arm64 和未来 RK3588 共用的 `eidolon-asr` 2-pass 推理服务与测试入口。详见
[asr/README.md](asr/README.md)。

## 仓库结构

```text
asr/    语音识别模型
llm/    大语言模型
tts/    语音合成模型
laya/   laya 决策模型服务（独立 uv 项目：PyTorch / ONNX 两个后端，HTTP API，见 laya/README.md）
jevk5/  JevK5 决策模型独立 uv 项目：本地推理、合成数据审核与小规模适配训练
evals/  跨模型参与决策评测协议与冻结数据清单
```

建议每个模型使用以下目录结构：

```text
<类型>/<模型名>/<版本>/
├── README.md       # 模型用途、来源、输入输出和部署要求
├── LICENSE         # 上游模型许可证（如适用）
├── checksums.txt   # 模型文件校验值
├── config/         # 推理配置、tokenizer、词表等
└── model/          # 权重与运行时模型文件
```

模型目录名应稳定且可被部署配置直接引用；版本目录建议使用上游版本号、发布日期或内容摘要，避免使用 `latest`。

## 添加模型

1. 确认模型许可证允许目标私有化部署场景与再分发方式。
2. 在对应类型目录下创建独立的模型和版本目录。
3. 在模型目录的 `README.md` 中记录来源、上游版本或 commit、转换流程、运行时、硬件要求、量化方式和已知限制。
4. 保留上游 `LICENSE`、`NOTICE` 或其他归属声明；第三方模型不会因进入本仓库而改变其原始许可。
5. 为所有交付文件生成并提交 SHA-256 校验值：

   ```bash
   shasum -a 256 <文件> > checksums.txt
   ```

6. 提交前在目标推理运行时中完成加载和最小推理验证。

## Host 能力与验证

### 按 Host 选择模型服务

`eidolon_ops/config/eidolon-rk3588.toml` 和 `eidolon_ops/config/eidolon-pi.toml`
分别配置 Orange Pi 5 Max 和 Pi 5。`[capabilities].provides` 中的
`local_asr`、`local_llm`、`local_tts`、`local_laya` 决定本地安装的模型服务、
工件和健康检查；`[services].units` 必须列出对应的 systemd unit，启用任一
本地模型时还须在 `[sources.eidolon_models]` 指向本仓库。Ops 在发布前校验三者
一致，切换后停止从配置中移除的旧模型服务。

发布包在归档时按能力排除未启用的 ASR/TTS 模型目录，运行环境也只安装已启用服务
需要的 Python extra。独立工件（如 LLM、Laya 权重）只为已启用能力传输。
成功激活并清理旧发布后，Ops 会检查已取消选择的旧工件：仅当目录仍与组件契约
中的固定摘要完全一致、相关服务已停止时才删除；检查不通过会保留目录并在发布
结果中说明。dry-run 和健康检查失败时不会删除旧模型。

当前两台 Host 均只选择 `local_laya`，因此只有 `eidolon-laya.service` 作为本地
模型服务启动。RK3588 的 `rknpu2` 仍描述硬件能力；Laya r14 当前运行 ONNX CPU
后端。ASR、TTS 和会话 LLM 的本地服务不启动，调用仍遵照各 Host 的 Provider
配置。Pi 5 需恢复网络可达后才能部署并验证这份配置。

Laya r14 权重不在 Git 中。发布工作站须先备齐 Ops 配置所引用的冻结工件源，
由组件契约校验逐文件摘要；换工作站时要搬运相同字节。只修改能力声明不会
绕过工件校验。

- [HOST-RK3588.md](HOST-RK3588.md)：**Orange Pi 5 Max（RK3588）的实测档案**。系统与 NPU、
  内存带宽、funasr 2pass、Qwen3-1.7B、prompt 前缀缓存、CosyVoice2 分组件与端到端、
  Kokoro 对照、bge，以及 ASR/LLM/TTS/bge 的联合压测与 CPU/NPU 分配方案。开头有结论速查。
- [HARDWARE.md](HARDWARE.md)：三台 Host 的横向对照表，各机明细分文件。
- [VERIFICATION.md](VERIFICATION.md)：ASR / TTS / LLM 在 RK3588 上的验证方案，含前置条件、
  分阶段门槛和放弃判据。

两个结论会直接影响选型，先读再动手：**内存带宽 20 GB/s 是本地 LLM 的硬天花板**（与 6 TOPS 无关）；
**RK3588 的 A76 比树莓派 5 低 6%**，所以 RKNN 通路不是优化项，而是这块板唯一的立项理由。

## ASR 快速开始

```bash
./scripts/eidolon-asr doctor
./scripts/eidolon-asr test
./scripts/eidolon-asr serve
```

默认健康检查为 `http://127.0.0.1:8768/readyz`，流式接口为
`ws://127.0.0.1:8768/v1/stream`。

## 大文件管理

常见模型权重、运行时文件和归档文件已通过 [Git LFS](https://git-lfs.com/) 管理。首次使用前安装并初始化 Git LFS：

```bash
git lfs install
git lfs pull
```

不要提交模型下载缓存、推理缓存、运行日志、临时转换文件或任何凭据。提交模型前还应确认远端 Git LFS 的单文件大小、存储和流量配额满足交付要求。

## 安全与合规

- 不得提交访问令牌、私钥、用户数据、训练数据或含敏感信息的样本。
- 模型来源必须可追溯，且应记录已知的使用限制与合规要求。
- 对外发布或用于商业环境前，应分别审查仓库内容与每个第三方模型的许可证。

## License

Copyright © 2026 Li Jinsong.

本仓库中由 Li Jinsong 持有版权的材料，允许依据 [PolyForm Noncommercial License 1.0.0](LICENSE) 进行许可范围内的非商业使用。商业使用需要另行取得书面授权，请联系 [lijinsong@aimanthor.com](mailto:lijinsong@aimanthor.com)。

第三方模型、权重、配置、词表、代码和其他材料保留其原始许可证；仓库级许可证不会重新许可这些材料。详情见 [LICENSING.md](LICENSING.md) 与 [NOTICE](NOTICE)。
