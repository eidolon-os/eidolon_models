# eidolon_models

Eidolon OS 私有化部署模型仓库，用于统一管理、版本化和交付 ASR、LLM、TTS 等运行时模型资产。

## 仓库结构

```text
asr/    语音识别模型
llm/    大语言模型
tts/    语音合成模型
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
