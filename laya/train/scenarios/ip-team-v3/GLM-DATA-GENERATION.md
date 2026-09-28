# IP Team 文本数据：GLM-5.3-Flash 草稿生成

本脚本只为 IP Team 小脑生成**可审稿的 text 数据草稿**。它不会调用 ASR，不修改 r14 权重，不自动并入 train/val/calib/comparison/holdout，也不会在缺少 API Key 时发请求。已封存评估集只用于本地去重检查，其内容不进入提示词。

## 配置与运行

填入本目录被 Git 忽略、权限为 600 的 `.env`：

```dotenv
EIDOLON_IP_DATA_API_KEY=你的普通APIKey
EIDOLON_IP_DATA_BASE_URL=https://open.bigmodel.cn/api/paas/v4
EIDOLON_IP_DATA_MODEL=glm-5.3-flash
EIDOLON_IP_DATA_REASONING_EFFORT=low
```

上面的默认地址来自[智谱 HTTP API 文档](https://docs.bigmodel.cn/cn/guide/develop/http/introduction)。若 API Key 确实来自 Z.ai 国际平台，才改为 `https://api.z.ai/api/paas/v4`。本任务使用**普通 API**；脚本只接受这两个官方 HTTPS 主机和通用端点。GLM-5.3-Flash 的官方模型码和文本参数见 [Z.ai 模型文档](https://docs.z.ai/guides/vlm/glm-5.3-flash)。

从仓库根目录执行：

```bash
# 不需 Key、不写文件、不付费：预览下一批的切片与请求数量
laya/.venv/bin/python laya/train/scenarios/ip-team-v3/generate_glm.py --dry-run --count 4

# 填好 .env 后，每次默认只发起一次付费请求
laya/.venv/bin/python laya/train/scenarios/ip-team-v3/generate_glm.py

# 明确需要多批时，每次最多 8 次；同一输出目录会按计划续跑
laya/.venv/bin/python laya/train/scenarios/ip-team-v3/generate_glm.py --count 4
```

`generation-plan.yaml` 预设 27 次请求，其中 4 次为双分支反事实，理论最多 31 个独立候选家族；失败或被拒稿也计入请求配额，防止自动重试产生意外费用。脚本按切片轮询，逐次请求，`max_tokens=4096`、低推理强度，不并发。当前 v2 使用扁平 JSON 素材，由本地代码构造决策步骤，再做结构、去重、窗口检查；仍需人工审查语义金标。需要新一版配额时更新计划 `version` 并指定新 `--out` 目录，避免覆盖旧产物。

输出默认在 `laya/train/private/ip-team-v3-glm-flash-v2/`，不提交 Git。先前的 v1 API 冒烟记录保存在 `laya/train/private/ip-team-v3-glm-flash/`，不会覆盖：

- `calls/*.json`：每次请求的状态、提示词哈希、响应 ID、供应商 token 用量；不写入密钥。
- `responses/*.txt`：模型原始输出，便于审查被拒稿的原因。
- `drafts/*.yaml`：通过结构检查、旧数据精确去重、现有 `episodes.generate` 展开及 2048-token 完整性检查的候选对话家族。**这只是机器可读草稿，语义金标仍需人工审稿。**

若进程在发送请求后中断，`calls` 会留下 `pending`，下一次运行会停止，避免不确定该请求是否计费时重复发送。API 错误或格式不合法只记录一次并停止，不自动补发。请检查 `calls` 和原始响应后再决定下一步。

审稿时逐步确认用户约束、已公开成员文本、唯一下一步、角色槽位、等待/结束边界与反事实分支是否成立。只有审定的家族才能进入新的、版本化的**训练** YAML 和下一轮训练配置；不要把模型自标的答案直接当金标，也不要改写已封存的 comparison/holdout。`generation-plan.yaml` 与脚本可按新的错误诊断调整，但每次修改须保留版本和来源记录。

## 故事角色团实验的分目录用法

官网 `ensemble` 场景中的候选是**故事角色本人**，不是写作或运营团队。本目录 `generation-plan-ensemble.yaml`、`generation-plan-ensemble-safety.yaml`、`generation-plan-ensemble-exclusive.yaml` 与两个单独的检查集计划分别管理角色对话与收束边界；每个计划必须对应独立的 `--out` 原始目录、审稿清单和 `train/data/generated/ip-team-v4/` 下的版本化审定目录。例如：

```bash
laya/.venv/bin/python laya/train/scenarios/ip-team-v3/generate_glm.py \
  --plan laya/train/scenarios/ip-team-v3/generation-plan-ensemble-safety.yaml \
  --out laya/train/private/ip-team-v4-glm-ensemble-safety --dry-run

# 审定每次调用后，再整理已明确接受的家族；拒稿保留在私有原始目录。
laya/.venv/bin/python laya/train/scenarios/ip-team-v3/curate_glm.py \
  --review laya/train/scenarios/ip-team-v3/review-ensemble-safety.yaml \
  --dest laya/train/data/generated/ip-team-v4/ensemble-safety-reviewed-2 \
  --namespace safety_
```

真实调用须去掉 `--dry-run` 并按需要显式设置 `--count`，每次最多 8 次。新的产品场景检查集必须另用自己的计划、输出目录和 ID 命名空间；已用于诊断的合成检查集不再被称作独立验收集。已完成的轮次、来源、门槛和失败记录见 [IP-V4-CHARACTERS-EXPERIMENT.md](IP-V4-CHARACTERS-EXPERIMENT.md)。
