# 家居受限上下文决策实验

状态：**实验，未接入Agent、未发布为服务、未替换r14**。2026-09-29固定一次两轮CPU训练；没有根据locked评测反复调参。结果不足以支持多轮自动执行。

## 项目与逻辑边界

- Agent拥有唯一会话状态、版本和Proposal。仅由Agent从有效上下文建立有限选项；设备候选来自现有SDK/Hub目录，成功执行回执才能建立焦点。
- Laya回答一个受限`next`选择题：选某个既有目标/动作、取消，或交给LLM重新理解。它不保存历史、不规划新动作、不调用设备。pending target选择必须保留待确认动作；动作改变、新目标、开放请求一律escalate。
- LLM理解开放表达、产生新提案、澄清和改口。将来经验证的Laya续接只消费Agent持有的结构化上下文，不能新增模型私有记忆或绕过LLM擅自补动作。
- Hub仍独占执行和状态；Channel仅输入展示。即使小模型以后达标，所有提案也要经过已有SDK契约、语义版本与执行回执检查。
- 这里只复用Models现有Record、gen/assemble/train/calibrate/eval管线。没有新增路由服务、共享记忆库或运行时关键词表。

## 数据与一次实验

280条模板记录去重后262条，按话语模板家族和相同state分组：train188、val45、calib29。独立44条locked记录使用不同话语和房间，包含候选顺序交换、相同指代有/无焦点、待确认动作改变、取消与话题切换。确认eval话语与训练/验证/校准完全无重叠，家族不跨split。数据为撰写的合成种子，不是用户分布，不可用作正式产品准确率。

以r14 `laya-smart-home/45f3dedb/torch`初始化，CPU两线程、只解冻末两层和head、两轮。由val NLL选择epoch2，训练约55.9秒。仅在独立calib29条上拟合温度T=2.4527；calibration bucket均不足30条，校准样本严重不足。

| 44条locked评测 | r14原始权重 | 实验权重（校准后） |
|---|---:|---:|
| 正确 | 18/44 | 34/44 |
| 错误且top1为可执行选项 | 22 | 2 |
| p≥0.9覆盖 | 4/44 | 0/44 |
| ECE | 0.2233 | 0.4873 |

实验的两个可执行错误：“两个都不要选”被选为target_1、“刚才那个先保持不动”被选为close；概率分别0.2762、0.3414。其他错误主要将新请求/缺失指代判成取消，虽不执行也会吞掉用户需求。**不能靠提高阈值宣称通过：0.9下覆盖为零，产品没有加速收益。**已有非回退统计gate为PASS只表示相对基线未下降，不是产品准入。

下一阶段应先在开发集补齐取消/重新理解的对照分布和校准覆盖，最终另设未看过的验收集；本次locked在人工分析后已成为已知诊断集。新权重还没有单句任务保留能力回归、RKNN转换/误差验证或NPU时延验证，不可替换生产模型。

## 复核（2026-09-29，接手后）

上面“0.9下覆盖为零、没有加速收益”的结论**主要是实验设置造成的**，不能读成任务做不到；但这次实验本身也测不出能不能做到。逐项证据在 `results/20260929/review.json`。

1. **温度是被calib的构成拟合出来的，不可迁移。** calib 29条只来自4个模板家族，其中16条来自“都行”“别关主卧灯，打开客厅灯”两个家族，模型在这16条上全错（calib准确率41%）。在“大多答错”的样本上最小化NLL得到T=2.45；把它用到val上，NLL反而从0.526恶化到0.881，val里原本8条p≥0.9且全对的也被压没了。不加温度（T=1）时模型同样偏保守：locked平均置信0.49、准确率0.77。原始输出≥0.9的只有2条，而且都错——正是无焦点时“把它开一下/那就关了它”判成取消。
2. **这次校准会关掉单句任务的自动执行。** `calibrate`按题型写全局温度，并清空按选项数分档的温度：r14原来是`1.41 + {3-5: 1.82, 6-10: 1.28, 11+: 1.30}`，现在变成`2.45 + {}`。候选权重跑单句开发集，argmax准确率不变（locked-182 99.35%、v2-dev 96.70%、v3-dev 94.06%），但**0.9自动执行覆盖从76.4% / 70.2% / 43.5%全部掉到0**；现有的配对非回退gate三套都是PASS，因为它只比较准确率。续接题（4个选项）和intent题（3个选项）落在同一个`choice:3-5`档，按选项数分档也分不开两个任务。**这份`rl_agent_config.json`不能用于任何单句影子评测。**
3. **切分和覆盖有结构缺口。** 按家族切分只有29个组：val里**一条escalate都没有**（32条target、13条cancel），用val NLL选轮看不到“吞请求”这一类。无焦点时上下文不带房间，4个房间生成的记录完全相同，被去重18条，训练里只剩4条无焦点记录；“动作改口”只有1个家族；`submitted`从来没有取过true。
4. **配方太小。** 188条、只解冻末2层和head、batch 2、2轮，一共约188步；r14是15k条、22层全解冻。基线的“错误可执行22条”大多是r14对第二个选项的位置偏好（它从没学过这道题），不是有意义的对照。

契约也要先改，才能再造数据：
- 可执行选项为空时（例如无焦点）不调用Laya，由Agent按结构规则直接交给LLM。
- 取消只表示“撤回尚未提交的待确认提案”。执行成功后说“取消”，通常是想撤回刚才的动作，应当交给LLM；Agent依据回执处理。
- 焦点失效由Agent判断，Laya看不到失效焦点。

新的任务定义、数据方案、配方、门槛和停止规则见 `PLAN.md` 和 `LABELING.md`（c系列）；本节以上的实验结果只作历史证据。

## 复现

从eidolon_models根目录执行，使用已有laya虚拟环境。下面目录为本次实验输出；复现应换用新的run目录，保留历史证据。

```sh
laya/.venv/bin/eidolon-laya-train gen --scenario laya/train/scenarios/smart-home-continuation --config laya/train/scenarios/smart-home-continuation/gen.yaml --out laya/train/runs/home-continuation-20260929/seed.jsonl
laya/.venv/bin/eidolon-laya-train assemble --config laya/train/scenarios/smart-home-continuation/assemble.yaml --run-dir "$PWD/laya/train/runs/home-continuation-20260929" --out laya/train/runs/home-continuation-20260929/dataset
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 laya/.venv/bin/eidolon-laya-train train --config laya/train/scenarios/smart-home-continuation/train.yaml --dataset laya/train/runs/home-continuation-20260929/dataset --out laya/train/runs/home-continuation-20260929/checkpoint --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 laya/.venv/bin/eidolon-laya-train calibrate --checkpoint laya/train/runs/home-continuation-20260929/checkpoint --calib laya/train/runs/home-continuation-20260929/dataset/calib.jsonl --device cpu
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 laya/.venv/bin/eidolon-laya-train eval --checkpoint laya/train/runs/home-continuation-20260929/checkpoint --eval-set laya/train/scenarios/smart-home-continuation/eval/locked.jsonl --out laya/train/runs/home-continuation-20260929/candidate --device cpu
```

基线eval使用`laya/models/laya-smart-home/45f3dedb/torch`，同一locked集；报告含权重与配置sha256。原始报告、训练摘要、split清单及检查结果位于`results/20260929/`。权重留在本地ignored的train/runs，未登记到models清单、未复制到opi5max，也未改动并行的participation训练。
