# c5–c10 复跑入口

当前候选为c10 `e8254243`，Mac模型门槛、冻结合成验收和本地服务验证通过；存在已知历史新增误触发与局部分组退化，详见C10-RESULTS.md。生产固定版本未修改，OPI5 Max/NPU由用户延后。

## 先恢复原始证据

原始JSON/JSONL、日志、实际数据和失败实验放在Git LFS的`evidence.tar.gz`，初始化权重单独保存在`initialization/c9`，候选在`laya/models/laya-smart-home/e8254243`。取得LFS对象后，从仓库根目录执行：

```sh
E=laya/train/scenarios/smart-home-continuation/results/20261002-c5
(cd "$E" && shasum -a 256 -c evidence.sha256)
tar -xzf "$E/evidence.tar.gz" -C "$E"
```

包内`evidence-files-sha256.json`逐文件固定原始证据；`initialization/c9/SHA256.json`固定初始化。c10的replay保存实际dataset/data/baseline-new；c9的replay/baseline-c4保存门槛使用的旧基线报告。c10无需旧runs目录即可按下文复跑。早期c5–c9命令作为历史实验入口保留，其中旧初始化依赖不等于当前候选的清洁检出复跑依赖。模型训练不承诺跨Metal/PyTorch版本逐位相同；实际版本在c10/environment.json。

从 Models 仓库根目录运行。使用已有 `laya/.venv`，无需改动生产服务。训练前登记见同目录 PLAN.md；`data/` 是本轮冻结输入，原始 Agent 报告的逐文件哈希见 source-audit.json。

## 数据与训练

选择一个尚不存在的 run 目录，避免覆盖 c5 原始记录。

```sh
R=laya/train/runs/c5-reproduction
mkdir -p "$R/data"
cp laya/train/scenarios/smart-home-continuation/results/20261002-c5/data/* "$R/data/"
laya/.venv/bin/eidolon-laya-train assemble \
  --config laya/train/scenarios/smart-home-continuation/assemble-c5.yaml \
  --run-dir "$R" --out "$R/dataset"
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 laya/.venv/bin/eidolon-laya-train train \
  --config laya/train/scenarios/smart-home-continuation/train-c5.yaml \
  --dataset "$R/dataset" --out "$R/checkpoint" --device mps
```

原有 c4/r14 来源目录是既有 c 系列管线的依赖；`dataset-manifest.json` 固定其路径和 SHA-256。本轮没有把已挖掘的六句混入训练。不同 run 目录会改变 manifest 的绝对路径，不改变实际 Record 的 state/questions/labels 与固定分组。

补充数据生成可单独复核：`tools/target_pairs.py --out <空目录>` 与 `tools/target_accept.py --out <同目录>`，得到的五份新数据 JSONL 已验证与 `data/` 逐字节相同。MPS 训练使用固定 seed，但不承诺跨 PyTorch/Metal 版本的权重逐位相同。

## 基线、校准与开发回归

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 laya/.venv/bin/eidolon-laya-train eval \
  --checkpoint laya/models/laya-smart-home/7b695ba8/torch \
  --eval-set "$R/data/mined-regression.jsonl" --eval-set "$R/data/target-dev.jsonl" \
  --out "$R/baseline-new" --device mps
```

训练结束后，从 `laya/` 下运行：

```sh
bash train/scenarios/smart-home-continuation/tools/finish_target_run.sh train/runs/c5-reproduction mps
```

该脚本复用现有 items、eval、calibrate、calib_split、swap 与 cmetrics/agent_replay 语义，保存 raw logits、温度前后配置、全部逐题结果和新增误执行。不会运行新验收集、启动服务或发布。

`eval`/`items` 的输入就是 Record：取每行 `state` 和 `questions` 即是 `/v1/systemone` 请求。模型不保存会话或执行设备。单句执行统计使用0.8、续接使用0.95、pick取消使用0.5；这是 Laya 层可执行上限，不替代 Agent 的目标校验、动作解析或 Hub 执行验收。

新 target-dev 另有命令构造复核，直接读取基线 Agent `c22ddd1` 的原始 `_control` 函数、动作表和 lexicon，使用 SDK `c375645` 的 DTO/设备能力定义。临时文件与导入仅用于离线评测，不修改 Agent/SDK，也不调用服务。运行如下，c5 比较时把 `--report` 换成对应 eval 路径：

```sh
laya/.venv/bin/python laya/train/scenarios/smart-home-continuation/tools/target_commands.py \
  --agent-repo ../eidolon_agent --sdk-repo ../eidolon_sdk \
  --records "$R/data/target-dev.jsonl" \
  --report "$R/baseline-new/target-dev.json" \
  --out "$R/baseline-new/target-dev.commands.json"
```

基线668条中，347条超过续接执行阈值（104条错误）；其中317条能构造 Proposal（74条错误）。命令构造仍不等于 Hub 接受或真实设备执行。冻结门槛继续使用更保守的104条模型层口径，不以动作映射拦截放宽门槛。所用源码哈希和逐条 Proposal 保存在 `baseline/target-dev.commands.json`。

## 冻结验收与后端验证

只有开发门槛和历史回归完成后，才在选定权重/校准上运行 `target-accept.jsonl`、`target-accept-single.jsonl`、`controls-accept.jsonl` 一次。44条目标话语派生132个上下文对照，另有44条单句和40条控制；对照相关，不能把216条当作216个独立真实用户样本。

候选打包沿用 `eidolon-laya-train package`，revision 取实际 safetensors SHA-256 前8位；NPU 沿用 `eidolon-laya export-npu` 和 `deploy/rk3588/laya_npu.py convert/run`。具体模型目录、权重/校准哈希、评测结论、板上复跑命令在结果完成后补入交接文档。当前这份复跑说明不表示模型已经通过验收或导出。

## c6 / c7 / c8后续实验

c6复用c5数据，从c4开始一轮低学习率训练；c7是固定50/50参数合并；两者均失败。逐轮预登记与结果分别在C6-PLAN/RESULTS、C7-PLAN/RESULTS中，不能将其中任一称作已验证候选。

c8使用同一份数据，从c7开始，添加训练期冻结c4教师约束。配置为train-c8.yaml，精确代码、输入哈希、smoke证据在c8目录。原始命令从laya目录执行：

```sh
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/eidolon-laya-train train \
  --config train/scenarios/smart-home-continuation/train-c8.yaml \
  --dataset train/runs/c8/dataset --out train/runs/c8/checkpoint --device mps
bash train/scenarios/smart-home-continuation/tools/finish_target_run.sh train/runs/c8 mps
```

复跑须使用新的输出目录，不能覆盖原始证据。c8训练开始不代表回归通过。

c8最终结果：9d9faaa7，训练与完整finish脚本正常退出，但历史单句误执行9>8，未通过。详见C8-RESULTS.md及c8/gate.json。新验收、打包和NPU命令仍未执行，不存在本轮可部署候选。

## c9修复与提交前验证

用户追加授权后，按C9-PLAN.md开展新的数据对照实验，c8失败结论保留。从仓库根目录执行以下命令可生成新run（输出目录必须不存在）：

```sh
laya/.venv/bin/python laya/train/scenarios/smart-home-continuation/tools/prepare_intent_run.py \
  --base laya/train/runs/c8 --out laya/train/runs/c9-reproduction
```

工具将旧拆分逐条保留，仅追加分组后的购物/禁止/转述与正控制对照，写入新的manifest及旧manifest副本；训练、验证、校准三份JSONL已验证可逐字节复现。从laya目录运行train-c9.yaml（初始化c8，冻结encoder）和同一个finish_target_run.sh。run中存在intent-dev.jsonl时，finish脚本自动增加该集合；其c4基线须先通过eval生成到baseline-new。当前c4/c8在新增128条开发题上均无误执行、控制覆盖100%，因此该集合只是能力保持检查，不能作为已修复历史困难表达的证据。

新增回归测试：tests/test_intent_pairs.py、tests/test_board_harness.py。后者用模拟运行时确认home-collide同时请求、可配置预算，以及NPU三次预热不计入正式题项耗时；这些测试本身不提供实际板上时延结论。

## c10 Mac复跑（已通过预登记门槛，残留问题见结果）

c9结果见C9-RESULTS.md，c10预登记见C10-PLAN.md。c10保留c9原数据后追加多分句对照：

```sh
laya/.venv/bin/python laya/train/scenarios/smart-home-continuation/tools/prepare_intent_run.py \
  --base laya/train/runs/c9 --out laya/train/runs/c10-reproduction --composition
```

c9/c10数据三份拆分分别已验证逐字节复现。无需旧runs目录的复跑方式是使用随证据保存的c10/replay/{dataset,data,baseline-new}，将其复制到新的run目录；初始化使用initialization/c9，教师使用已跟踪的生产c4。初始化c9自身未通过，保留目的仅是精确复现训练。

从laya目录运行（R必须是新的输出目录；E指本结果目录）：

```sh
E=train/scenarios/smart-home-continuation/results/20261002-c5
R=train/runs/c10-reproduction
mkdir -p "$R"
cp -R "$E/c10/replay/dataset" "$E/c10/replay/data" "$E/c10/replay/baseline-new" "$R/"
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/eidolon-laya-train train \
  --config train/scenarios/smart-home-continuation/train-c10.yaml \
  --init "$E/initialization/c9" --dataset "$R/dataset" --out "$R/checkpoint" --device mps
bash train/scenarios/smart-home-continuation/tools/finish_target_run.sh \
  "$R" mps "$E/c9/replay/baseline-c4"
.venv/bin/python train/scenarios/smart-home-continuation/tools/check_target_gates.py \
  --run "$R" --baseline "$E/c9/replay/baseline-c4" \
  --adjudication "$E/c15-adjudication.json" --out "$R/gate.json"
```

门槛工具通过退出0，失败退出1。finish不会运行新封存验收。用户最新范围优先：只做Mac FP32训练和测试，不执行前文的NPU导出或板端步骤；满足Mac要求后通知用户，再决定后续。

## 已冻结候选的 Mac 服务验证

```sh
cd laya
HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2 .venv/bin/eidolon-laya \
  --service smart_home --model-dir models/laya-smart-home/e8254243 \
  serve --backend torch --device mps --host 127.0.0.1 --port 18771
```

先检查测试端口空闲；使用独立端口，不替换已有服务。请求端设置`NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost`，避免macOS系统代理将localhost转发。`tools/service_check.py --pair <records> <offline-report> --save <run>/service-responses`可逐题回放；`tools/check_service_replay.py --run <run>`复核全部保存答案与原阈值决策。原始首个直连响应另存service-first-direct-response.json，模型载入时长另列于服务日志，均不混入稳态统计。

本轮实际回放mined-regression、新验收三组、c-dev、target-dev、历史locked-accept，共2,676条请求。CPU/MPS比较的是相同冻结权重，310个题目答案；不是对新验收再做候选选择。原216条现在已经使用过，未来训练不能再宣称它是未见过的盲测。

并发使用`deploy/rk3588/concurrency.py run <fixture-dir> home-alone`和`home-collide --pairs 40 --home-url http://127.0.0.1:18771/v1/systemone`，测试仅含home.jsonl；`report --home-budget-ms 1000`生成报告。原始40条fixture和全部80次并发响应在c10/mac-concurrency。测试后停止自己启动的服务；不修改生产固定版本，也不执行板端命令。
