# 中文智能家居控制训练数据：能找到什么

快照 2026-09-26。目标：给决策模型（是不是命令 / 哪台设备 / 什么动作，含"多设备"和"家里没有"两个出口）找中文训练数据。
搜了 HF、魔搭、GitHub、比赛（SMP / 讯飞 / CCF）、大厂开放平台、论文。凡实际下载核实过的数字都写明；"未核实 / 未找到"如实标注。

**结论**：没有一个现成中文集同时有"设备清单 + 多设备 / 场景 + 设备不存在 + 非命令负例"。可用组合是
**zhihao666/smartHome（有清单）+ SCENIC（有多设备 / 隐式 / 场景）+ Home Assistant zh-CN 意图模板 + MASSIVE zh-CN（iot 子集当正例、其余 17 个场景当非命令负例）**，
再借 HomeBench（英文）的规则造"设备不存在 / 多设备混合"。两个出口和非命令负例**必须自造**——这正是我们训练框架里出口注入和 llm 生成器的活。

## A. 中文、可直接转训练数据

| # | 数据集 | 规模 | 形式 | 有什么 | 缺什么 | 许可 |
|---|---|---|---|---|---|---|
| 1 | [zhihao666/smartHome](https://huggingface.co/datasets/zhihao666/smartHome) | train 5,000 + val 500 | 单轮：`用户指令 + 设备列表 [{deviceName, deviceTypeName, floorName, roomName}]` → `<think>` 三步推理 + `control_device` 调用 | **每条一份随机家庭设备清单**；13 类设备（新风 / 灯 / 插座 / 投影 / 车库门 / 空调 / 地暖 / 晾衣架 / 扫地机 / 风扇 / 窗帘 / 除湿 / 加湿）+ 未知类型；动作 TurnOn 2,392 / TurnOff 1,095 / SetTemperature 584 / Pause 288 / Open 285 / Close 268 / SetLevel 88；设备命名是全屋智能行业口味（香格里拉帘、风管机、分集水器） | 无负例、无拒绝、无多设备、无"不存在"；多候选消歧策略是"默认选第一个"（隐患）；README 空、来源不明 | Apache-2.0，HF 直下 |
| 2 | [SCENIC / Smart Home Instruct](https://github.com/huluk98/SCENIC)（清华深研院，[论文](https://arxiv.org/abs/2606.22296)） | SFT 9,772；对比组 9,772（anchor / positive / hard-negative）；Bench 200 | 单句 → 中文确认回复（"好的，已关闭书房加湿器。"），槽隐含在 1,411 个规范回复里，可规则反解 | **多设备 / 多动作 / 隐式指令 / 场景标签**（call_meeting / cooking / gaming）；设备：空调 2,213 / 电视 1,711 / 灯 1,662 / 净化器 834 / 风扇 810 / 窗帘 788 / 插座 532 / 摄像头 525 …；房间：客厅 2,080 / 卧室 1,146 / 书房 1,066 / 厨房 661 / 玄关 445 / 阳台 414；动作含定时、模式、亮度色温、频道、音量；**9,772 组"同设备异参数"困难负例** | 无设备清单、无闲聊、无"不存在" | MIT，git clone |
| 3 | [Home Assistant 意图 zh-CN](https://github.com/OHF-Voice/intents/tree/main/sentences/zh-CN) | 45 意图；约 756 行模板（可展开成海量句）+ 约 968 条带槽测试句 | hassil 模板 + 测试句（name / area / domain / brightness / temperature …） | 意图槽最规范；覆盖开关 / 亮度 / 温控 / 风速 / 窗帘位置 / 音量 / 媒体 / 扫地机 / 状态查询 / 天气 / 计时；`HassNevermind`（"算了"）；真实用户在用、有反馈 | 无闲谈负例；"不存在"在运行时产生 | CC-BY-4.0，2026-09-25 仍在更新 |
| 4 | [MASSIVE zh-CN](https://huggingface.co/datasets/AmazonScience/massive)（Amazon） | 每语言 16,521（train 11,514 / dev 2,033 / test 2,974） | 单句 + 场景 / 意图 / 槽，zh-CN 由母语者本地化 | iot 场景 9 意图（灯开关 / 调亮暗 / 变色 / 扫地 / 咖啡 / 插座开关）；**其余 17 个场景是同一批人写的同风格非家居句**——最干净的"不是控制命令"负例 | 设备只有灯 / 插座 / 咖啡机 / 扫地机 | CC-BY-4.0 |

## B. 中文、只能评测或参考

- **Reject or Not?**（美的 AI 研究院，[arXiv 2512.10257](https://arxiv.org/abs/2512.10257)）：11,913 条真实线上日志脱敏，13 类拒识（唤醒词 / 非人声 / ASR 乱码 154 / 非面向助手闲聊 / 语义不合理命令 232 / 可答闲聊 1,093 / 支持的命令 9,872 …），Qwen-2.5-3B 微调 96.44%。**最贴合"是不是控制命令"的中文真实数据，论文说开源但没有下载链接**——值得写邮件问作者。
- [EdgeBench-Home](https://huggingface.co/datasets/chico-research/EdgeBench-Home)：zh 420 / en 420，带 device_specs 和期望调用，支持多设备；CC-BY-4.0，2026-09-24；冻结评测集，可做双语回放。
- [Charles95/smart_home_control](https://huggingface.co/datasets/Charles95/smart_home_control)：3,530 条，只有空气净化器一种设备；格式（intent + 归一化 slot）可参考。
- [youkwan/HA-Requests-Zh](https://huggingface.co/datasets/youkwan/HA-Requests-Zh)：29,652 行**繁体**，acon96 Home-Assistant-Requests 的用户句翻译，回复仍是英文、重复多；无许可声明。
- 小数据：tuya_ac 129 条（空调红外码）、LLM_SMARTHOME2 153 条（粤语，有"设备已是目标状态 → suggestion"这一类，值得参考）、SmartHomeCLLM（绿城，模型开源数据不开源）。
- 通用任务型 NLU（无家居域，当困难负例）：SMP2019 ECDT 2,579 条 29 域（"打开汽车之家"是 app/LAUNCH——"打开"但非家居的困难负例）、FewJoint 6,694 条 59 域、SMP2018；CrossWOZ / RiSAWOZ 无家居域。

## C. 英文，结构最贴题

- [HomeBench](https://github.com/BITHLP/HomeBench)（北理工，ACL 2025）：173,647 条，100 个家（每家 ≥ 47 设备、15 类、12 房间）；test 里 **unexist_device 4,194 / unexist_attribute 2,571 / multi2~10 混合 4,072**；输出 `bathroom.light.set_brightness(20)`，无效项 `error_input`；GPT-3.5 生成，无 LICENSE。**复用它的 100 份设备清单和无效指令生成规则**，用 LLM 造中文。
- acon96 Home-Assistant-Requests V2：240,044 行，en/de/fr/es/pl，MIT，生成器开源可自加语言。
- 评测类：SimuHome（ICLR 2026，含 feasible / infeasible）、SmartHome-Bench、SmartBench、MIST。

## D. 找不到 / 不开放

小米小爱、天猫精灵、DuerOS、小艺、涂鸦：只有意图 / 槽位配置文档，无语料。讯飞比赛只有场景文本分类，无指令 NLU。数据堂 / MagicData 的家居语音是商业付费。魔搭搜"智能家居"只有一条付费唤醒词。

## 怎么接进训练框架

| 来源 | 用法 | adapter |
|---|---|---|
| zhihao666/smartHome | 转记录：state = 指令，动态清单 = 它自带的设备列表，金标 = tool_call 里的设备 + 动作 → 再用 `exit_injection` 造"不存在" | `import: zhihao_smarthome` |
| SCENIC | 规范回复反解 device / room / action；配我们的户型清单采样；多设备句金标 = "多个设备或整屋" | `import: scenic` |
| HA zh-CN | hassil 模板展开 + 测试句；当回放和覆盖校验 | `import: ha_intents` |
| MASSIVE zh-CN | iot 子集 → 控制 / 查询；其余场景 → "无关"负例（先解决非命令 0% 的切片） | `import: massive_zh` |
| HomeBench | 只借清单和规则给 llm 生成器当 contexts | `contexts_file` |

## 本地副本（scratchpad，重启后会没）

SCENIC `examples/data/`、HomeBench `dataset/`、HA intents `sentences/tests/zh-CN`、zhihao666 train.jsonl、SMP2019 train.json。
正式使用时放到 `laya/train/data/<name>/`（gitignore），manifest 记 sha256。
