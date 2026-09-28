# 写手指南：episode 格式与场景家族

写之前完整读 `LABELING.md`。每个 episode 是一段完整的对话过程：场景 + 一串事件，**每个事件之后都要写出"下一步应该怎样"的金标**。

## 1. 格式（一行一个 JSON）

```json
{"episode": "<前缀>-0001", "family": "T03", "mode": "team",
 "scene_goal": "",
 "candidates": [
   {"id": "zhou", "name": "阿舟", "role": "爱讲冷笑话的厨师，熟悉家常菜"},
   {"id": "lan", "name": "小岚", "role": "细心的图书管理员，喜欢推荐书"}
 ],
 "events": [
   {"user": "阿舟，你分两次给我讲讲红烧肉，先讲备料。",
    "gold": {"action": "respond", "speaker": ["zhou"]}},
   {"say": "zhou", "text": "备料很简单：五花肉一斤，冰糖一小把，葱姜……",
    "gold": {"action": "respond", "speaker": ["zhou"]}},
   {"say": "zhou", "text": "做法是先焯水，再小火炒糖色……",
    "gold": {"action": "finish"}},
   {"user": "他说的冰糖可以换成白糖吗？", "gold": {"action": "respond", "speaker": ["zhou"]}}
 ]}
```

- `mode`：`team`（2–5 位候选）或 `companion`（恰好 1 位候选）。
- `candidates[].id`：episode 内唯一的短 ID（拼音 / 英文），**名字可以重复**（同名角色用不同 id）。`role` 一两句话。
- `events`：按时间顺序。`{"user": 文本}` 是用户的一次输入（开始新的一轮）；`{"say": id, "text": 文本}` 是某位角色**已经说完**的一次发言。
- `gold`：这个事件之后的决策。`action` ∈ respond / clarify / wait / finish；respond / clarify 必须有 `speaker`（id 列表，所有合适的都写上）；
  clarify 还要 `clarify_about` ∈ 指代不明 / 要求不明 / 对象不在场。wait / finish 不写 speaker。
- **角色发言要和上一步的金标一致**：金标是 respond 某人，下一个事件就应该是这个人说话（或者用户打断）；金标是 wait / finish，下一个事件应该是用户的新输入。
- 角色台词要像真的角色在说话（按角色设定、自然口语），不要说"我讲完了""轮到你了"这类替模型做判断的元信息。
- 一个 episode 3–8 个事件；可以有两到三轮（用户多次输入）。

## 2. 场景家族（`family`）

团队（`team`）：

| 代号 | 家族 | 要覆盖的决策 |
|---|---|---|
| T01 | 点名一人说一次 | respond 点名者 → finish |
| T02 | 点名一人说多次 / 分几段 | 同一人连续 respond → 讲完才 finish |
| T03 | 点名两人（依次或"你们俩"） | respond A → respond B → finish |
| T04 | 开放问题 | 多个合适人选；根据进展再 respond 或 finish |
| T05 | 讨论 / 交棒 | B 回应 A 的实际观点；讨论到位后 finish |
| T06 | 按角色专长选人 | 问题落在某人专长上 → 只有他合适 |
| T07 | 追问某人刚说的 | respond 刚说的那位 |
| T08 | 指代不明 → 澄清 → 恢复 | clarify → wait（澄清问题说完）→ 用户说明 → respond 对应者 |
| T09 | 点名不在场的角色 | clarify（对象不在场）→ 用户改口 → respond |
| T10 | 同名角色 | 有区分线索 → respond 对的那位；无线索 → clarify（指代不明） |
| T11 | 要求安静 / 暂停与恢复 | wait → 用户恢复 → respond |
| T12 | 明确结束与新请求 | finish → 新请求 respond |
| T13 | 中途改要求 / 打断 | 用户新输入改变人选或内容 |
| T14 | "每人说一句" | 还没说过的都合适；都说过 → finish |
| T15 | 只提及不点名 | 名字出现但不是在叫他 → 按实际请求选人 |
| T16 | 要求内容不明 | clarify（要求不明）→ 用户说明 → respond |

陪伴（`companion`，1 位候选）：

| 代号 | 家族 | 要覆盖的决策 |
|---|---|---|
| C01 | 闲聊与继续 | respond → finish → 用户继续 → respond |
| C02 | 安静陪伴（可简短回应） | respond 一句 → finish |
| C03 | 明确别说话 / 暂停与恢复 | wait → 恢复 → respond |
| C04 | 明确结束与新请求 | finish → respond |
| C05 | 需求不明 → 澄清 → 恢复 | clarify → wait → respond |
| C06 | 多段请求（讲三个故事 / 列几条） | 未完成 → respond；完成 → finish |
| C07 | 倾诉 / 情绪 | 需要回应 → respond；用户说"不用安慰我，让我静静" → wait |
| C08 | 收尾话 / 纯确认 | "好的""嗯嗯" → finish；"谢谢，那……"（还有问题）→ respond |

**每个家族都要有对照**：同样的开头，决策不同（例如 T01 和 T02 的请求只差"说一次 / 分两次"；C02 和 C03 只差"可以说句话 / 别说话"）。
