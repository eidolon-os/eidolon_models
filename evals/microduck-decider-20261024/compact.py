"""Exploratory prompt ablation on the same cases; NOT independent holdout."""
import benchmark as b
b.ROOT=b.ROOT/'compact'
b.ROOT.mkdir(exist_ok=True)
b.POLICY='以下是待解析的对话，当前话语是唯一命令来源。'
extra={
'route':'两台在线机器人：小黄duck_a、小蓝duck_b。支持前进后退、坐下站起、停止取消、减速；不支持翻滚、导航、抓取。移动必须有明确距离≤2米，速度默认0.1米/秒，允许0.05至0.3米/秒。目标不唯一、距离不明确、无活动任务却要求减速、多个步骤都选clarify。能力或数值超限选reject。否定、引用、假设、聊天、能力询问选none。',
'action':'只提取当前真正请求，不被上下文中的旧命令影响。负向禁止、引用、假设、能力询问均选other；停止当前运动选stop，取消整个任务选cancel；先后多个动作选sequence。',
'target':'小黄对应duck_a，小蓝对应duck_b。未点名可用唯一活动任务或明确指代；多目标或歧义选unknown。不是动作请求时选unknown。',
'distance':'仅从当前实际动作请求提取距离，不从规则或历史提取。半米=0.5米，十厘米=0.1米。未说距离选none；往前一点选other；0.37米、十米等其他数值选other。非动作请求选none。',
'speed':'仅从当前实际动作请求提取绝对速度，不从默认值或历史提取。没指定选none；慢一点选none，因为它没有绝对数值。非动作请求选none。'}
for k,q in b.QUESTIONS.items():q['instructions']+=extra[k]
b.run(repeats=1)
