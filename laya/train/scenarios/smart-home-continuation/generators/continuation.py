"""Seed cases with explicit plan semantics; evaluation uses separate phrasing.

These synthetic cases establish an experiment, not a production acceptance set.
All counterparts of the same utterance family share an assembly group.
"""
from eidolon_laya_train.records import Record, record_id

SELECT = ['{room}那个', '选{room}', '{room}的', '就是{room}那盏', '我要{room}的灯', '{room}那台']
CANCEL = ['算了', '取消', '别执行了', '不用了', '先别动', '保持原样', '我反悔了', '暂时不要操作']
ESCALATE = ['都行', '随便', '哪个都可以', '你决定', '它在哪儿', '明天天气怎么样', '打开它', '别关主卧灯，打开客厅灯']


def generate(scenario, config, rng):
    homes = [('客厅', '主卧'), ('书房', '次卧'), ('餐厅', '走廊'), ('卧室', '阳台')]
    for rooms in homes:
        for action in ('打开', '关闭'):
            decisions = {f'target_{i}': f'{action}{room}灯；只选择设备，保持待确认动作{action}' for i, room in enumerate(rooms)}
            context = {'status': 'awaiting_target', 'pending_action': action,
                       'targets': [{'ref': f'target_{i}', 'name': f'{room}灯'} for i, room in enumerate(rooms)],
                       'submitted': False}
            cases = []
            for i, room in enumerate(rooms):
                cases += [(t.format(room=room), f'target_{i}', f'select-{n}') for n, t in enumerate(SELECT)]
            cases += [(t, 'cancel', f'cancel-{n}') for n, t in enumerate(CANCEL)]
            cases += [(t, 'escalate', f'escalate-{n}') for n, t in enumerate(ESCALATE)]
            reverse = '关闭' if action == '打开' else '打开'
            cases += [(f'{reverse}{rooms[0]}灯', 'escalate', 'change-action')]
            for utterance, gold, family in cases:
                state = {'utterance': utterance, 'context': context}
                yield Record(id=record_id(scenario.name, state), scenario=scenario.name,
                             source='authored-template-seed', state=state,
                             questions=scenario.build_questions({'decisions': decisions}), labels={'next': {'gold': gold}},
                             tags=['pending-target', gold if gold in ('cancel','escalate') else 'select'],
                             meta={'derived_from': f'continuation/{family}'})
    # A focus supports explicit new actions, unlike an unfinished target choice.
    for room in ('客厅', '主卧', '书房', '次卧'):
        decisions = {'open': f'打开{room}灯', 'close': f'关闭{room}灯'}
        for has_focus in (True, False):
            context = {'status': 'confirmed_focus' if has_focus else 'no_focus',
                       'focus': {'ref': 'focus', 'name': f'{room}灯', 'source': 'successful_execution'} if has_focus else None}
            active = decisions if has_focus else {}
            for n, (utterance, gold) in enumerate([('打开它','open'), ('把它打开','open'), ('关闭它','close'), ('把它关上','close'), ('取消','cancel'), ('不要关它','cancel')]):
                if not has_focus and gold != 'cancel':gold='escalate'
                state={'utterance':utterance,'context':context}
                yield Record(id=record_id(scenario.name,state),scenario=scenario.name,source='authored-template-seed',state=state,
                             questions=scenario.build_questions({'decisions':active}),labels={'next':{'gold':gold}},
                             tags=['focus' if has_focus else 'no-focus',gold],meta={'derived_from':f'continuation/focus-{n}'})
