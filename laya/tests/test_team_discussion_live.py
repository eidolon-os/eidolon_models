"""Opt-in d3 semantic evaluation using synthetic public context only.

Set EIDOLON_D3_EVAL_URL to the SystemOne endpoint to run.
"""
import os
import pytest
from eidolon_sdk.biz.participation import DecisionRequest
from eidolon_models_laya.remote_participation import RemoteParticipationAdapter

URL = os.environ.get("EIDOLON_D3_EVAL_URL")
pytestmark = pytest.mark.skipif(not URL, reason="explicit d3 evaluation endpoint required")
CASES = [
    ("open_handoff", "未来的城市生活可以有哪些有趣的变化？", [
        ("a", "我想把闲置停车场改成社区花园，大家下楼就能种菜和聊天。")], "respond", "b"),
    ("return_to_challenged_speaker", "未来的城市生活可以有哪些有趣的变化？", [
        ("a", "我想把闲置停车场改成社区花园，大家下楼就能种菜和聊天。"),
        ("b", "花园很有吸引力，但行动不便的人还需要就近停车。Aria，你会怎样保留他们的通行空间？")], "respond", "a"),
    ("discussion_complete", "把停车场改成花园，有什么好处和风险？", [
        ("a", "好处是增加绿地和邻里交流，可以先在闲置区域试点。"),
        ("b", "风险是挤占停车和无障碍通道，需要保留这些空间，维护费用也要明确。"),
        ("a", "那就只改闲置边角，保留无障碍通道，先由愿意参与的居民做小规模维护，评估后再扩大。")], "finish", None),
    ("simple_answer_complete", "一小时有多少分钟？", [("a", "一小时有60分钟。")], "finish", None),
    ("single_addressee", "Aria，你更喜欢花园还是停车场？", [("a", "我更喜欢花园，大家可以在绿地里一起活动。")], "finish", None),
    ("wait_for_user", "先别说话，让我想想。", [], "wait", None),
]

@pytest.mark.parametrize("case,question,history,action,speaker", CASES, ids=[c[0] for c in CASES])
def test_natural_discussion_decisions(case, question, history, action, speaker):
    user = dict(message_id="u", author_kind="user", author_id="user", text=question)
    messages = [user] + [dict(message_id=f"m{i}", author_kind="companion", author_id=who, text=text)
                         for i, (who, text) in enumerate(history)]
    request = DecisionRequest.model_validate(dict(
        decision_id=case, context_ref="synthetic-discussion-eval", context_version=len(messages),
        membership_revision=1, cancellation_epoch=0, timeout_ms=5000,
        user_request=user, trigger=messages[-1], context=dict(recent_messages=messages),
        candidates=[dict(companion_id="a", display_name="Aria", description="好奇，关注创意、社区和人与人之间的连接。"),
                    dict(companion_id="b", display_name="小禾", description="务实，关心可行性、不同使用者的需要和容易被忽略的风险。")],
        constraints=dict(remaining_replies=8-len(history), allowed_actions=["respond", "clarify", "wait", "finish"]),
    ))
    adapter = RemoteParticipationAdapter(URL)
    try:
        result, explanation = adapter.decide_explained(request)
    finally:
        adapter.close()
    assert result.proposal is not None, explanation
    assert result.proposal.action == action, (result, explanation)
    if speaker:
        assert result.proposal.participants == (speaker,), (result, explanation)
