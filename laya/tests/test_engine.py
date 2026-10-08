"""Staged questions (ask_if): later questions run only when an earlier answer calls for them.
Real tokenizer, fake backend (no weights)."""

from pathlib import Path

import numpy as np
import pytest

from eidolon_models_laya.engine import DecisionEngine
from eidolon_models_laya.sequence import ModelConfig, Tokenizer

TOK = Path(__file__).resolve().parents[1] / "models" / "laya-multilingual" / "1c5edc17" / "torch" / "tokenizer"

QUESTIONS = {
    "intent": {"type": "choice", "instructions": "意图？", "criteria": ["控制", "查询", "无关"]},
    "device": {"type": "choice", "instructions": "哪台设备？", "criteria": ["客厅灯", "空调", "没有对应的设备"]},
    "action": {"type": "choice", "instructions": "什么动作？", "criteria": ["打开或启动", "关闭或停止"]},
}
SMART_HOME = {"device": {"intent": ["控制", "查询"]}, "action": {"intent": ["控制"]}}


class PickBackend:
    """Every question answers option ``pick``; records how many items each forward saw."""

    name = "fake"

    def __init__(self, pick: int):
        self.pick, self.calls = pick, []

    def forward(self, batch):
        n, k = batch["marker_mask"].shape
        self.calls.append(n)
        logits = np.full((n, k), -5.0, np.float32)
        logits[:, min(self.pick, k - 1)] = 5.0
        return logits, np.zeros((n, 2), np.float32)


def engine(pick: int) -> DecisionEngine:
    if not (TOK / "tokenizer.json").exists():
        pytest.skip("base tokenizer not fetched")
    cfg = ModelConfig.from_dict({"max_len": 256, "head_max_len": 128})
    return DecisionEngine(PickBackend(pick), Tokenizer(TOK), cfg)


def test_non_command_costs_one_question():
    e = engine(2)  # intent = 无关
    p = e.predict({"utterance": "邻居家的猫吵死了"}, QUESTIONS, ask_if=SMART_HOME)
    assert list(p.answers) == ["intent"] and p.answers["intent"]["choice"] == "无关"
    assert p.skipped == {"device": "intent=无关", "action": "intent=无关"}
    assert e.backend.calls == [1]
    assert p.as_response("m", "fake")["skipped"] == p.skipped


def test_backend_input_capacity_bounds_service_and_reports_truncation():
    e = engine(0)
    e.backend.max_len = 128
    bounded = DecisionEngine(
        e.backend, e.tokenizer, ModelConfig.from_dict({"max_len": 1024, "head_max_len": 128}),
    )
    assert bounded.max_len == 128
    result = bounded.predict({"utterance": "打开客厅灯" * 300}, QUESTIONS)
    assert result.truncated == list(QUESTIONS)


def test_command_asks_the_rest_in_one_second_pass():
    e = engine(0)  # intent = 控制
    p = e.predict({"utterance": "开灯"}, QUESTIONS, ask_if=SMART_HOME)
    assert list(p.answers) == ["intent", "device", "action"] and not p.skipped
    assert e.backend.calls == [1, 2]


def test_query_skips_only_the_action():
    e = engine(1)  # intent = 查询
    p = e.predict({"utterance": "空调开着吗"}, QUESTIONS, ask_if=SMART_HOME)
    assert list(p.answers) == ["intent", "device"] and p.skipped == {"action": "intent=查询"}


def test_without_ask_if_everything_is_one_pass():
    e = engine(2)
    p = e.predict({"utterance": "晚安"}, QUESTIONS)
    assert list(p.answers) == list(QUESTIONS) and e.backend.calls == [3]
    assert "skipped" not in p.as_response("m", "fake")


@pytest.mark.parametrize(
    "ask_if, match",
    [
        ({"device": {"nope": ["控制"]}}, "not another of the questions"),
        ({"device": {"intent": ["控制了"]}}, "options"),
        ({"device": {"action": ["打开或启动"]}, "action": {"device": ["空调"]}}, "cycle"),
        ({"ghost": {"intent": ["控制"]}}, "not one of the questions"),
    ],
)
def test_bad_ask_if_is_rejected(ask_if, match):
    with pytest.raises(ValueError, match=match):
        engine(0).predict({"utterance": "开灯"}, QUESTIONS, ask_if=ask_if)


class SlowSubmitBackend(PickBackend):
    """Parallel backend with per-item futures: the first item (intent) is fast, the rest slow."""

    def __init__(self, pick: int, slow: float = 0.4):
        super().__init__(pick)
        self.slow, self.submitted, self.threads = slow, [], []

    def submit(self, batch):
        import threading
        import time as _time
        from concurrent.futures import Future

        logits, act = self.forward(batch)
        n = logits.shape[0]
        self.submitted.append(n)
        futs = [Future() for _ in range(n)]
        k = batch["marker_mask"].sum(1)

        def run(i, delay):
            _time.sleep(delay)
            futs[i].set_result((logits[i, : k[i]], act[i]))

        for i in range(n):
            t = threading.Thread(target=run, args=(i, 0.02 if i == 0 else self.slow))
            t.start()
            self.threads.append(t)
        return futs

    def join(self):  # dropped items still finish; don't leave them running into interpreter teardown
        for t in self.threads:
            t.join()


def speculative_engine(pick: int) -> DecisionEngine:
    e = engine(pick)
    return DecisionEngine(SlowSubmitBackend(pick), e.tokenizer, e.cfg, speculative=True)


def test_speculative_returns_after_intent_for_a_non_command():
    import time as _time

    e = speculative_engine(2)
    t = _time.perf_counter()
    p = e.predict({"utterance": "邻居家的猫吵死了"}, QUESTIONS, ask_if=SMART_HOME)
    assert _time.perf_counter() - t < 0.3  # did not wait for the 0.4 s device / action items
    assert list(p.answers) == ["intent"] and p.skipped == {"device": "intent=无关", "action": "intent=无关"}
    assert e.backend.submitted == [3]  # all three were started at once
    e.backend.join()


def test_speculative_matches_staged_answers():
    for pick in (0, 1, 2):
        e = speculative_engine(pick)
        spec = e.predict({"utterance": "开灯"}, QUESTIONS, ask_if=SMART_HOME)
        e.backend.join()
        staged = engine(pick).predict({"utterance": "开灯"}, QUESTIONS, ask_if=SMART_HOME)
        assert spec.answers == staged.answers and spec.skipped == staged.skipped


def test_speculative_needs_a_backend_that_can_submit():
    e = engine(0)
    assert not DecisionEngine(e.backend, e.tokenizer, e.cfg, speculative=True).speculative


def test_question_state_override_isolated_in_one_batch():
    e = engine(0)
    q = QUESTIONS["intent"]
    state = {"utterance": "关灯"}
    history = {"utterance": "客厅的", "context": {"Agent": "关哪盏？"}}
    _, _, items, cut = e._prepare(state, {"plain": q, "context": {**q, "state": history}}, False)
    _, _, expected, _ = e._prepare(history, {"only": q}, False)
    assert items[1]["ids"] == expected[0]["ids"]
    assert items[0]["ids"] != items[1]["ids"] and not cut
    e.predict(state, {"plain": q, "context": {**q, "state": history}})
    assert e.backend.calls == [2]


def test_question_override_truncation_is_reported_per_question():
    e = engine(0)
    p = e.predict("短句", {"plain": QUESTIONS["intent"],
                  "long": {**QUESTIONS["intent"], "state": "历史对话" * 1000}})
    assert p.truncated == ["long"]
