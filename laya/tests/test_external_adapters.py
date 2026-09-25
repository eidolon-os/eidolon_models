"""External dataset adapters, on inline fake rows (no network, no downloaded files)."""

import json
import random
from pathlib import Path

from eidolon_laya_train import external as X
from eidolon_laya_train.records import target_vector
from eidolon_laya_train.scenario import Scenario

SMART_HOME = Path(__file__).resolve().parent.parent / "train" / "scenarios" / "smart-home"


def _zhihao_row(utt, devices, calls):
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": f"用户指令:{utt}; 设备列表如下: {json.dumps(devices, ensure_ascii=False)}",
            },
            {
                "role": "assistant",
                "content": f'<think>…</think><tool_call name="control_device">{json.dumps(calls, ensure_ascii=False)}</tool_call>',
            },
        ]
    }


def test_zhihao_parse_and_gold():
    devs = [
        {"deviceName": "筒灯", "deviceTypeName": "灯", "roomName": "客厅", "floorName": "1楼"},
        {"deviceName": "筒灯", "deviceTypeName": "灯", "roomName": "主卧", "floorName": "2楼"},
        {"deviceName": "风管机", "deviceTypeName": "空调", "roomName": "客厅", "floorName": "1楼"},
    ]
    p = X.parse_zhihao_row(
        _zhihao_row(
            "把主卧筒灯关了",
            devs,
            [{"deviceName": "筒灯", "action": "TurnOff", "location": "2楼|主卧"}],
        )
    )
    crit, keyed = X.zhihao_device_map(p["devices"])
    assert set(crit) == {"客厅筒灯", "主卧筒灯", "风管机"}
    assert X.zhihao_gold_device(p, keyed) == ("主卧筒灯", False)
    p2 = dict(p, utterance="筒灯关了")  # room not named: both lamps acceptable
    assert X.zhihao_gold_device(p2, keyed) == (["客厅筒灯", "主卧筒灯"], True)
    p3 = dict(p, calls=[{"deviceName": "筒灯"}, {"deviceName": "风管机"}])
    assert X.zhihao_gold_device(p3, keyed) == (X.EXIT_MULTI, False)
    assert X.parse_zhihao_row({"messages": [{"role": "user", "content": "x"}]}) is None


def test_scenic_segments_parse():
    assert X.parse_scenic_response("好的，已关闭书房加湿器。") == [
        {"room": "书房", "device": "加湿器", "action": X.OFF, "kind": "simple"}
    ]
    segs = X.parse_scenic_response("好的，已打开客厅空调；已将客厅空调温度调到26度。")
    assert [s["action"] for s in segs] == [X.ON, X.SET] and segs[1]["device"] == "空调"
    assert X.parse_scenic_response("好的，已为客厅灯设置晚上10点关闭。")[0]["kind"] == "timed"
    assert X.parse_scenic_response("好的，已让扫地机器人返回充电。")[0]["action"] == X.OFF
    assert X.parse_scenic_response("今天天气不错") is None


def test_scenic_home_and_gold_device():
    rows = [[{"room": "客厅", "device": "空调", "action": X.ON, "kind": "simple"}]] * 3 + [
        [{"room": "", "device": "扫地机器人", "action": X.ON, "kind": "robot"}]
    ] * 3
    home = X.scenic_home(rows, min_count=3)
    assert home == {"客厅空调": "客厅·空调", "扫地机器人": "全屋·扫地机器人"}
    assert X.scenic_gold_device([{"room": "", "device": "空调"}], home) == ["客厅空调"]
    assert X.scenic_gold_device(rows[0] + rows[3], home) == X.EXIT_MULTI


def test_massive_labels_are_legal_options_and_speaker_scenarios_excluded(tmp_path):
    import gzip

    scn = Scenario.load(SMART_HOME)
    rows = [
        {
            "id": "1",
            "locale": "zh-CN",
            "utt": "把灯关掉",
            "scenario": "iot",
            "intent": "iot_hue_lightoff",
        },
        {
            "id": "2",
            "locale": "zh-CN",
            "utt": "明天天气怎么样",
            "scenario": "weather",
            "intent": "weather_query",
        },
        {
            "id": "3",
            "locale": "zh-CN",
            "utt": "音量调大一点",
            "scenario": "audio",
            "intent": "audio_volume_up",
        },
        {
            "id": "4",
            "locale": "zh-CN",
            "utt": "打开插座",
            "scenario": "iot",
            "intent": "iot_wemo_on",
        },
        {
            "id": "5",
            "locale": "en-US",
            "utt": "turn off the light",
            "scenario": "iot",
            "intent": "iot_hue_lightoff",
        },
    ]
    (tmp_path / "train").mkdir()
    with gzip.open(tmp_path / "train" / "zh-CN.json.gz", "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    recs = list(
        X.import_massive_zh(scn, {"path": str(tmp_path), "splits": ["train"]}, rng=random.Random(1))
    )
    by = {r.state["utterance"]: r for r in recs}
    assert set(by) == {"把灯关掉", "明天天气怎么样", "打开插座"}  # audio excluded, en-US dropped
    assert by["明天天气怎么样"].labels == {"intent": {"gold": "无关"}}
    assert by["打开插座"].labels["device"]["gold"] == X.EXIT_NONE
    for r in recs:
        for qid, lab in r.labels.items():
            target_vector(r.questions[qid], lab)
