from __future__ import annotations

import pytest

from eidolon_models_asr.capacity import CapacityError, CapacityManager


async def test_fifo_queue_promotes_waiters_and_reports_runtime_state() -> None:
    capacity = CapacityManager(max_connections=4, realtime_slots=2, max_queued=2)
    assert await capacity.open_connection() is True
    first = await capacity.reserve()
    second = await capacity.reserve()
    third = await capacity.reserve()
    fourth = await capacity.reserve()

    assert first.active is True
    assert first.queue_wait_ms == 0
    assert second.active is True
    assert (third.queued, third.queue_position) == (True, 1)
    assert (fourth.queued, fourth.queue_position) == (True, 2)
    assert await capacity.snapshot() == {
        "connections": 1,
        "active_utterances": 2,
        "queued_utterances": 2,
        "max_connections": 4,
        "realtime_slots": 2,
        "max_queued_utterances": 2,
        "max_active_utterances": 4,
    }

    await capacity.release(first)
    await capacity.wait_ready(third, 0.1)
    assert third.active is True
    assert third.queue_wait_ms >= 0
    assert fourth.active is False

    await capacity.release(second)
    await capacity.wait_ready(fourth, 0.1)
    assert fourth.active is True

    await capacity.release(third)
    await capacity.release(fourth)
    await capacity.close_connection()


async def test_queue_full_and_connection_limit_are_explicit() -> None:
    capacity = CapacityManager(max_connections=1, realtime_slots=1, max_queued=1)
    assert await capacity.open_connection() is True
    assert await capacity.open_connection() is False
    active = await capacity.reserve()
    queued = await capacity.reserve()
    with pytest.raises(CapacityError, match="queued utterance") as rejected:
        await capacity.reserve()
    assert rejected.value.code == "capacity_exceeded"
    await capacity.release(active)
    await capacity.release(queued)
    await capacity.close_connection()


async def test_queue_wait_timeout_removes_the_waiter() -> None:
    capacity = CapacityManager(max_connections=2, realtime_slots=1, max_queued=1)
    active = await capacity.reserve()
    queued = await capacity.reserve()

    with pytest.raises(CapacityError, match="waited more") as timed_out:
        await capacity.wait_ready(queued, 0.01)
    assert timed_out.value.code == "capacity_timeout"
    assert (await capacity.snapshot())["queued_utterances"] == 0

    await capacity.release(active)
