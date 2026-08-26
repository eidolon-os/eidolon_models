"""Explicit connection and utterance capacity management."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass


class CapacityError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(eq=False)
class UtteranceLease:
    ready: asyncio.Future[None]
    queued: bool
    queue_position: int
    reserved_at: float
    activated_at: float | None = None
    active: bool = False
    released: bool = False

    @property
    def queue_wait_ms(self) -> float:
        if self.activated_at is None:
            return 0.0
        return round((self.activated_at - self.reserved_at) * 1000.0, 3)


class CapacityManager:
    """FIFO admission with shared realtime slots and a bounded wait queue."""

    def __init__(self, *, max_connections: int, realtime_slots: int, max_queued: int) -> None:
        self.max_connections = max_connections
        self.realtime_slots = realtime_slots
        self.max_queued = max_queued
        self._connections = 0
        self._active = 0
        self._waiters: deque[UtteranceLease] = deque()
        self._lock = asyncio.Lock()

    async def open_connection(self) -> bool:
        async with self._lock:
            if self._connections >= self.max_connections:
                return False
            self._connections += 1
            return True

    async def close_connection(self) -> None:
        async with self._lock:
            self._connections = max(0, self._connections - 1)

    async def reserve(self) -> UtteranceLease:
        loop = asyncio.get_running_loop()
        ready: asyncio.Future[None] = loop.create_future()
        reserved_at = loop.time()
        async with self._lock:
            if self._active < self.realtime_slots:
                lease = UtteranceLease(
                    ready=ready,
                    queued=False,
                    queue_position=0,
                    reserved_at=reserved_at,
                    activated_at=reserved_at,
                    active=True,
                )
                self._active += 1
                ready.set_result(None)
                return lease
            if len(self._waiters) >= self.max_queued:
                raise CapacityError(
                    "capacity_exceeded",
                    "all realtime slots and queued utterance positions are in use",
                )
            lease = UtteranceLease(
                ready=ready,
                queued=True,
                queue_position=len(self._waiters) + 1,
                reserved_at=reserved_at,
            )
            self._waiters.append(lease)
            return lease

    async def wait_ready(self, lease: UtteranceLease, timeout_seconds: float) -> None:
        try:
            await asyncio.wait_for(asyncio.shield(lease.ready), timeout=timeout_seconds)
        except TimeoutError as exc:
            await self.release(lease)
            raise CapacityError(
                "capacity_timeout",
                f"utterance waited more than {timeout_seconds:g} seconds for a realtime slot",
            ) from exc

    async def release(self, lease: UtteranceLease) -> None:
        async with self._lock:
            if lease.released:
                return
            lease.released = True
            if lease.active:
                lease.active = False
                self._active = max(0, self._active - 1)
                self._promote_next_locked()
                return
            try:
                self._waiters.remove(lease)
            except ValueError:
                pass
            if not lease.ready.done():
                lease.ready.cancel()

    def _promote_next_locked(self) -> None:
        while self._waiters and self._active < self.realtime_slots:
            lease = self._waiters.popleft()
            if lease.released or lease.ready.cancelled():
                continue
            lease.active = True
            lease.activated_at = asyncio.get_running_loop().time()
            self._active += 1
            if not lease.ready.done():
                lease.ready.set_result(None)
            return

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {
                "connections": self._connections,
                "active_utterances": self._active,
                "queued_utterances": len(self._waiters),
                "max_connections": self.max_connections,
                "realtime_slots": self.realtime_slots,
                "max_queued_utterances": self.max_queued,
                "max_active_utterances": self.realtime_slots + self.max_queued,
            }
