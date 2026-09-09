"""The resident synthesis engine, and this service's sole way of reaching it.

One process, started once, holding 700 MB of NPU graphs. Everything about that
process's protocol is here so that the WebSocket layer beside this file never
has to know the engine is a subprocess at all.

What the engine promises, and what this reads:

* stdin — one utterance per line.
* a descriptor of its own — raw PCM, s16le/24 kHz/mono, flushed per chunk. The
  engine claims it before loading librkllm, because librkllm prints its banner
  to stdout and 567 bytes of "I rkllm: ..." is not audio.
* stderr — `--- eidolon-tts ready ---` once, when loading and warm-up are done,
  then `key=value` lines bracketed by `--- eidolon-tts utterance begin/end ---`
  per utterance. The library's own chatter lands on the same stream and outside
  those brackets, which is what the brackets are for.

Where the utterance ends is taken from `pcm_stream_bytes` in that report, not
from a marker in the audio: the engine writes exactly that many bytes and this
reads until it has them. The two streams need no ordering guarantee between
them for that to be exact.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from eidolon_models_tts.config import Settings

logger = logging.getLogger(__name__)

READY_SENTINEL = "--- eidolon-tts ready ---"
UTTERANCE_BEGIN = "--- eidolon-tts utterance begin ---"
UTTERANCE_END = "--- eidolon-tts utterance end ---"

#: How much PCM to hand upward at a time. 4800 samples is 200 ms at 24 kHz —
#: small enough that the first frame leaves as soon as the engine has produced
#: anything, large enough that a four-second utterance is twenty messages and
#: not a thousand.
AUDIO_CHUNK_BYTES = 4800 * 2


class EngineUnavailable(RuntimeError):
    """The engine is not loaded, or died. Retryable, and says so."""


class SynthesisFailed(RuntimeError):
    """The engine refused this utterance. The next one may still work."""


@dataclass
class UtteranceReport:
    """The engine's own account of one utterance, as `key=value` lines."""

    values: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.values.get("status") == "PASS"

    @property
    def pcm_bytes(self) -> int:
        return int(self.values.get("pcm_stream_bytes", "0"))

    @property
    def error(self) -> str:
        return self.values.get("error", "the engine reported no reason")

    def number(self, key: str) -> float | None:
        try:
            return float(self.values[key])
        except (KeyError, ValueError):
            return None


class Engine:
    """A started engine. Not started by its constructor — see `start`."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._process: asyncio.subprocess.Process | None = None
        self._reports: asyncio.Queue[UtteranceReport] = asyncio.Queue()
        self._stderr_task: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        #: The engine says one thing at a time, so this serialises requests
        #: rather than letting two interleave on one stdin.
        self._turn = asyncio.Lock()
        self._init_warmup_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._process is not None and (
            self._process.returncode is None
        )

    @property
    def init_warmup_ms(self) -> float | None:
        return self._init_warmup_ms

    def command(self) -> list[str]:
        """The engine's argument vector. Nine positionals, in its own order."""

        settings = self._settings
        return [
            str(settings.engine),
            str(settings.rkllm_model),
            str(settings.speech_head),
            str(settings.speech_embedding),
            str(settings.flow_encoder),
            str(settings.flow_estimator),
            str(settings.hift_root),
            str(settings.runtime_root),
            str(settings.flow_fixture),
            str(settings.output_root),
            f"--voice-profile-root={settings.voice_profile_root}",
            f"--text-frontend-root={settings.text_frontend_root}",
            f"--head-core={settings.head_core}",
            f"--encoder-core={settings.encoder_core}",
            f"--flow-core={settings.flow_core}",
            f"--hift-core={settings.hift_core}",
            # Without this the engine uses its own default of 288 and the
            # ceiling on an utterance becomes the text length: 44 characters
            # truncate mid-sentence, 80 refuse to synthesize. See
            # config.DEFAULT_MAX_CONTEXT.
            f"--max-context={settings.max_context}",
            "--serve",
            "--pcm-stream=-",
            "--sample",
            "--precompute-full-prompt",
        ]

    async def start(self) -> None:
        missing = [str(path) for path in self._settings.required_paths() if not path.exists()]
        if missing:
            # Refused here rather than by the engine: its failure for a missing
            # asset is a non-zero exit two seconds in, with the reason on a
            # stderr nobody is reading yet.
            raise EngineUnavailable("synthesis assets are absent: " + ", ".join(missing))
        self._settings.output_root.mkdir(parents=True, exist_ok=True)

        self._process = await asyncio.create_subprocess_exec(
            *self.command(),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._stderr_task = asyncio.create_task(self._read_stderr())
        try:
            await asyncio.wait_for(
                self._ready.wait(), timeout=self._settings.engine_start_timeout_s
            )
        except TimeoutError as error:
            await self.stop()
            raise EngineUnavailable(
                f"the engine did not become ready within "
                f"{self._settings.engine_start_timeout_s:.0f}s"
            ) from error
        logger.info("tts engine ready init_warmup_ms=%s", self._init_warmup_ms)

    async def stop(self) -> None:
        process, self._process = self._process, None
        self._ready.clear()
        if process is not None and process.returncode is None:
            if process.stdin is not None and not process.stdin.is_closing():
                # Closing stdin ends the engine's loop, which lets it destroy
                # the RKLLM handle itself. A signal would leave the NPU
                # context to the kernel to clean up.
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), timeout=10)
            except TimeoutError:
                process.kill()
                await process.wait()
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            self._stderr_task = None

    async def synthesize(self, text: str) -> AsyncIterator[bytes | UtteranceReport]:
        """Yield audio chunks as they arrive, then the engine's own report.

        The report comes last and is the only thing that says the utterance
        ended, so a caller that stops early gets no report and knows it.
        """

        if not self.ready:
            raise EngineUnavailable("the engine is not loaded")
        async with self._turn:
            process = self._process
            if process is None or process.stdin is None or process.stdout is None:
                raise EngineUnavailable("the engine is not loaded")
            # Any report left from an abandoned request would be read as this
            # one's. Nothing should be here — the lock serialises requests and
            # each drains its own — so an entry means a previous caller left
            # mid-utterance and this is where it is noticed.
            while not self._reports.empty():
                stale = self._reports.get_nowait()
                logger.warning("discarding a report from an abandoned utterance: %s", stale.values)

            process.stdin.write((text + "\n").encode("utf-8"))
            await process.stdin.drain()

            report_task = asyncio.create_task(self._reports.get())
            delivered = 0
            try:
                while True:
                    read_task = asyncio.create_task(process.stdout.read(AUDIO_CHUNK_BYTES))
                    done, _ = await asyncio.wait(
                        {read_task, report_task},
                        return_when=asyncio.FIRST_COMPLETED,
                        timeout=self._settings.request_timeout_s,
                    )
                    if not done:
                        read_task.cancel()
                        raise EngineUnavailable("the engine stopped answering")
                    if read_task in done:
                        chunk = read_task.result()
                        if not chunk:
                            # stdout closed: the engine is gone, and whatever
                            # it was saying is not coming.
                            raise EngineUnavailable("the engine closed its audio stream")
                        delivered += len(chunk)
                        yield chunk
                        continue
                    read_task.cancel()
                    report = report_task.result()
                    if not report.ok:
                        raise SynthesisFailed(report.error)
                    # The report is written after the audio, but on a different
                    # pipe, so the tail may still be in flight. The byte count
                    # is exact, which is what makes this terminate.
                    while delivered < report.pcm_bytes:
                        chunk = await asyncio.wait_for(
                            process.stdout.read(
                                min(AUDIO_CHUNK_BYTES, report.pcm_bytes - delivered)
                            ),
                            timeout=self._settings.request_timeout_s,
                        )
                        if not chunk:
                            raise EngineUnavailable(
                                "the engine closed its audio stream mid-utterance"
                            )
                        delivered += len(chunk)
                        yield chunk
                    if delivered != report.pcm_bytes:
                        raise EngineUnavailable(
                            f"the engine reported {report.pcm_bytes} PCM bytes "
                            f"and delivered {delivered}"
                        )
                    yield report
                    return
            finally:
                report_task.cancel()

    async def _read_stderr(self) -> None:
        """Turn the engine's stderr into readiness and one report per utterance.

        Anything outside the brackets is the NPU runtime talking about itself.
        It goes to the journal at debug: on a board it is the only account of
        what the driver did, and losing it to keep the log tidy has cost this
        project a day before.
        """

        process = self._process
        if process is None or process.stderr is None:
            return
        collecting: UtteranceReport | None = None
        while True:
            raw = await process.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == READY_SENTINEL:
                self._ready.set()
                continue
            if line == UTTERANCE_BEGIN:
                collecting = UtteranceReport()
                continue
            if line == UTTERANCE_END:
                if collecting is not None:
                    await self._reports.put(collecting)
                collecting = None
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                if collecting is not None:
                    collecting.values[key] = value
                    continue
                if key == "service_init_warmup_ms":
                    try:
                        self._init_warmup_ms = float(value)
                    except ValueError:
                        pass
                    continue
            logger.debug("tts engine: %s", line)
        # stderr closed: the engine exited. Whoever is waiting on readiness or
        # on a report must not wait forever.
        self._ready.clear()
        await self._reports.put(
            UtteranceReport({"status": "FAIL", "error": "the engine exited"})
        )
