"""Where this Host's synthesis assets and engine are, and how it is tuned.

Every path is told to this service rather than derived from where the package
happens to sit. The ASR service learned that the hard way: `PROJECT_ROOT` from
``__file__`` lands inside ``.venv/lib/python3.13`` once the component is
installed, and the service restarted 114 times looking for its models there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The directory holding every asset the engine loads. Set by the launcher,
#: which knows where the release put them.
MODEL_ROOT_ENV = "EIDOLON_TTS_MODEL_ROOT"
#: The built engine. Built on the Host from the source this component ships —
#: it links the board's NPU runtime, so it is not a release artifact.
ENGINE_ENV = "EIDOLON_TTS_ENGINE"

DEFAULT_VOICE = "testwav_prompt3s"

#: NPU core masks. The head on core 0 and everything downstream on core 2 is
#: the arrangement HOST-RK3588.md §2.22-2.23 measured: the LLM keeps the A55
#: cluster, TTS keeps the NPU, and neither starves the other in one turn.
DEFAULT_HEAD_CORE = "0"
DEFAULT_DOWNSTREAM_CORE = "2"

#: How much RKLLM context the engine may use, in tokens.
#:
#: 2048 is what the weights themselves hold (`qwen2_body_w8a8_c2_ctx2048`), and
#: the engine's own default of 288 is not a working value: it derives the
#: generation budget as `min(text_tokens * 20, max_context - prefill_tokens)`
#: and refuses outright when that lands under `text_tokens * 2`, so at 288 the
#: ceiling on an utterance is its *text length* rather than the model's
#: context. Measured on the board, 3 s voice, four runs per length:
#:
#:            288        2048      audio     rtf(median)
#:   14 ch    4/4        4/4        4.08 s   0.845
#:   20 ch    4/4        4/4        5.00 s   0.785
#:   44 ch    0/4        4/4        9.80 s   0.828   <- truncated at 6.04 s
#:   80 ch    0/4        4/4       16.88 s   0.860   <- no audio at all
#:  126 ch    0/4        ok        26.20 s   0.896   <- no audio at all
#:
#: **This value and the engine's per-utterance KV clear are one change, not
#: two.** Raising it alone was tried and was worse than leaving it low: serve
#: mode reuses a single RKLLM handle and prefills with `keep_history = true`
#: (HOST-RK3588.md §2.26 recorded that), so the KV history accumulates across
#: utterances. 288 is small enough that the residue is squeezed out before it
#: matters — it only showed up as harmless jitter in the token count. Give it
#: 2048 of room and the residue accumulates instead: utterances 1-3 came out
#: intact and then the fourth began reciting the voice prompt's own text,
#: with steady rtf going 0.9 -> 1.5-1.6.
#:
#: So the engine now clears the cache at the top of every utterance
#: (`rkllm_clear_kv_cache`, two nullptrs = clear all) and proves it with
#: `rkllm_get_kv_cache_size`. With that in place the same sentence eight times
#: from a cold start is 8/8 intact at both values, and `audio_seconds` is
#: identical across all eight — which also made serve mode reproducible, ending
#: the token-count jitter §2.26 had accepted as a cost.
#:
#: `tests/test_tts_engine_command.py` holds the two together: a value above the
#: engine's default requires the clear to still be in the engine's source. Do
#: not raise one without the other; §2.21's two-core model and NPU core
#: assignment had the same shape.
#:
#: Overridable, which is what made the bad value recoverable without rolling a
#: release back: one drop-in and the next restart was correct.
DEFAULT_MAX_CONTEXT = 2048

#: The longest text this Host synthesizes without the audio breaking up.
#:
#: Not the same question as `protocol.MAX_TEXT_CHARACTERS` (400), and the
#: difference matters: that one is "will the service refuse the request", this
#: one is "will the listener hear a gap". A request of 200 characters is
#: accepted and answered, and the audio drops out in the middle of it.
#:
#: Measured on the board (HOST-RK3588.md §2.28), by the buffer floor
#: `minimum_buffer_after_ms` — a chunk arriving late is harmless while the
#: buffer stays positive, so the floor is the only quantity that says whether
#: anything was audible:
#:
#:    14 ch   4.08 s   +763 ms   rtf 0.97
#:    20 ch   5.00 s   +737 ms   rtf 0.93
#:    44 ch   9.80 s   +447 ms   rtf 0.98
#:    60 ch  13.20 s   +504 ms   rtf 1.01   <- this value, 20 rounds back to back
#:    80 ch  16.88 s   +209 ms   rtf 1.00-1.02
#:   126 ch  26.20 s   -246..-372 ms        <- gaps, every run
#:
#: The cause is rtf crossing 1: past that the deficit accumulates at about
#: 43 ms per second of audio, and 26 seconds is where it eats the whole buffer.
#: 60 was chosen over 80 because 80 leaves only 209 ms — and because 60 is
#: already what Channel's aggregator sends (`hard_max_chars`), so this states
#: the property that arrangement was relying on rather than inventing a new one.
#:
#: A caller that exceeds this should split on sentence boundaries. Raising this
#: number requires new measurements of the buffer floor, not an argument.
SAFE_TEXT_CHARACTERS = 60


@dataclass(frozen=True)
class Settings:
    model_root: Path
    engine: Path
    voice: str = DEFAULT_VOICE
    host: str = "127.0.0.1"
    port: int = 8770
    head_core: str = DEFAULT_HEAD_CORE
    encoder_core: str = DEFAULT_DOWNSTREAM_CORE
    flow_core: str = DEFAULT_DOWNSTREAM_CORE
    hift_core: str = DEFAULT_DOWNSTREAM_CORE
    max_context: int = DEFAULT_MAX_CONTEXT
    #: How long a start may take before the service calls the engine dead.
    #: The engine loads 700 MB and warms four NPU graphs; measured at 2.9 s on
    #: an idle board, and this leaves room for a board that is not idle.
    engine_start_timeout_s: float = 120.0
    #: How long one utterance may take. An RTF of about 0.85 means audio is
    #: produced slightly faster than it plays, so the cap is generous rather
    #: than tight: it exists to notice an engine that has stopped, not to
    #: enforce a deadline the protocol already reports per utterance.
    request_timeout_s: float = 180.0

    @property
    def rkllm_model(self) -> Path:
        return self.model_root / "qwen2_body_w8a8_c2_ctx2048.rkllm"

    @property
    def speech_head(self) -> Path:
        return self.model_root / "speech_head_fp16.rknn"

    @property
    def speech_embedding(self) -> Path:
        return self.model_root / "speech_embedding.f32.bin"

    @property
    def flow_encoder(self) -> Path:
        return self.model_root / "flow_encoder_chunk25_cache256_fp16.rknn"

    @property
    def flow_estimator(self) -> Path:
        return self.model_root / "flow_estimator_cache200_fp16.rknn"

    @property
    def hift_root(self) -> Path:
        # The engine composes `hift_f0_fp16_seq<N>.rknn` and
        # `hift_decoder_fp16_mel<N>.rknn` under this root itself.
        return self.model_root

    @property
    def runtime_root(self) -> Path:
        return self.model_root / "fixture"

    @property
    def flow_fixture(self) -> Path:
        return self.model_root / "flow"

    @property
    def text_frontend_root(self) -> Path:
        return self.model_root / "text_frontend"

    @property
    def voice_profile_root(self) -> Path:
        return self.model_root / "voices" / self.voice

    #: Written to only in benchmark mode. Serve mode writes no files per
    #: utterance, but the engine still takes the positional and creates it.
    @property
    def output_root(self) -> Path:
        return Path(os.environ.get("EIDOLON_TTS_OUTPUT_ROOT", "/run/eidolon/tts"))

    def required_paths(self) -> tuple[Path, ...]:
        """Everything that must exist before the engine is worth starting.

        Checked here rather than left to the engine because the engine's
        failure for a missing asset is a non-zero exit two seconds in, with the
        reason on a stderr nobody is reading yet.
        """

        shapes = (24, 50, 54, 58)
        return (
            self.engine,
            self.rkllm_model,
            self.speech_head,
            self.speech_embedding,
            self.flow_encoder,
            self.flow_estimator,
            *(self.hift_root / f"hift_f0_fp16_seq{n}.rknn" for n in shapes),
            *(self.hift_root / f"hift_decoder_fp16_mel{n}.rknn" for n in shapes),
            self.runtime_root / "flow_prompt_token_50.i32.bin",
            self.runtime_root / "prompt_feat_100.f32.bin",
            self.flow_fixture / "spks.f32.bin",
            # What the text frontend's constructor opens, and only that. It
            # used to ask for a `manifest.json` in each of these two roots —
            # files the engine never reads. Requiring them made two conversion
            # -run artifacts into a startup precondition, so the package had to
            # carry them, and they were the one part of it with no upstream to
            # point at (they also embedded the producing machine's absolute
            # paths). The gap was in this list, not in the provenance.
            self.text_frontend_root / "tokenizer.bin",
            self.text_frontend_root / "text_embedding.f32.bin",
            self.text_frontend_root / "control_embedding.f32.bin",
            self.voice_profile_root / "prompt_text_token.i32.bin",
            self.voice_profile_root / "llm_prompt_speech_token.i32.bin",
        )


def load_settings() -> Settings:
    model_root = os.environ.get(MODEL_ROOT_ENV, "").strip()
    engine = os.environ.get(ENGINE_ENV, "").strip()
    if not model_root or not engine:
        raise RuntimeError(
            f"{MODEL_ROOT_ENV} and {ENGINE_ENV} are required; the launcher sets both"
        )
    return Settings(
        model_root=Path(model_root),
        engine=Path(engine),
        voice=os.environ.get("EIDOLON_TTS_VOICE", DEFAULT_VOICE).strip() or DEFAULT_VOICE,
        host=os.environ.get("EIDOLON_TTS_HOST", "127.0.0.1"),
        port=int(os.environ.get("EIDOLON_TTS_PORT", "8770")),
        head_core=os.environ.get("EIDOLON_TTS_HEAD_CORE", DEFAULT_HEAD_CORE),
        encoder_core=os.environ.get("EIDOLON_TTS_ENCODER_CORE", DEFAULT_DOWNSTREAM_CORE),
        flow_core=os.environ.get("EIDOLON_TTS_FLOW_CORE", DEFAULT_DOWNSTREAM_CORE),
        hift_core=os.environ.get("EIDOLON_TTS_HIFT_CORE", DEFAULT_DOWNSTREAM_CORE),
        max_context=int(os.environ.get("EIDOLON_TTS_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT))),
    )
