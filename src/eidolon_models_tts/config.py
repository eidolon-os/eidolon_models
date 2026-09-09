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
#: 288 is the engine's own default, restated here so it is a decision rather
#: than an accident — and it is a decision that costs something. The engine
#: derives its generation budget as
#: `min(text_tokens * 20, max_context - prefill_tokens)` and refuses outright
#: when that lands under `text_tokens * 2`, so at 288 the ceiling on an
#: utterance is its text length rather than the model's context: 44 characters
#: truncate mid-sentence at 6.04 s every time, 80 produce no audio at all.
#:
#: **Do not raise this on its own.** It was raised to 2048 (what the weights
#: hold — `qwen2_body_w8a8_c2_ctx2048`) and that was worse, not better. Serve
#: mode reuses one RKLLM handle across utterances without clearing its KV
#: cache, and 288 is small enough that the residue is squeezed out before it
#: matters. Give it 2048 of room and the residue accumulates instead. Measured
#: on the board, same sentence six times from a cold start:
#:
#:   288:  6/6 intact
#:   2048: utterances 1-3 intact, then "大妈妈妈之前也这样子，忽来过，今天
#:         天气不错，我们出去走走吧。" — the voice prompt's own text plus the
#:         previous utterances, and steady rtf 0.9 -> 1.5-1.6
#:
#: So the small context has been hiding a cross-utterance bug rather than
#: being the whole problem. HOST-RK3588.md §2.26 had already recorded the
#: behaviour — serve mode prefills with `keep_history = true`, so the KV
#: history accumulates — but at 288 it only showed up as harmless jitter in the
#: token count. The fix is one line in the engine,
#: `cosyvoice2_streaming_pipeline.cpp:2071`: the *first* prefill of each
#: utterance should pass `false` (the two later call sites pass `true`
#: correctly, being single-token continuations within one utterance). The two
#: changes go together the way §2.21's two-core model and NPU core assignment
#: did. Until that lands, a larger value here trades truncated long sentences
#: for corrupted ones, which is worse: a truncation is at least the beginning
#: of this sentence.
#:
#: Overridable because that is what made the bad value recoverable without
#: rolling a release back: one drop-in set it to 288 and the next restart was
#: correct.
DEFAULT_MAX_CONTEXT = 288


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
