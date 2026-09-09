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
            self.text_frontend_root / "manifest.json",
            self.voice_profile_root / "manifest.json",
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
    )
