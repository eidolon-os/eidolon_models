from __future__ import annotations

import os

import pytest

from eidolon_models_asr.config import (
    Settings,
    apply_cpu_affinity,
    detect_host_kind,
    parse_cpu_list,
    resolve_backend,
)


def test_auto_backend_is_portable_cpu_baseline() -> None:
    assert resolve_backend("auto") == "onnx-cpu"
    assert Settings().resolved_backend == "onnx-cpu"
    assert Settings().offline_enabled is True
    assert Settings().punctuation_enabled is True
    assert Settings().max_connections == 64
    assert Settings().realtime_slots == 2
    assert Settings().max_queued_utterances == 6
    assert Settings().max_utterance_seconds == 60
    assert Settings().max_queue_wait_seconds == 10


def test_punctuation_can_be_disabled_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_PUNCTUATION_ENABLED", "off")
    assert Settings.from_env().punctuation_enabled is False


def test_offline_can_be_disabled_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_OFFLINE_ENABLED", "off")
    assert Settings.from_env().offline_enabled is False


def test_invalid_punctuation_environment_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_PUNCTUATION_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="EIDOLON_ASR_PUNCTUATION_ENABLED"):
        Settings.from_env()


def test_invalid_offline_environment_value_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_OFFLINE_ENABLED", "sometimes")
    with pytest.raises(ValueError, match="EIDOLON_ASR_OFFLINE_ENABLED"):
        Settings.from_env()


def test_capacity_can_be_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_MAX_CONNECTIONS", "12")
    monkeypatch.setenv("EIDOLON_ASR_REALTIME_SLOTS", "3")
    monkeypatch.setenv("EIDOLON_ASR_MAX_QUEUED_UTTERANCES", "4")
    monkeypatch.setenv("EIDOLON_ASR_MAX_UTTERANCE_SECONDS", "45")
    monkeypatch.setenv("EIDOLON_ASR_MAX_QUEUE_WAIT_SECONDS", "7.5")
    settings = Settings.from_env()
    assert settings.max_connections == 12
    assert settings.realtime_slots == 3
    assert settings.max_queued_utterances == 4
    assert settings.max_utterance_seconds == 45
    assert settings.max_queue_wait_seconds == 7.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_connections": 0},
        {"realtime_slots": 0},
        {"max_queued_utterances": -1},
        {"max_utterance_seconds": 0},
        {"max_queue_wait_seconds": 0},
        {"max_connections": 2, "realtime_slots": 2, "max_queued_utterances": 1},
    ],
)
def test_invalid_capacity_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        Settings(**kwargs)


def test_rknn_cannot_be_selected_without_artifact() -> None:
    with pytest.raises(ValueError, match="not available"):
        resolve_backend("rknn")


def test_host_kind_is_diagnostic_string() -> None:
    assert detect_host_kind()


def test_default_threads_follow_process_affinity(monkeypatch) -> None:
    """A service pinned to 2 cores must not size its pool to every core.

    ``os.cpu_count()`` ignores the affinity mask, so a ``taskset``-pinned
    process used to oversubscribe. ``os.process_cpu_count()`` respects it.
    """
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setattr(os, "process_cpu_count", lambda: 2)
    assert Settings().intra_op_threads == 2
    assert Settings.from_env().intra_op_threads == 2


def test_default_threads_stay_clamped_to_four(monkeypatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 16)
    assert Settings().intra_op_threads == 4
    assert Settings.from_env().intra_op_threads == 4


@pytest.mark.parametrize("reported", [None, 0])
def test_default_threads_never_drop_below_one(monkeypatch, reported) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: reported)
    assert Settings().intra_op_threads == 1
    assert Settings.from_env().intra_op_threads == 1


def test_threads_environment_override_beats_affinity(monkeypatch) -> None:
    monkeypatch.setattr(os, "process_cpu_count", lambda: 2)
    monkeypatch.setenv("EIDOLON_ASR_THREADS", "3")
    assert Settings.from_env().intra_op_threads == 3


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("4", {4}),
        ("4,5", {4, 5}),
        ("0-3", {0, 1, 2, 3}),
        ("4-7", {4, 5, 6, 7}),
        ("0-3,7", {0, 1, 2, 3, 7}),
        (" 4 , 5 ", {4, 5}),
        ("4,4,5", {4, 5}),
    ],
)
def test_cpu_list_parses_taskset_syntax(spec, expected) -> None:
    assert parse_cpu_list(spec) == expected


@pytest.mark.parametrize("spec", ["", "  ", ",", "5-4", "-1", "a", "0-b"])
def test_invalid_cpu_list_is_rejected(spec) -> None:
    with pytest.raises(ValueError):
        parse_cpu_list(spec)


def test_unset_affinity_leaves_the_process_alone(monkeypatch) -> None:
    monkeypatch.delenv("EIDOLON_ASR_CPU_AFFINITY", raising=False)
    assert apply_cpu_affinity() is None


def test_blank_affinity_leaves_the_process_alone(monkeypatch) -> None:
    monkeypatch.setenv("EIDOLON_ASR_CPU_AFFINITY", "   ")
    assert apply_cpu_affinity() is None


def test_affinity_is_applied_and_threads_follow(monkeypatch) -> None:
    """The whole point: pin the cores, and the pool size follows by itself."""
    applied: list[frozenset[int]] = []
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda pid, cpus: applied.append(frozenset(cpus)), raising=False
    )
    monkeypatch.setenv("EIDOLON_ASR_CPU_AFFINITY", "4,5")
    assert apply_cpu_affinity() == {4, 5}
    assert applied == [frozenset({4, 5})]

    # after pinning, process_cpu_count() reports the mask, so the pool is 2
    monkeypatch.setattr(os, "process_cpu_count", lambda: 2)
    assert Settings.from_env().intra_op_threads == 2


def test_affinity_on_a_platform_without_support_is_an_error(monkeypatch) -> None:
    """A pin that silently does nothing is what oversubscribes the pool."""
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    monkeypatch.setenv("EIDOLON_ASR_CPU_AFFINITY", "4,5")
    with pytest.raises(ValueError, match="no CPU affinity support"):
        apply_cpu_affinity()


def test_unusable_cpu_index_reports_the_valid_range(monkeypatch) -> None:
    def refuse(pid, cpus):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(os, "sched_setaffinity", refuse, raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    monkeypatch.setenv("EIDOLON_ASR_CPU_AFFINITY", "99")
    with pytest.raises(ValueError, match="valid indices are 0-7"):
        apply_cpu_affinity()


def test_the_model_root_is_told_rather_than_inferred(monkeypatch, tmp_path) -> None:
    """Two levels up from this package is the repository root only in a checkout.

    Installed, `site-packages/eidolon_models_asr/config.py` puts those same two
    levels inside the venv — and a release installs the package. The service
    started, looked for the weights under `.venv/lib/python3.13/asr/`, exited 2
    and restarted a hundred times while they sat in the component root beside
    it, and the release rolled the Host back on the readiness timeout.
    """

    import importlib

    from eidolon_models_asr import config as config_module

    monkeypatch.setenv(config_module.MODEL_ROOT_ENV, str(tmp_path))
    reloaded = importlib.reload(config_module)
    try:
        assert reloaded.PROJECT_ROOT == tmp_path.resolve()
        assert reloaded.DEFAULT_MODEL_DIR.is_relative_to(tmp_path.resolve())
        assert reloaded.DEFAULT_MANIFEST.is_relative_to(tmp_path.resolve())
    finally:
        monkeypatch.delenv(config_module.MODEL_ROOT_ENV, raising=False)
        importlib.reload(config_module)


def test_without_the_variable_a_checkout_still_needs_no_configuration() -> None:
    """The derivation stays as the fallback: it is right for the case it was
    written for, which is running out of a checkout with nothing set."""

    import importlib

    from eidolon_models_asr import config as config_module

    reloaded = importlib.reload(config_module)

    assert (reloaded.PROJECT_ROOT / "pyproject.toml").is_file()
