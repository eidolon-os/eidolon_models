from pathlib import Path

import pytest

from eidolon_models_laya.config import DEFAULT_PORT, Settings, is_loopback


def test_defaults_are_loopback_torch():
    s = Settings.from_env({"EIDOLON_LAYA_HOME": "/srv/laya"})
    assert s.backend == "torch"
    assert s.host == "127.0.0.1" and s.port == DEFAULT_PORT
    assert s.model_dir == Path("/srv/laya/models/laya-multilingual/1c5edc17")
    assert s.max_len is None and s.api_key is None


def test_env_parsing():
    s = Settings.from_env(
        {
            "EIDOLON_LAYA_BACKEND": "ONNX",
            "EIDOLON_LAYA_THREADS": "1",
            "EIDOLON_LAYA_MAX_LEN": "2048",
            "EIDOLON_LAYA_API_KEY": "  k  ",
            "EIDOLON_LAYA_MODEL_DIR": "models/x",
            "EIDOLON_LAYA_HOME": "/h",
        }
    )
    assert s.backend == "onnx" and s.threads == 1 and s.max_len == 2048 and s.api_key == "k"
    assert s.model_dir == Path("/h/models/x")


@pytest.mark.parametrize(
    "env",
    [
        {"EIDOLON_LAYA_BACKEND": "tensorrt"},
        {"EIDOLON_LAYA_THREADS": "many"},
        {"EIDOLON_LAYA_MAX_LEN": "10"},
        {"EIDOLON_LAYA_ALLOW_NO_AUTH": "maybe"},
    ],
)
def test_bad_values_are_rejected(env):
    with pytest.raises(ValueError):
        Settings.from_env(env)


def test_public_bind_needs_a_key():
    with pytest.raises(ValueError, match="API_KEY"):
        Settings(host="0.0.0.0").validate_exposure()
    Settings(host="0.0.0.0", api_key="k").validate_exposure()
    Settings(host="0.0.0.0", allow_no_auth=True).validate_exposure()
    Settings(host="127.0.0.1").validate_exposure()


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("::1", True),
        ("localhost", True),
        ("0.0.0.0", False),
        ("10.0.0.5", False),
        ("eidolon.local", False),
    ],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected
