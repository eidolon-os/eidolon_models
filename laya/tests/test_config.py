from pathlib import Path

import pytest

from eidolon_models_laya.config import (
    DEFAULT_MODEL,
    DEFAULT_PORT,
    Settings,
    backend_for,
    default_services_file,
    host_capabilities,
    is_loopback,
    laya_home,
)


def test_defaults_are_loopback_torch():
    s = Settings()
    assert s.backend == "torch"
    assert s.host == "127.0.0.1" and s.port == DEFAULT_PORT
    assert s.model_dir == DEFAULT_MODEL
    assert s.max_len is None and s.api_key is None


def test_the_backend_follows_the_hosts_npu_declaration():
    assert backend_for(frozenset({"rknpu2", "local_laya"})) == "rknn"
    assert backend_for(frozenset({"local_laya"})) == "torch"
    assert host_capabilities({"EIDOLON_HOST_CAPABILITIES": "local_laya, rknpu2,"}) == {
        "local_laya",
        "rknpu2",
    }
    assert host_capabilities({}) == frozenset()


def test_the_product_services_on_an_npu_board():
    home = Settings.for_service("smart_home", capabilities=frozenset({"rknpu2", "local_laya"}))
    assert home.backend == "rknn" and home.port == 8771 and home.max_pending == 1
    assert home.rknn_placement == "1:128,160,256,384,512|2:128,160,256" and home.speculative
    assert home.model_dir == Path("/var/lib/eidolon/models/laya-smart-home-c10-b160-rknn-e8254243")
    assert not home.enable_participation
    team = Settings.for_service("participation", capabilities=frozenset({"rknpu2"}))
    assert team.backend == "rknn" and team.port == 8773 and team.enable_participation
    assert team.rknn_placement == "0:384,512,640" and not team.speculative


def test_the_product_services_from_a_source_checkout():
    home = Settings.for_service("smart_home", capabilities=frozenset())
    assert home.backend == "torch" and home.device == "auto"
    assert home.model_dir == laya_home() / "models" / "laya-smart-home" / "e8254243"
    assert home.rknn_placement is None and home.max_pending == 4


def test_the_two_services_share_no_npu_core():
    def cores(placement: str) -> set[str]:
        return {part.split(":")[0] for part in placement.split("|")}

    npu = frozenset({"rknpu2"})
    home = Settings.for_service("smart_home", capabilities=npu).rknn_placement
    team = Settings.for_service("participation", capabilities=npu).rknn_placement
    assert not cores(home) & cores(team)


def _services(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "services.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_server_file_with_a_key(tmp_path: Path):
    key = tmp_path / "api-key"
    key.write_text("secret\n", encoding="utf-8")
    path = _services(
        tmp_path,
        f"""schema_version = 1
[systemone]
host = "0.0.0.0"
threads = 1
api_key_file = "{key}"
[systemone.onnx]
model_dir = "models/laya-multilingual/1c5edc17"
""",
    )
    s = Settings.for_service("systemone", path=path, backend="onnx")
    assert s.api_key == "secret" and s.threads == 1 and s.backend == "onnx"
    assert s.model_dir == laya_home() / "models" / "laya-multilingual" / "1c5edc17"
    s.validate_exposure()


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("schema_version = 2\n", "schema_version"),
        ("schema_version = 1\n[kitchen]\n", "unknown tables"),
        ("schema_version = 1\n[smart_home]\nportt = 1\n", "not a setting"),
        ('schema_version = 1\n[smart_home]\nport = "8771"\n', "wrong type"),
        ("schema_version = 1\n[smart_home]\nport = true\n", "wrong type"),
        ('schema_version = 1\n[smart_home.torch]\ndevice = "tpu"\nmodel_dir = "m"\n', "device"),
        ('schema_version = 1\n[smart_home.torch]\ndevice = "cpu"\n', "no model_dir"),
    ],
)
def test_a_bad_services_file_is_refused(tmp_path: Path, text: str, message: str):
    with pytest.raises(ValueError, match=message):
        Settings.for_service("smart_home", path=_services(tmp_path, text), backend="torch")


def test_a_backend_the_service_does_not_describe_is_refused():
    with pytest.raises(ValueError, match="no 'onnx' backend"):
        Settings.for_service("smart_home", backend="onnx")
    with pytest.raises(ValueError, match="no service 'systemone'"):
        Settings.for_service("systemone")


def test_a_missing_model_says_why(tmp_path: Path):
    s = Settings(service="smart_home", backend="rknn", model_dir=tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="artifact is not installed"):
        s.require_model()
    with pytest.raises(FileNotFoundError, match="does not run Laya"):
        Settings(service="smart_home", model_dir=tmp_path / "absent").require_model()


def test_public_bind_needs_a_key():
    with pytest.raises(ValueError, match="api_key_file"):
        Settings(host="0.0.0.0").validate_exposure()
    Settings(host="0.0.0.0", api_key="k").validate_exposure()
    Settings(host="0.0.0.0", allow_no_auth=True).validate_exposure()
    Settings(host="127.0.0.1").validate_exposure()


def test_the_default_services_file_is_this_checkouts():
    assert default_services_file() == laya_home() / "deploy" / "services.toml"


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
