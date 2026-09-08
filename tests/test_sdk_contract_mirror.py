"""This service's protocol module against the contract that defines it.

`eidolon_sdk.biz.contracts.local_asr` is the source of truth: Channel speaks
the other end of this protocol and imports it from there. This service holds a
mirror instead of the dependency, because the SDK carries grpcio, livekit-api,
sqlalchemy and cryptography, and a local speech service on a constrained board
has no business installing any of them to learn four strings.

That is the arrangement the contract package prescribes for its non-Python
clients — "lightweight mirrors ... cross-repository contract tests keep those
mirrors aligned with this source" — and this is that test.

The SDK's source is read rather than imported, so nothing here depends on the
SDK being installed; `ast` rather than a regex, so the comparison is against
the values and not their formatting.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from eidolon_models_asr import protocol

_SDK_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "eidolon_sdk"
    / "eidolon_sdk"
    / "biz"
    / "contracts"
    / "local_asr.py"
)

#: Mirror name here → contract name there. Written out rather than derived from
#: a prefix rule: a rule would silently stop covering a constant whose name did
#: not fit it, and the whole point is that nothing about this goes unchecked.
_MIRRORED = {
    "PROTOCOL_VERSION": "LOCAL_ASR_PROTOCOL_VERSION",
    "CAPABILITY": "LOCAL_ASR_CAPABILITY",
    "PORT_ROLE": "LOCAL_ASR_PORT_ROLE",
    "STREAM_PATH": "LOCAL_ASR_STREAM_PATH",
    "READY_PATH": "LOCAL_ASR_READY_PATH",
    "INFO_PATH": "LOCAL_ASR_INFO_PATH",
    "START_UTTERANCE": "START_UTTERANCE",
    "START_UTTERANCE_LEGACY": "START_UTTERANCE_LEGACY",
    "END_UTTERANCE": "END_UTTERANCE",
    "PING": "PING",
    "CLOSE_STREAM": "CLOSE_STREAM",
    "UTTERANCE_STARTED": "UTTERANCE_STARTED",
    "TRANSCRIPT": "TRANSCRIPT",
    "PONG": "PONG",
    "ERROR": "ERROR",
    "AUDIO_SAMPLE_RATE": "AUDIO_SAMPLE_RATE",
    "AUDIO_CHANNELS": "AUDIO_CHANNELS",
    "AUDIO_FORMAT": "AUDIO_FORMAT",
    "ERROR_CAPACITY": "ERROR_CAPACITY",
    "ERROR_UTTERANCE_TOO_LONG": "ERROR_UTTERANCE_TOO_LONG",
    "ERROR_BAD_REQUEST": "ERROR_BAD_REQUEST",
    "ERROR_INTERNAL": "ERROR_INTERNAL",
}


def _contract_constants() -> dict[str, object]:
    if not _SDK_CONTRACT.is_file():
        pytest.skip("the contract mirror needs the sibling eidolon_sdk repository")
    tree = ast.parse(_SDK_CONTRACT.read_text(encoding="utf-8"))
    values: dict[str, object] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            first = node.targets[0]
            target = first.id if isinstance(first, ast.Name) else None
        if target is None or node.value is None:
            continue
        value = node.value
        # `frozenset({...})` and `Final` annotations are calls, not literals.
        if isinstance(value, ast.Call) and len(value.args) == 1:
            value = value.args[0]
        try:
            values[target] = ast.literal_eval(value)
        except ValueError:
            continue
    return values


def test_every_mirrored_constant_equals_the_contract() -> None:
    contract = _contract_constants()

    for mine, theirs in _MIRRORED.items():
        assert theirs in contract, f"the contract no longer defines {theirs}"
        assert getattr(protocol, mine) == contract[theirs], (
            f"protocol.{mine} is {getattr(protocol, mine)!r}; "
            f"the contract says {theirs} is {contract[theirs]!r}"
        )


def test_the_contract_has_not_grown_a_constant_this_mirror_ignores() -> None:
    """A protocol change starts in the SDK, and this is what makes it arrive.

    Without this, a new message type would be added there, Channel would send
    it, and this service would answer `bad_request` to a message the contract
    says is valid — with both sides individually passing their own tests.
    """

    contract = _contract_constants()
    ignored = {
        # Sets the SDK derives from the constants this mirror already checks,
        # and a helper that builds a message out of them.
        "CLIENT_MESSAGE_TYPES",
        "SERVER_MESSAGE_TYPES",
        "ERROR_CODES",
        "RETRYABLE_ERROR_CODES",
        # A field name only a client needs to read.
        "IS_FINAL_FIELD",
    }
    unmirrored = set(contract) - set(_MIRRORED.values()) - ignored

    assert not unmirrored, (
        f"the contract defines {sorted(unmirrored)}, which this service's "
        "protocol module neither mirrors nor names as a client-only concern"
    )
