"""A small client for the laya decision API — standard library only.

    from laya_client import LayaClient
    client = LayaClient()                                  # $LAYA_URL, default: the ECS demo
    result = client.decide(state, questions)
    result["answers"]["dept"]["choice"]

Copy this file into your project as-is; it has no dependencies.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_URL = "http://8.141.101.214:8771"


class LayaError(RuntimeError):
    """A non-2xx reply. ``code`` is the server's error code, e.g. ``invalid_question``."""

    def __init__(self, status: int, code: str | None, message: str | None):
        super().__init__(f"HTTP {status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


class LayaClient:
    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        retries: int = 3,
        use_system_proxy: bool = False,
    ):
        self.base_url = (base_url or os.environ.get("LAYA_URL") or DEFAULT_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("LAYA_API_KEY")
        self.timeout = timeout
        self.retries = retries
        # A desktop proxy (e.g. Clash on 7890) only adds a hop to a server we can reach.
        handlers = [] if use_system_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    def decide(
        self, state: Any, questions: dict[str, dict], *, truncate_left: bool = False
    ) -> dict:
        """POST /v1/systemone. ``questions`` maps an id to a choice / score / noul definition."""
        body = {"state": state, "questions": questions, "options": {"truncate_left": truncate_left}}
        return self._call("POST", "/v1/systemone", body)

    def info(self) -> dict:
        return self._call("GET", "/v1/info")

    def ready(self) -> bool:
        try:
            return self._call("GET", "/readyz").get("status") == "ready"
        except (LayaError, OSError):
            return False

    def _call(self, method: str, path: str, body: dict | None = None) -> dict:
        headers = {"Accept": "application/json"}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body, ensure_ascii=False).encode()
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path, data=data, headers=headers, method=method
        )
        for attempt in range(self.retries):
            try:
                with self._opener.open(request, timeout=self.timeout) as resp:
                    return json.load(resp)
            except urllib.error.HTTPError as err:
                # 503 = the server's queue is full; it says when to come back.
                if err.code == 503 and attempt + 1 < self.retries:
                    time.sleep(float(err.headers.get("Retry-After", "1")))
                    continue
                try:
                    detail = json.loads(err.read() or b"{}").get("error", {})
                except ValueError:
                    detail = {}
                raise LayaError(err.code, detail.get("code"), detail.get("message")) from None
        raise AssertionError("unreachable")
