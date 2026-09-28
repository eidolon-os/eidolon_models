"""Offline contract check for the GLM data generator's shared HTTP client."""

from __future__ import annotations

import io
import json
import urllib.request

from eidolon_laya_train.generators import ChatClient


def test_chat_client_preserves_usage_and_glm_request_options(monkeypatch):
    sent = []

    class Opener:
        def open(self, request, timeout):
            sent.append({"url": request.full_url, "body": json.loads(request.data),
                         "authorization": request.get_header("Authorization"), "timeout": timeout})
            return io.BytesIO(json.dumps({"id": "fake-response", "model": "glm-5.3-flash",
                                          "choices": [{"message": {"content": '{"episodes":[]}'}}],
                                          "usage": {"total_tokens": 42}}).encode())

    monkeypatch.setenv("EIDOLON_IP_DATA_API_KEY", "test-only")
    monkeypatch.setattr(urllib.request, "build_opener", lambda *args: Opener())
    client = ChatClient({"base_url": "https://api.z.ai/api/paas/v4", "model": "glm-5.3-flash",
                         "api_key_env": "EIDOLON_IP_DATA_API_KEY", "max_tokens": 6000,
                         "thinking": {"type": "enabled"}, "reasoning_effort": "low"})
    response = client.complete_with_meta("return JSON", temperature=1.0)

    assert response == {"text": '{"episodes":[]}', "id": "fake-response",
                        "model": "glm-5.3-flash", "usage": {"total_tokens": 42}}
    assert sent == [{"url": "https://api.z.ai/api/paas/v4/chat/completions",
                     "body": {"model": "glm-5.3-flash",
                              "messages": [{"role": "user", "content": "return JSON"}],
                              "temperature": 1.0, "max_tokens": 6000,
                              "thinking": {"type": "enabled"}, "reasoning_effort": "low"},
                     "authorization": "Bearer test-only", "timeout": 120.0}]
