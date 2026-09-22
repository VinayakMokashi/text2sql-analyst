"""The OpenAI-compatible client's retry behaviour, without any network access."""

import httpx
import openai
import pytest

from text2sql.llm import LLMError
from text2sql.llm.openai_compat import OpenAICompatibleLLM, retry_delay


def rate_limit_error(retry_after: str | None = "0") -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        429, headers=headers, request=httpx.Request("POST", "http://test/chat/completions")
    )
    return openai.RateLimitError("rate limited", response=response, body=None)


def ok_completion(text: str):
    return openai.types.chat.ChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    )


@pytest.fixture()
def llm():
    return OpenAICompatibleLLM("test", "m", "http://test", "key", max_retries=3, max_wait_s=5)


def test_rate_limits_are_retried_and_reported_separately(llm, monkeypatch):
    replies = [rate_limit_error("0.01"), rate_limit_error("0.01"), ok_completion("SELECT 1")]

    def create(**_kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm._client.chat.completions, "create", create)
    resp = llm.complete("s", "u")
    assert resp.text == "SELECT 1"
    assert resp.waited_s == pytest.approx(0.02)
    assert resp.latency_s < 1.0


def test_gives_up_when_the_server_asks_to_wait_too_long(llm, monkeypatch):
    def create(**_kwargs):
        raise rate_limit_error("3600")  # e.g. a daily quota is exhausted

    monkeypatch.setattr(llm._client.chat.completions, "create", create)
    with pytest.raises(LLMError, match="daily") as info:
        llm.complete("s", "u")
    assert info.value.rate_limited  # lets the app tell the user to come back later


def test_non_retryable_errors_fail_fast(llm, monkeypatch):
    calls = []

    def create(**_kwargs):
        calls.append(1)
        response = httpx.Response(401, request=httpx.Request("POST", "http://test"))
        raise openai.AuthenticationError("bad key", response=response, body=None)

    monkeypatch.setattr(llm._client.chat.completions, "create", create)
    with pytest.raises(LLMError, match="bad key") as info:
        llm.complete("s", "u")
    assert len(calls) == 1
    assert not info.value.rate_limited


def test_retry_delay_falls_back_to_backoff():
    assert retry_delay(rate_limit_error("2.5"), attempt=0) == 2.5
    assert retry_delay(rate_limit_error(None), attempt=3) == 8.0
    assert retry_delay(rate_limit_error("soon"), attempt=0) == 1.0


def llm_with_transport(handler, **kwargs):
    client = OpenAICompatibleLLM("test", "m", "http://test/v1", "key", **kwargs)
    client._client = openai.OpenAI(
        base_url="http://test/v1",
        api_key="key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"error": {"message": "upstream failed"}}),
        httpx.Response(200, text="<html>a web page</html>", headers={"content-type": "text/html"}),
        httpx.Response(
            200,
            json={
                "id": "x",
                "object": "chat.completion",
                "created": 0,
                "model": "m",
                "choices": [],
            },
        ),
    ],
)
def test_odd_http_200_responses_become_llm_errors(response):
    llm = llm_with_transport(lambda _request: response)
    with pytest.raises(LLMError, match="unexpected response"):
        llm.complete("s", "u")


def test_reply_cut_off_before_any_answer_is_reported():
    body = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"role": "assistant", "content": None},
            }
        ],
    }
    llm = llm_with_transport(lambda _request: httpx.Response(200, json=body))
    with pytest.raises(LLMError, match="token limit"):
        llm.complete("s", "u")


def test_unreachable_server_gives_up_after_a_few_tries(monkeypatch):
    calls = []

    def refuse(request):
        calls.append(1)
        raise httpx.ConnectError("connection refused", request=request)

    monkeypatch.setattr("time.sleep", lambda _s: None)
    llm = llm_with_transport(refuse, max_retries=8)
    with pytest.raises(LLMError, match="server is running"):
        llm.complete("s", "u")
    assert len(calls) == 3  # first try + 2 retries, not 9
