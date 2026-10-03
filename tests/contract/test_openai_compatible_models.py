"""A model served over the OpenAI-compatible API, as the platform calls it.

Checked against a server started inside the test that records each request.
That is a contract check: it shows what is sent and how the reply is read.
Whether a given provider honours a seed, or accepts these fields at all, only
a live call with real keys can say, and the platform records what it sent so
that the question can be asked of a stored run.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from adapters.openai_compatible import ModelServerNotConfigured, OpenAICompatibleGenerator


class _Server:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.reply: dict[str, Any] = {
            "model": "served-model-2026", "system_fingerprint": "fp_1",
            "choices": [{"message": {"content": "  an answer  "}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        }
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length)) if length else None
                server.requests.append({"method": self.command, "path": self.path, "body": body,
                                        "headers": {k.lower(): v for k, v in self.headers.items()}})
                if self.path.endswith("/models"):
                    payload: Any = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
                elif self.path.endswith("/embeddings"):
                    payload = {"data": [{"index": i, "embedding": [float(i), 1.0, 0.0]}
                                        for i, _ in enumerate(body["input"])],
                               "model": body["model"]}
                else:
                    payload = server.reply
                raw = json.dumps(payload).encode()
                self.send_response(server.status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = do_POST = _answer

            def log_message(self, *args: Any) -> None:
                pass

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.http.server_address[1]}/v1"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()


@pytest.fixture
def server() -> Iterator[_Server]:
    fake = _Server()
    yield fake
    fake.http.shutdown()


def _generator(server: _Server, **kwargs: Any) -> OpenAICompatibleGenerator:
    return OpenAICompatibleGenerator(base_url=server.base_url, api_key="the-key", model="model-a", **kwargs)


def test_a_prompt_goes_out_as_one_user_message_with_the_key(server: _Server) -> None:
    assert _generator(server).generate("What is the period?", temperature=0.2, max_tokens=64) == "an answer"
    sent = server.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["authorization"] == "Bearer the-key"
    assert sent["body"]["model"] == "model-a"
    assert sent["body"]["messages"] == [{"role": "user", "content": "What is the period?"}]
    assert (sent["body"]["temperature"], sent["body"]["max_tokens"]) == (0.2, 64)


def test_a_seed_is_sent_only_when_the_run_fixed_one(server: _Server) -> None:
    _generator(server).generate("q")
    _generator(server).with_seed(7).generate("q")
    assert "seed" not in server.requests[0]["body"]
    assert server.requests[1]["body"]["seed"] == 7


def test_what_was_sent_and_what_served_it_are_kept_for_the_run(server: _Server) -> None:
    """A provider may ignore a seed, and may serve another model than the one
    named. Neither can be seen from the answer, so both are recorded."""
    generator = _generator(server).with_seed(7)
    generator.generate("q")
    assert generator.last_call == {"seed_sent": True, "served_model": "served-model-2026",
                                   "system_fingerprint": "fp_1"}
    assert generator._last_token_counts == {"prompt_eval_count": 11, "eval_count": 3}


def test_another_model_and_another_seed_make_copies(server: _Server) -> None:
    base = _generator(server)
    other = base.with_model("model-b")
    assert other is not base and base.with_model("model-a") is base
    other.generate("q")
    assert server.requests[0]["body"]["model"] == "model-b"
    assert server.requests[0]["headers"]["authorization"] == "Bearer the-key"


def test_a_realm_s_own_server_and_key_replace_the_gateway_s(server: _Server) -> None:
    base = OpenAICompatibleGenerator(base_url="http://gateway.invalid/v1", api_key="gateway-key", model="m")
    bound = base.for_realm({"openai_compatible": {"base_url": server.base_url, "api_key": "realm-key",
                                                   "model": "model-b"}})
    bound.generate("q")
    assert server.requests[0]["headers"]["authorization"] == "Bearer realm-key"
    assert server.requests[0]["body"]["model"] == "model-b"


def test_a_realm_that_declares_no_such_server_leaves_it_as_it_was(server: _Server) -> None:
    base = _generator(server)
    assert base.for_realm({"ollama": {"host": "h"}}) is base and base.for_realm(None) is base


def test_a_server_nobody_configured_says_so_instead_of_calling_somewhere() -> None:
    with pytest.raises(ModelServerNotConfigured):
        OpenAICompatibleGenerator(base_url=None, api_key=None, model=None).generate("q")


def test_a_structured_reply_is_asked_for_in_the_api_s_own_way(server: _Server) -> None:
    _generator(server).generate("q", response_format="json")
    assert server.requests[0]["body"]["response_format"] == {"type": "json_object"}


def test_a_refusal_by_the_server_carries_its_status_and_no_key(server: _Server) -> None:
    server.status = 401
    with pytest.raises(Exception) as raised:
        _generator(server).generate("q")
    assert "401" in str(raised.value) and "the-key" not in str(raised.value)


def test_the_models_a_server_offers_are_listed(server: _Server) -> None:
    assert _generator(server).list_models() == ["model-a", "model-b"]
    assert server.requests[0]["path"] == "/v1/models"
    assert server.requests[0]["headers"]["authorization"] == "Bearer the-key"


def test_where_it_is_served_from_is_said_without_the_key(server: _Server) -> None:
    said = _generator(server).served_from()
    assert said == {"provider": "openai_compatible", "address": server.base_url, "model": "model-a",
                    "remote": False}
    assert "the-key" not in json.dumps(said)


# ── Embeddings ────────────────────────────────────────────────────────────────

from adapters.openai_compatible import OpenAICompatibleEmbedder  # noqa: E402


def _embedder(server: _Server, **kwargs: Any) -> OpenAICompatibleEmbedder:
    return OpenAICompatibleEmbedder(base_url=server.base_url, api_key="the-key", model="Embed/Model-3", **kwargs)


def test_texts_come_back_as_vectors_in_the_order_they_went(server: _Server) -> None:
    vectors = _embedder(server).embed(["first", "second", "third"])
    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0]
    sent = server.requests[0]
    assert sent["path"] == "/v1/embeddings" and sent["headers"]["authorization"] == "Bearer the-key"
    assert sent["body"] == {"model": "Embed/Model-3", "input": ["first", "second", "third"]}


def test_many_texts_go_out_in_batches(server: _Server) -> None:
    vectors = _embedder(server, batch=2).embed(["a", "b", "c", "d", "e"])
    assert len(vectors) == 5
    assert [len(r["body"]["input"]) for r in server.requests] == [2, 2, 1]


def test_the_identifier_names_the_model_so_its_vectors_get_a_collection_of_their_own(server: _Server) -> None:
    from adapters.qdrant import _collection_name
    embedder = _embedder(server)
    assert embedder.embedder_id == "openai_compatible__embed-model-3"
    assert _collection_name("fixed", embedder.embedder_id, "c", "r") != _collection_name("fixed", "bge_m3", "c", "r")


def test_the_size_of_a_vector_is_asked_once(server: _Server) -> None:
    embedder = _embedder(server)
    assert embedder.vector_size == 3 and embedder.vector_size == 3
    assert len(server.requests) == 1


def test_a_realm_s_embedding_model_is_not_its_answering_model(server: _Server) -> None:
    base = OpenAICompatibleEmbedder(base_url="http://gateway.invalid/v1", model="m")
    bound = base.for_realm({"openai_compatible": {"base_url": server.base_url, "api_key": "realm-key",
                                                   "model": "answers", "embedding_model": "embeds"}})
    bound.embed(["a"])
    assert server.requests[0]["body"]["model"] == "embeds"
    assert server.requests[0]["headers"]["authorization"] == "Bearer realm-key"


def test_an_embedder_with_no_model_named_says_so(server: _Server) -> None:
    with pytest.raises(ModelServerNotConfigured):
        OpenAICompatibleEmbedder(base_url=server.base_url).embed(["a"])
    assert server.requests == []


def test_a_reply_with_fewer_vectors_than_texts_is_refused(server: _Server, monkeypatch) -> None:
    """Paired by position, a short reply would attach vectors to the wrong
    texts and the load would finish."""
    import adapters.openai_compatible as module
    monkeypatch.setattr(module._Server, "post", lambda self, path, payload: {"data": [{"index": 0, "embedding": [1.0]}]})
    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        _embedder(server).embed(["a", "b"])
