from types import SimpleNamespace

import gemma_service


def test_generate_accepts_summary_overrides(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load(self) -> None:
            return None

        def generate(self, *, user_text: str, system_prompt: str, enable_thinking=None, max_new_tokens=None):
            calls.append(
                {
                    "user_text": user_text,
                    "system_prompt": system_prompt,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return SimpleNamespace(text="Summary")

    monkeypatch.setattr(gemma_service, "GemmaChatRuntime", FakeRuntime)

    app = gemma_service.create_app()
    client = app.test_client()
    response = client.post(
        "/generate",
        json={
            "question": "summarize the fuel gas system",
            "context": "Fuel gas evidence",
            "enable_thinking": True,
            "max_new_tokens": 448,
            "response_style": "detailed_summary",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["answer"] == "Summary"
    assert calls[0]["enable_thinking"] is True
    assert calls[0]["max_new_tokens"] == 448
    assert "Write a short contract-grounded answer in plain English." in calls[0]["user_text"]
    assert "Start with one direct answer sentence." in calls[0]["user_text"]
    assert "keep the answer short, grounded, and specific" in calls[0]["system_prompt"]


def test_generate_defaults_do_not_force_summary_overrides(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load(self) -> None:
            return None

        def generate(self, *, user_text: str, system_prompt: str, enable_thinking=None, max_new_tokens=None):
            calls.append(
                {
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return SimpleNamespace(text="Answer")

    monkeypatch.setattr(gemma_service, "GemmaChatRuntime", FakeRuntime)

    app = gemma_service.create_app()
    client = app.test_client()
    response = client.post(
        "/generate",
        json={
            "question": "what does the contract say about fuel gas supply",
            "context": "Fuel gas evidence",
        },
    )

    assert response.status_code == 200
    assert calls[0]["enable_thinking"] is None
    assert calls[0]["max_new_tokens"] is None


def test_generate_accepts_deep_answer_overrides(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load(self) -> None:
            return None

        def generate(self, *, user_text: str, system_prompt: str, enable_thinking=None, max_new_tokens=None):
            calls.append(
                {
                    "user_text": user_text,
                    "system_prompt": system_prompt,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return SimpleNamespace(text="Deep answer")

    monkeypatch.setattr(gemma_service, "GemmaChatRuntime", FakeRuntime)

    app = gemma_service.create_app()
    client = app.test_client()
    response = client.post(
        "/generate",
        json={
            "question": "what does the contract say about fuel gas supply",
            "context": "Fuel gas evidence",
            "enable_thinking": True,
            "max_new_tokens": 896,
            "response_style": "deep_answer",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["answer"] == "Deep answer"
    assert calls[0]["enable_thinking"] is True
    assert calls[0]["max_new_tokens"] == 896
    assert "Answer the exact question first in plain English." in calls[0]["user_text"]
    assert "After the answer, include the most relevant section/page location" in calls[0]["user_text"]
    assert "cite the best-supported location instead of listing generic matches" in calls[0]["system_prompt"]


def test_generate_accepts_expand_answer_overrides(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load(self) -> None:
            return None

        def generate(self, *, user_text: str, system_prompt: str, enable_thinking=None, max_new_tokens=None):
            calls.append(
                {
                    "user_text": user_text,
                    "system_prompt": system_prompt,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return SimpleNamespace(text="Expanded answer")

    monkeypatch.setattr(gemma_service, "GemmaChatRuntime", FakeRuntime)

    app = gemma_service.create_app()
    client = app.test_client()
    response = client.post(
        "/generate",
        json={
            "question": "what does the contract say about fuel gas supply",
            "context": "Fuel gas evidence",
            "previous_answer": "Short answer.",
            "max_new_tokens": 896,
            "response_style": "expand_answer",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["answer"] == "Expanded answer"
    assert calls[0]["enable_thinking"] is None
    assert calls[0]["max_new_tokens"] == 896
    assert "Earlier answer already shown to the user:" in calls[0]["user_text"]
    assert "Keep it compact, and include the most relevant section/page location" in calls[0]["user_text"]
    assert "explicitly asked for more detail" in calls[0]["system_prompt"]


def test_generate_candidate_select_uses_structured_json_prompt(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeRuntime:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def load(self) -> None:
            return None

        def generate(self, *, user_text: str, system_prompt: str, enable_thinking=None, max_new_tokens=None):
            calls.append(
                {
                    "user_text": user_text,
                    "system_prompt": system_prompt,
                    "enable_thinking": enable_thinking,
                    "max_new_tokens": max_new_tokens,
                }
            )
            return SimpleNamespace(text='{"candidate_id":"candidate_b","supporting_quote":"quote","insufficient_support":false}')

    monkeypatch.setattr(gemma_service, "GemmaChatRuntime", FakeRuntime)

    app = gemma_service.create_app()
    client = app.test_client()
    response = client.post(
        "/generate",
        json={
            "question": "What is the turbine we are using?",
            "context": "Candidate ID: candidate_a\nCandidate ID: candidate_b",
            "response_style": "candidate_select",
            "max_new_tokens": 192,
        },
    )

    assert response.status_code == 200
    assert response.get_json()["answer"] == '{"candidate_id":"candidate_b","supporting_quote":"quote","insufficient_support":false}'
    assert calls[0]["max_new_tokens"] == 192
    assert calls[0]["enable_thinking"] is None
    assert "Choose the single best candidate that directly answers the question." in calls[0]["user_text"]
    assert '"candidate_id"' in calls[0]["user_text"]
    assert "return json only with keys candidate_id, supporting_quote, insufficient_support" in calls[0]["system_prompt"].lower()
