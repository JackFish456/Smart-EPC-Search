from __future__ import annotations

import argparse
from flask import Flask, jsonify, request

from epc_smart_search.config import STRICT_SYSTEM_PROMPT
from gemma_runtime import GemmaChatRuntime


def create_app() -> Flask:
    app = Flask(__name__)
    runtime = GemmaChatRuntime(max_new_tokens=280, temperature=0.1, top_p=0.85, enable_thinking=False)
    state: dict[str, object] = {"runtime": runtime, "ready": False, "error": None}

    try:
        runtime.load()
        state["ready"] = True
    except Exception as exc:  # pragma: no cover - startup path
        state["error"] = str(exc)

    @app.get("/health")
    def health():
        if state["ready"]:
            return jsonify({"status": "ok"})
        return jsonify({"status": "error", "error": state["error"]}), 503

    @app.post("/generate")
    def generate():
        payload = request.get_json(silent=True) or {}
        question = str(payload.get("question", "")).strip()
        context = str(payload.get("context", "")).strip()
        if not question or not context:
            return jsonify({"error": "question and context are required"}), 400
        if not state["ready"]:
            return jsonify({"error": state["error"] or "Gemma runtime is unavailable."}), 503
        prompt = (
            "Contract excerpts:\n"
            f"{context}\n\n"
            "User question:\n"
            f"{question}\n\n"
            "Return only the final answer in plain English. "
            "Do not show analysis, numbered steps, or chain-of-thought. "
            "Answer using only the excerpts. If the excerpts do not fully support the answer, "
            "respond exactly with: I can't verify that from the contract."
        )
        result = runtime.generate(user_text=prompt, system_prompt=STRICT_SYSTEM_PROMPT)
        return jsonify({"answer": result.text.strip()})

    return app


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8051)
    args = parser.parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
