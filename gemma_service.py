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
        enable_thinking = payload.get("enable_thinking")
        max_new_tokens = payload.get("max_new_tokens")
        response_style = str(payload.get("response_style", "")).strip().lower()
        if not question or not context:
            return jsonify({"error": "question and context are required"}), 400
        if not state["ready"]:
            return jsonify({"error": state["error"] or "Gemma runtime is unavailable."}), 503
        if enable_thinking is not None:
            enable_thinking = bool(enable_thinking)
        if max_new_tokens is not None:
            try:
                max_new_tokens = int(max_new_tokens)
            except (TypeError, ValueError):
                return jsonify({"error": "max_new_tokens must be an integer"}), 400
            if max_new_tokens <= 0:
                return jsonify({"error": "max_new_tokens must be positive"}), 400
        system_prompt = STRICT_SYSTEM_PROMPT
        if response_style == "detailed_summary":
            system_prompt = (
                f"{STRICT_SYSTEM_PROMPT} "
                "For summary or explanation requests, provide a detailed, well-organized synthesis grounded in the excerpts."
            )
            prompt = (
                "Contract excerpts:\n"
                f"{context}\n\n"
                "User request:\n"
                f"{question}\n\n"
                "Write a detailed contract-grounded summary in plain English. "
                "Cover the most important obligations, scope, thresholds, dates, conditions, exceptions, and dependencies that appear in the excerpts. "
                "Prefer short markdown-style section headers and bullet lists when they help readability. "
                "Do not mention analysis, reasoning steps, or chain-of-thought. "
                "Use only the excerpts. If the excerpts do not fully support a summary, "
                "respond exactly with: I can't verify that from the contract."
            )
        else:
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
        result = runtime.generate(
            user_text=prompt,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
            max_new_tokens=max_new_tokens,
        )
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
