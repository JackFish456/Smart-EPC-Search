from __future__ import annotations

import argparse
import os
from flask import Flask, jsonify, request

from epc_smart_search.app_paths import AI_DISABLE_ENV_VAR, MODEL_DIR_OVERRIDE_ENV_VAR
from epc_smart_search.config import STRICT_SYSTEM_PROMPT
from gemma_runtime import GemmaChatRuntime


def create_app() -> Flask:
    app = Flask(__name__)
    if os.environ.get(AI_DISABLE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}:
        runtime = None
        state: dict[str, object] = {"runtime": None, "ready": False, "error": "AI mode is disabled for this app instance."}
    else:
        model_override = os.environ.get(MODEL_DIR_OVERRIDE_ENV_VAR, "").strip() or None
        runtime = GemmaChatRuntime(model_path=model_override, max_new_tokens=192, temperature=0.1, top_p=0.85, enable_thinking=False)
        state = {"runtime": runtime, "ready": False, "error": None}

    if runtime is not None:
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
        previous_answer = str(payload.get("previous_answer", "")).strip()
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
                "For summary or explanation requests, keep the answer short, grounded, and specific to the user's exact ask. "
                "When the excerpts include requirements, permit conditions, compliance obligations, testing procedures, guarantees, or limits, lead with those. "
                "Treat glossary, acronym, or definition material as supporting context only."
            )
            prompt = (
                "Contract excerpts:\n"
                f"{context}\n\n"
                "User request:\n"
                f"{question}\n\n"
                "Write a short contract-grounded answer in plain English. "
                "Start with one direct answer sentence. "
                "Then summarize only the strongest supported points from the excerpts. "
                "Prefer requirements, compliance obligations, testing procedures, guarantees, and limits when they appear. "
                "Use glossary or acronym material only as secondary support. "
                "Then list the most relevant section and page if the excerpts provide them. "
                "Do not broaden beyond what the user asked, and do not add generic background. "
                "Do not mention analysis, reasoning steps, or chain-of-thought. "
                "Use only the excerpts. If the excerpts do not fully support a summary, "
                "respond exactly with: I can't verify that from the contract."
            )
        elif response_style == "deep_answer":
            system_prompt = (
                f"{STRICT_SYSTEM_PROMPT} "
                "Provide a direct answer grounded only in the excerpts. "
                "Prioritize the exact question and cite the best-supported location instead of listing generic matches."
            )
            prompt = (
                "Contract excerpts:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{question}\n\n"
                "Answer the exact question first in plain English. "
                "Keep the answer concise. "
                "After the answer, include the most relevant section/page location from the excerpts when available. "
                "Do not broaden beyond what was asked. "
                "Do not show analysis, numbered reasoning steps, or chain-of-thought. "
                "Use only the excerpts. If the excerpts do not fully support the answer, "
                "respond exactly with: I can't verify that from the contract."
            )
        elif response_style == "expand_answer":
            system_prompt = (
                f"{STRICT_SYSTEM_PROMPT} "
                "The user already saw a shorter grounded answer and explicitly asked for more detail. "
                "Expand only with details supported by the excerpts, and avoid repeating the earlier answer verbatim."
            )
            prompt = (
                "Contract excerpts:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{question}\n\n"
                "Earlier answer already shown to the user:\n"
                f"{previous_answer or 'None provided.'}\n\n"
                "Write a slightly more detailed contract-grounded answer in plain English. "
                "Answer the same question with only the extra specifics supported by the excerpts. "
                "Keep it compact, and include the most relevant section/page location when available. "
                "Do not repeat the earlier answer verbatim, do not mention analysis or chain-of-thought, and do not use outside knowledge. "
                "Use only the excerpts. If the excerpts do not fully support the expanded answer, "
                "respond exactly with: I can't verify that from the contract."
            )
        elif response_style == "candidate_select":
            system_prompt = (
                "You are selecting the best-supported contract candidate from a small provided set. "
                "Use only the candidate evidence in the prompt. "
                "Return JSON only with keys candidate_id, supporting_quote, insufficient_support. "
                "Do not add explanation, markdown, or extra keys."
            )
            prompt = (
                "Candidate evidence:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{question}\n\n"
                "Choose the single best candidate that directly answers the question. "
                "If none of the candidates directly support the answer, return "
                '{"candidate_id":"", "supporting_quote":"", "insufficient_support":true}. '
                "Otherwise return "
                '{"candidate_id":"exact id from the prompt", "supporting_quote":"short direct quote from the winning evidence", "insufficient_support":false}.'
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
        active_runtime = state["runtime"]
        if active_runtime is None:
            return jsonify({"error": state["error"] or "Gemma runtime is unavailable."}), 503
        result = active_runtime.generate(
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
