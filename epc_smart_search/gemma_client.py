from __future__ import annotations

import atexit
import json
import subprocess
import time
from pathlib import Path

import requests

from epc_smart_search.app_paths import APP_DATA_ROOT, GEMMA_TEST_PYTHON, WORKSPACE_ROOT
from epc_smart_search.config import GEMMA_SERVICE_HOST, GEMMA_SERVICE_PORT


class GemmaServiceClient:
    def __init__(self, host: str = GEMMA_SERVICE_HOST, port: int = GEMMA_SERVICE_PORT) -> None:
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self._process: subprocess.Popen[str] | None = None
        self._stdout_handle = None
        self._stderr_handle = None
        self.stdout_log_path = APP_DATA_ROOT / "gemma_service.stdout.log"
        self.stderr_log_path = APP_DATA_ROOT / "gemma_service.stderr.log"
        atexit.register(self.stop)

    def ensure_running(self) -> None:
        if self._is_healthy():
            return
        if not GEMMA_TEST_PYTHON.exists():
            raise RuntimeError(
                f"Gemma Python not found: {GEMMA_TEST_PYTHON}\n"
                "Set EPC_GEMMA_PYTHON or EPC_GEMMA_TEST_ROOT to your Gemma environment."
            )

        self._stdout_handle = self.stdout_log_path.open("a", encoding="utf-8")
        self._stderr_handle = self.stderr_log_path.open("a", encoding="utf-8")
        self._write_launch_banner()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._process = subprocess.Popen(
            [str(GEMMA_TEST_PYTHON), str(WORKSPACE_ROOT / "gemma_service.py"), "--port", str(self.port)],
            cwd=str(WORKSPACE_ROOT),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            creationflags=creation_flags,
        )

        deadline = time.time() + 90
        while time.time() < deadline:
            if self._is_healthy():
                return
            if self._process.poll() is not None:
                stderr_tail = self._tail_log(self.stderr_log_path)
                raise RuntimeError(
                    f"Gemma helper service exited with code {self._process.returncode}.\n"
                    f"See {self.stderr_log_path} for details."
                    + (f"\nRecent stderr:\n{stderr_tail}" if stderr_tail else "")
                )
            time.sleep(1.0)

        health_error = self._health_error()
        raise RuntimeError(
            "Gemma helper service did not become ready within 90 seconds.\n"
            + (f"Health endpoint reported: {health_error}\n" if health_error else "")
            + f"See {self.stdout_log_path} and {self.stderr_log_path} for details."
        )

    def ask(
        self,
        question: str,
        context: str,
        *,
        enable_thinking: bool | None = None,
        max_new_tokens: int | None = None,
        response_style: str | None = None,
        previous_answer: str | None = None,
    ) -> str:
        self.ensure_running()
        payload: dict[str, object] = {"question": question, "context": context}
        if enable_thinking is not None:
            payload["enable_thinking"] = enable_thinking
        if max_new_tokens is not None:
            payload["max_new_tokens"] = max_new_tokens
        if response_style is not None:
            payload["response_style"] = response_style
        if previous_answer is not None:
            payload["previous_answer"] = previous_answer
        response = requests.post(f"{self.base_url}/generate", json=payload, timeout=300)
        response.raise_for_status()
        return str(response.json().get("answer", "")).strip()

    def stop(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)

    def _is_healthy(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3)
            return response.ok
        except Exception:
            return False

    def _health_error(self) -> str | None:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=3)
        except Exception:
            return None
        try:
            payload = response.json()
        except json.JSONDecodeError:
            return response.text.strip() or None
        return str(payload.get("error", "")).strip() or None

    def _write_launch_banner(self) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for handle, label in ((self._stdout_handle, "stdout"), (self._stderr_handle, "stderr")):
            if handle is None:
                continue
            handle.write(f"\n[{timestamp}] Launching Gemma service ({label})\n")
            handle.flush()

    @staticmethod
    def _tail_log(path: Path, *, limit: int = 1200) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")[-limit:].strip()
