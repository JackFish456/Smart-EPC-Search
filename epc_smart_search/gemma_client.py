from __future__ import annotations

import atexit
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from epc_smart_search.app_paths import APP_DATA_ROOT, INSTALL_ROOT, MODEL_DIR_OVERRIDE_ENV_VAR, WORKSPACE_ROOT, resolve_gemma_launch_spec
from epc_smart_search.config import GEMMA_SERVICE_HOST, GEMMA_SERVICE_PORT


@dataclass(slots=True, frozen=True)
class GemmaAvailability:
    available: bool
    message: str
    mode: str


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

    def get_availability(self) -> GemmaAvailability:
        launch_spec = resolve_gemma_launch_spec()
        if launch_spec.available:
            if launch_spec.mode == "bundled_service":
                return GemmaAvailability(True, "AI mode is available in this build.", launch_spec.mode)
            return GemmaAvailability(True, "AI mode is available from the local Gemma environment.", launch_spec.mode)
        return GemmaAvailability(
            False,
            launch_spec.reason or "AI mode unavailable on this machine; citation mode is active.",
            launch_spec.mode,
        )

    def is_available(self) -> bool:
        return self.get_availability().available

    def ensure_running(self) -> None:
        if self._is_healthy():
            return
        launch_spec = resolve_gemma_launch_spec()
        if not launch_spec.available or launch_spec.service_path is None:
            raise RuntimeError(launch_spec.reason or "AI mode unavailable on this machine; citation mode is active.")

        self._stdout_handle = self.stdout_log_path.open("a", encoding="utf-8")
        self._stderr_handle = self.stderr_log_path.open("a", encoding="utf-8")
        self._write_launch_banner()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        launch_env = self._build_launch_env(launch_spec.model_dir)
        command = self._build_launch_command(launch_spec, self.port)
        launch_cwd = launch_spec.service_path.parent if launch_spec.mode == "bundled_service" else WORKSPACE_ROOT
        self._process = subprocess.Popen(
            command,
            cwd=str(launch_cwd if launch_cwd.exists() else INSTALL_ROOT),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
            creationflags=creation_flags,
            env=launch_env,
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

    @staticmethod
    def _build_launch_command(launch_spec, port: int) -> list[str]:
        if launch_spec.mode == "bundled_service":
            return [str(launch_spec.service_path), "--port", str(port)]
        return [str(launch_spec.service_path), str(WORKSPACE_ROOT / "gemma_service.py"), "--port", str(port)]

    @staticmethod
    def _build_launch_env(model_dir: Path | None) -> dict[str, str]:
        launch_env = dict(os.environ)
        if model_dir is not None:
            launch_env[MODEL_DIR_OVERRIDE_ENV_VAR] = str(model_dir)
        return launch_env
