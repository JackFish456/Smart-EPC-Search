import tempfile
import uuid
from pathlib import Path

import epc_smart_search.gemma_client as gemma_client


def test_ensure_running_surfaces_exit_and_log_tail(monkeypatch) -> None:
    temp_root = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"gemma_client_{uuid.uuid4().hex[:8]}"
    temp_root.mkdir(parents=True, exist_ok=True)
    fake_python = temp_root / "python.exe"
    fake_python.write_text("", encoding="utf-8")
    stderr_text = "service failed to start"

    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            kwargs["stderr"].write(stderr_text)
            kwargs["stderr"].flush()
            self.returncode = 23

        def poll(self) -> int:
            return self.returncode

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return self.returncode

        def kill(self) -> None:
            return None

    monkeypatch.setattr(gemma_client, "GEMMA_TEST_PYTHON", fake_python)
    monkeypatch.setattr(gemma_client.requests, "get", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(gemma_client.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(gemma_client.time, "sleep", lambda seconds: None)

    client = gemma_client.GemmaServiceClient()
    client.stdout_log_path = temp_root / "stdout.log"
    client.stderr_log_path = temp_root / "stderr.log"

    try:
        client.ensure_running()
    except RuntimeError as exc:
        message = str(exc)
        assert "exited with code 23" in message
        assert str(client.stderr_log_path) in message
        assert stderr_text in message
    else:
        raise AssertionError("Expected ensure_running() to report the service startup failure")
