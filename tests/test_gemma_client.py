import tempfile
import uuid
from pathlib import Path

from epc_smart_search.app_paths import GemmaLaunchSpec
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

    monkeypatch.setattr(
        gemma_client,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="external_python",
            service_path=fake_python,
            model_dir=None,
            available=True,
            tier="ai_min",
            reason=None,
        ),
    )
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


def test_get_availability_reports_disabled_ai(monkeypatch) -> None:
    monkeypatch.setattr(
        gemma_client,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="disabled",
            service_path=None,
            model_dir=None,
            available=False,
            tier="lite",
            reason="AI mode is disabled for this app instance.",
        ),
    )

    client = gemma_client.GemmaServiceClient()
    availability = client.get_availability()

    assert availability.available is False
    assert availability.mode == "disabled"
    assert "disabled" in availability.message.lower()


def test_get_availability_reports_ai_min_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        gemma_client,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="bundled_service",
            service_path=Path("ai_runtime/gemma_service.exe"),
            model_dir=Path("models/gemma_min"),
            available=True,
            tier="ai_min",
            reason=None,
        ),
    )

    client = gemma_client.GemmaServiceClient()
    availability = client.get_availability()

    assert availability.available is True
    assert availability.message == "AI-Min mode is available on this machine."


def test_get_availability_reports_ai_high_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        gemma_client,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="bundled_service",
            service_path=Path("ai_runtime/gemma_service.exe"),
            model_dir=Path("models/gemma_high"),
            available=True,
            tier="ai_high",
            reason=None,
        ),
    )

    client = gemma_client.GemmaServiceClient()
    availability = client.get_availability()

    assert availability.available is True
    assert availability.message == "AI-High mode is available on this machine."


def test_ensure_running_uses_selected_model_dir_in_launch_env(monkeypatch) -> None:
    temp_root = Path(tempfile.gettempdir()) / "epc_smart_search_tests" / f"gemma_client_env_{uuid.uuid4().hex[:8]}"
    temp_root.mkdir(parents=True, exist_ok=True)
    fake_python = temp_root / "python.exe"
    model_dir = temp_root / "models" / "gemma_min"
    model_dir.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    class FakeProcess:
        def __init__(self, *args, **kwargs) -> None:
            captured["env"] = kwargs["env"]
            self.returncode = None

        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None) -> int:
            return 0

        def kill(self) -> None:
            return None

    health_checks = {"count": 0}

    def fake_health(*args, **kwargs):
        health_checks["count"] += 1
        if health_checks["count"] == 1:
            raise RuntimeError("down")

        class FakeResponse:
            ok = True

            def json(self):
                return {"status": "ok"}

        return FakeResponse()

    monkeypatch.setattr(
        gemma_client,
        "resolve_gemma_launch_spec",
        lambda: GemmaLaunchSpec(
            mode="external_python",
            service_path=fake_python,
            model_dir=model_dir,
            available=True,
            tier="ai_min",
            reason=None,
        ),
    )
    monkeypatch.setattr(gemma_client.requests, "get", fake_health)
    monkeypatch.setattr(gemma_client.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(gemma_client.time, "sleep", lambda seconds: None)

    client = gemma_client.GemmaServiceClient()
    client.stdout_log_path = temp_root / "stdout.log"
    client.stderr_log_path = temp_root / "stderr.log"

    client.ensure_running()

    assert captured["env"][gemma_client.MODEL_DIR_OVERRIDE_ENV_VAR] == str(model_dir)
