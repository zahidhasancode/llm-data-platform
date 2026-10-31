"""Server behaviour and the rules that keep a rolling update safe."""

from pathlib import Path

import pytest
import yaml

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from src.serving.app import create_app  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def client() -> TestClient:
    return TestClient(create_app(lambda context, question: f"SELECT '{question}';", "7"))


def test_sql_returns_the_answer_and_its_model_version() -> None:
    response = client().post("/v1/sql", json={"question": "Q", "context": "C"})
    assert response.json() == {"sql": "SELECT 'Q';", "model_version": "7"}


def test_empty_input_is_refused() -> None:
    assert client().post("/v1/sql", json={"question": "", "context": "C"}).status_code == 422


def test_health_reports_the_version() -> None:
    assert client().get("/health").json() == {"status": "ok", "model_version": "7"}


def test_a_server_that_is_not_ready_says_so() -> None:
    app = create_app(lambda c, q: "", "7")
    app.state.ready = False
    assert TestClient(app).get("/health").status_code == 503


def test_rolling_update_never_removes_capacity() -> None:
    deployment = next(yaml.safe_load_all((ROOT / "k8s" / "server.yaml").read_text()))
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["resources"]["limits"]["memory"]
