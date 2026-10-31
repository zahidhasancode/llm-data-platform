"""Model server: answer SQL questions, and report whether the model is ready.

`create_app` takes the generate function, so the HTTP behaviour is tested without a model.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.finetune.scoring import is_valid


log = logging.getLogger(__name__)


class SqlRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    context: str = Field(min_length=1, max_length=20000)


def _event(request: SqlRequest, sql: str, model_version: str) -> dict:
    """One event in the evaluation platform's `llm.requests` format."""
    return {
        "event_id": str(uuid.uuid4()),
        "prompt_name": "text2sql",
        "prompt_version": "1",
        "model_name": "text2sql-qlora",
        "model_version": model_version,
        "application_id": "sql-server",
        "input_text": request.question,
        "response": {
            "status": "success" if sql else "error",
            "output_text": sql or None,
            "metadata": {"valid_sql": is_valid(request.context, sql)},
        },
    }


def create_app(
    generate: Callable[[str, str], str],
    model_version: str,
    publish: Callable[[dict], None] | None = None,
) -> FastAPI:
    app = FastAPI(title="text2sql server")
    app.state.ready = True

    @app.get("/health")
    def health() -> dict:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ok", "model_version": model_version}

    @app.post("/v1/sql")
    def sql(request: SqlRequest) -> dict:
        answer = generate(request.context, request.question)
        if publish is not None:
            try:
                publish(_event(request, answer, model_version))
            except Exception:
                log.exception("Event not published")  # the caller still gets the answer
        return {"sql": answer, "model_version": model_version}

    return app
