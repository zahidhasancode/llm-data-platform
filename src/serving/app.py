"""Model server: answer SQL questions, and report whether the model is ready.

`create_app` takes the generate function, so the HTTP behaviour is tested without a model.
"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class SqlRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    context: str = Field(min_length=1, max_length=20000)


def create_app(generate: Callable[[str, str], str], model_version: str) -> FastAPI:
    app = FastAPI(title="text2sql server")
    app.state.ready = True

    @app.get("/health")
    def health() -> dict:
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="not ready")
        return {"status": "ok", "model_version": model_version}

    @app.post("/v1/sql")
    def sql(request: SqlRequest) -> dict:
        return {"sql": generate(request.context, request.question), "model_version": model_version}

    return app
