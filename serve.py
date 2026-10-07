"""Text-only System One server; currently supports the Clef backend."""

import asyncio
import sys
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import Any, Self

import torch
from fastapi import FastAPI, HTTPException
from huggingface_hub import snapshot_download
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DecisionPayload = dict[str, Any]
DecisionFunction = Callable[[DecisionPayload], DecisionPayload]


def load_clef_backend(
    model_id: str, revision: str, max_length: int
) -> DecisionFunction:
    path = snapshot_download(model_id, revision=revision)
    sys.path.insert(0, path)
    # Executes the reference helper supplied by the selected model revision.
    from joint_schema_model import load_release_model, systemone

    model, processor = load_release_model(path, device="cuda")
    return partial(systemone, model, processor, max_length=max_length)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SYSTEMONE_")

    backend: str = "clef"
    model: str = Field(default="Cloudflare/clef-flash", min_length=1)
    revision: str = Field(
        default="fde727a287004204b7518dcc983fe64379776712", min_length=1
    )
    max_length: int = Field(default=4096, gt=0)

    @model_validator(mode="after")
    def validate_backend(self) -> Self:
        # Update this restriction and lifespan dispatch when adding another backend.
        if self.backend != "clef":
            raise ValueError(
                f"Unsupported SYSTEMONE_BACKEND {self.backend!r}; only 'clef' is supported"
            )
        return self

    @model_validator(mode="after")
    def validate_model_revision(self) -> Self:
        if (
            self.model != type(self).model_fields["model"].default
            and "revision" not in self.model_fields_set
        ):
            raise ValueError(
                "Set SYSTEMONE_REVISION when selecting another SYSTEMONE_MODEL"
            )
        return self


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = Settings()
    app.state.decide = load_clef_backend(
        settings.model, settings.revision, settings.max_length
    )
    app.state.settings = settings
    app.state.lock = asyncio.Lock()
    print(
        f"Loaded {settings.model}@{settings.revision}; GPU: {torch.cuda.get_device_name()}",
        flush=True,
    )
    yield


app = FastAPI(lifespan=lifespan)


class DecisionRequest(BaseModel):
    model: str
    state: str = Field(max_length=20_000)
    questions: dict[str, dict[str, Any]] = Field(min_length=1, max_length=32)


class Health(BaseModel):
    backend: str
    model: str
    revision: str
    max_length: int
    gpu: str


@app.get("/health")
def health() -> Health:
    settings: Settings = app.state.settings
    return Health(
        backend=settings.backend,
        model=settings.model,
        revision=settings.revision,
        max_length=settings.max_length,
        gpu=torch.cuda.get_device_name(),
    )


@app.post("/v1/systemone")
async def decide(body: DecisionRequest) -> DecisionPayload:
    settings: Settings = app.state.settings
    decision_function: DecisionFunction = app.state.decide
    if body.model != settings.model:
        raise HTTPException(400, f"This instance serves {settings.model}")
    async with app.state.lock:
        try:
            return await asyncio.to_thread(decision_function, body.model_dump())
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise HTTPException(503, "CUDA out of memory") from exc
