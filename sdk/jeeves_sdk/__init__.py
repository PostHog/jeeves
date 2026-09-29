from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

import httpx2
import typesafe_sdk
from pydantic import BaseModel, ConfigDict
from typesafe_sdk import (
    Answer,
    AsyncModels,
    Choice,
    ChoiceAnswer,
    ChoiceModel,
    JSONContent,
    JSONValue,
    ListModelsResponse,
    ModelMetadata,
    Models,
    Noul,
    NoulAnswer,
    NoulCriteria,
    NoulModel,
    Question,
    QuestionModel,
    Questions,
    RetryPolicy,
    Score,
    ScoreAnswer,
    ScoreModel,
    TypeSafeAPIConnectionError,
    TypeSafeAPIError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeBadRequestError,
    TypeSafeError,
    TypeSafeInternalServerError,
    TypeSafeNotFoundError,
    TypeSafePermissionDeniedError,
    TypeSafeRateLimitError,
    TypeSafeUnprocessableEntityError,
    constants,
)

__version__ = "0.1.0"

BASE_URL_ENV = "JEEVES_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8009"
DEFAULT_TIMEOUT = 120.0
LOCAL_API_KEY = "local"


class Usage(typesafe_sdk.Usage):
    reasoning_tokens: int | None = None


class Reasoning(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    thought: bool
    closed: bool
    tokens: int
    text: str


class SystemOneResponse(typesafe_sdk.SystemOneResponse):
    usage: Usage
    reasoning: dict[str, Reasoning] | None = None
    latency_ms: float | None = None


def resolve(api_key: str | None, base_url: str | None, timeout: float | httpx2.Timeout | None, http_client: Any) -> dict[str, Any]:
    key = api_key if api_key is not None else os.environ.get(constants.API_KEY_ENV, "").strip() or LOCAL_API_KEY
    url = base_url if base_url is not None else os.environ.get(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL
    if timeout is None and http_client is None:
        timeout = DEFAULT_TIMEOUT
    return {"api_key": key, "base_url": url, "timeout": timeout}


def request_body(extra_body: Mapping[str, JSONValue | None] | None, think: bool | None, max_think: int | None,
                 nothink_threshold: float | None, return_reasoning: bool | None) -> Mapping[str, JSONValue | None] | None:
    options = {k: v for k, v in (("think", think), ("max_think", max_think), ("nothink_threshold", nothink_threshold),
                                  ("return_reasoning", return_reasoning)) if v is not None}
    if not options:
        return extra_body
    body = dict(extra_body or {})
    body["options"] = {**dict(body.get("options") or {}), **options}
    return body


class TypeSafeClient(typesafe_sdk.TypeSafeClient):
    def __init__(self, *, api_key: str | None = None, model: str | None = None, retry: RetryPolicy | None = None,
                 timeout: float | httpx2.Timeout | None = None, headers: Mapping[str, str] | None = None,
                 transport: httpx2.BaseTransport | None = None, http_client: httpx2.Client | None = None,
                 base_url: str | None = None) -> None:
        super().__init__(model=model, retry=retry, headers=headers, transport=transport, http_client=http_client,
                         **resolve(api_key, base_url, timeout, http_client))

    def system_one(self, state: JSONContent, questions: Mapping[str, Question], *, model: str | None = None,
                   retry: RetryPolicy | None = None, timeout: float | httpx2.Timeout | None = None,
                   extra_headers: Mapping[str, str] | None = None, extra_body: Mapping[str, JSONValue | None] | None = None,
                   response_model: type[BaseModel] | None = None, think: bool | None = None, max_think: int | None = None,
                   nothink_threshold: float | None = None, return_reasoning: bool | None = None) -> Any:
        return super().system_one(state, questions, model=model, retry=retry, timeout=timeout, extra_headers=extra_headers,
                                  extra_body=request_body(extra_body, think, max_think, nothink_threshold, return_reasoning),
                                  response_model=SystemOneResponse if response_model is None else response_model)


class AsyncTypeSafeClient(typesafe_sdk.AsyncTypeSafeClient):
    def __init__(self, *, api_key: str | None = None, model: str | None = None, retry: RetryPolicy | None = None,
                 timeout: float | httpx2.Timeout | None = None, headers: Mapping[str, str] | None = None,
                 transport: httpx2.AsyncBaseTransport | None = None, http_client: httpx2.AsyncClient | None = None,
                 base_url: str | None = None) -> None:
        super().__init__(model=model, retry=retry, headers=headers, transport=transport, http_client=http_client,
                         **resolve(api_key, base_url, timeout, http_client))

    async def system_one(self, state: JSONContent, questions: Mapping[str, Question], *, model: str | None = None,
                         retry: RetryPolicy | None = None, timeout: float | httpx2.Timeout | None = None,
                         extra_headers: Mapping[str, str] | None = None, extra_body: Mapping[str, JSONValue | None] | None = None,
                         response_model: type[BaseModel] | None = None, think: bool | None = None, max_think: int | None = None,
                         nothink_threshold: float | None = None, return_reasoning: bool | None = None) -> Any:
        return await super().system_one(state, questions, model=model, retry=retry, timeout=timeout, extra_headers=extra_headers,
                                        extra_body=request_body(extra_body, think, max_think, nothink_threshold, return_reasoning),
                                        response_model=SystemOneResponse if response_model is None else response_model)


JeevesClient = TypeSafeClient
AsyncJeevesClient = AsyncTypeSafeClient

__all__ = [
    "Answer", "AsyncJeevesClient", "AsyncModels", "AsyncTypeSafeClient", "Choice", "ChoiceAnswer", "ChoiceModel", "JSONContent",
    "JSONValue", "JeevesClient", "ListModelsResponse", "ModelMetadata", "Models", "Noul", "NoulAnswer", "NoulCriteria", "NoulModel",
    "Question", "QuestionModel", "Questions", "Reasoning", "RetryPolicy", "Score", "ScoreAnswer", "ScoreModel", "SystemOneResponse",
    "TypeSafeAPIConnectionError", "TypeSafeAPIError", "TypeSafeAPIResponseValidationError", "TypeSafeAPITimeoutError",
    "TypeSafeAuthenticationError", "TypeSafeBadRequestError", "TypeSafeClient", "TypeSafeError", "TypeSafeInternalServerError",
    "TypeSafeNotFoundError", "TypeSafePermissionDeniedError", "TypeSafeRateLimitError", "TypeSafeUnprocessableEntityError", "Usage",
    "constants",
]
