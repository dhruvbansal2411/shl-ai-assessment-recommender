"""FastAPI routes for the SHL recommender."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from app.chat import ChatService
from app.models import ChatRequest, ChatResponse, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def get_chat_service(request: Request) -> ChatService:
    """Resolve the app-scoped chat service."""

    service = getattr(request.app.state, "chat_service", None)
    if service is None:
        service = ChatService()
        request.app.state.chat_service = service
    return service


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness endpoint."""

    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Stateless catalog-grounded chat endpoint."""

    start = time.perf_counter()
    try:
        response = await service.handle(payload.messages)
        logger.info(
            "chat request messages=%s recommendations=%s latency_seconds=%.3f",
            len(payload.messages),
            len(response.recommendations),
            time.perf_counter() - start,
        )
        return response
    except RuntimeError as exc:
        logger.exception("chat runtime error")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("chat unexpected error")
        raise HTTPException(status_code=500, detail="Internal server error") from exc
