"""FastAPI application factory and ASGI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api import router
from app.chat import ChatService
from app.utils import configure_logging, get_settings

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared application services."""

    settings = get_settings()
    logger.info("starting %s environment=%s", settings.app_name, settings.environment)
    app.state.chat_service = ChatService()
    yield
    logger.info("shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()

