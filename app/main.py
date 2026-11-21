"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.middleware import RequestContextMiddleware
from app.api.routers import demo, health, panels
from app.container import Container, build_container
from app.core.config import Settings
from app.core.logging import configure_logging


def create_app(settings: Settings | None = None, container: Container | None = None) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level, json=settings.log_json)

    app = FastAPI(
        title="Comic Panel Analyzer",
        version="1.0.0",
        description="Detects comic page panels, their reading order and a reliability score.",
    )
    app.state.container = container or build_container(settings)
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(panels.router)
    if settings.enable_demo_page:
        app.include_router(demo.router)
    return app


def app_factory() -> FastAPI:
    """Entry point for `uvicorn --factory app.main:app_factory`."""
    return create_app()
