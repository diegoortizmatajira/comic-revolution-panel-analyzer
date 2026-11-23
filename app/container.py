"""Composition root: the only place concrete adapters are wired together."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.services.detection.registry import MlConfig
from app.services.feedback import FeedbackStore
from app.services.image.decoder import ImageLimits
from app.services.image.sources import UrlImageFetcher
from app.services.pipeline import PipelineFactory
from app.services.security import SlidingWindowRateLimiter, StaticApiKeyStore

if TYPE_CHECKING:
    from app.core.config import Settings
    from app.domain.ports import ApiKeyStore, ImageFetcher, RateLimiter
    from app.services.analyzer import PanelAnalyzer


@dataclass(frozen=True, slots=True)
class Container:
    settings: Settings
    analyzer: PanelAnalyzer  # the server defaults; a request with overrides gets pipeline.analyzer
    pipeline: PipelineFactory
    fetcher: ImageFetcher
    limits: ImageLimits
    api_keys: ApiKeyStore
    rate_limiter: RateLimiter
    feedback: FeedbackStore | None = None  # None: PANELS_FEEDBACK_DIR is not set


def ml_config(settings: Settings) -> MlConfig | None:
    if settings.ml_model is None:
        return None
    return MlConfig(
        model=settings.ml_model,
        input_size=settings.ml_input_size,
        confidence=settings.ml_confidence,
        nms_iou=settings.ml_nms_iou,
        output_format=settings.ml_output_format,
        panel_class=settings.ml_panel_class,
    )


def build_container(settings: Settings) -> Container:
    pipeline = PipelineFactory(settings, ml_config(settings))
    return Container(
        settings=settings,
        analyzer=pipeline.analyzer(),
        pipeline=pipeline,
        fetcher=UrlImageFetcher(
            max_bytes=settings.max_image_bytes,
            timeout_seconds=settings.url_fetch_timeout_seconds,
            max_redirects=settings.url_max_redirects,
            allow_private=settings.allow_private_urls,
        ),
        limits=ImageLimits(
            max_bytes=settings.max_image_bytes, max_pixels=settings.max_image_pixels
        ),
        api_keys=StaticApiKeyStore(settings.parsed_api_keys()),
        rate_limiter=SlidingWindowRateLimiter(settings.rate_limit_per_minute),
        feedback=(
            None
            if settings.feedback_dir is None
            else FeedbackStore(
                settings.feedback_dir,
                max_samples=settings.feedback_max_samples,
                max_total_bytes=settings.feedback_max_total_bytes,
            )
        ),
    )
