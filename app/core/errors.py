"""Domain exception hierarchy. Mapped to HTTP responses in one place (api/errors.py)."""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class InvalidImageError(AppError):
    code = "invalid_image"
    status_code = 400


class InvalidUrlError(AppError):
    code = "invalid_url"
    status_code = 400


class ImageTooLargeError(AppError):
    code = "image_too_large"
    status_code = 413


class UnsupportedImageFormatError(AppError):
    code = "unsupported_image_format"
    status_code = 415


class ImageFetchError(AppError):
    code = "image_fetch_failed"
    status_code = 502


class ImageFetchTimeoutError(AppError):
    code = "image_fetch_timeout"
    status_code = 504


class AnalysisTimeoutError(AppError):
    code = "analysis_timeout"
    status_code = 504


class UnauthorizedError(AppError):
    code = "unauthorized"
    status_code = 401


class RateLimitExceededError(AppError):
    code = "rate_limit_exceeded"
    status_code = 429


class InvalidOptionError(AppError):
    code = "invalid_option"
    status_code = 422


class FeedbackDisabledError(AppError):
    code = "feedback_disabled"
    status_code = 501


class FeedbackStorageFullError(AppError):
    code = "feedback_storage_full"
    status_code = 507


class FeedbackNotFoundError(AppError):
    code = "feedback_not_found"
    status_code = 404
