from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from app.api.deps import AuthDep, ContainerDep  # noqa: TC001
from app.core.deadline import time_limit
from app.core.errors import FeedbackDisabledError, ImageTooLargeError
from app.core.options import OVERRIDABLE
from app.domain.models import ReadingDirection
from app.evaluation.labels import PageLabels
from app.schemas.feedback import (
    FeedbackFields,
    FeedbackRequest,
    FeedbackResponse,
    parse_json_field,
)
from app.schemas.panels import (
    AnalyzeOptions,
    AnalyzeRequest,
    AnalyzeResponse,
    ErrorBody,
    OptionsResponse,
    UploadOptions,
    describe_options,
)
from app.services.feedback import FEEDBACK_TAG
from app.services.image.decoder import decode_image
from app.services.image.sources import decode_base64_image

router = APIRouter(prefix="/v1/panels", tags=["panels"])

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorBody} for code in (400, 401, 404, 413, 415, 422, 429, 501, 502, 504, 507)
}


async def _analyze(
    container: ContainerDep, data: bytes, options: AnalyzeOptions
) -> AnalyzeResponse:
    overrides = options.overrides()
    direction = ReadingDirection(
        overrides.pop("reading_direction", container.settings.reading_direction)
    )

    def work() -> AnalyzeResponse:
        with time_limit(container.settings.analysis_timeout_seconds):
            image = decode_image(data, container.limits)
            analyzer = container.pipeline.analyzer(overrides) if overrides else container.analyzer
            return AnalyzeResponse.from_result(analyzer.analyze(image, direction))

    # CPU-bound: keep the event loop free, and run only a few at a time so that a burst of
    # requests (or a few hard pages) cannot starve every other one of the CPU.
    async with container.analysis_slots:
        return await run_in_threadpool(work)


@router.post("/analyze", response_model=AnalyzeResponse, responses=_ERRORS)
async def analyze(body: AnalyzeRequest, container: ContainerDep, _key: AuthDep) -> AnalyzeResponse:
    """Analyze a page supplied as a URL or a base64-encoded image."""
    if body.image.url is not None:
        data = await container.fetcher.fetch(body.image.url)
    else:
        assert body.image.base64 is not None  # noqa: S101  (guaranteed by schema validator)
        data = decode_base64_image(body.image.base64, container.limits.max_bytes)
    return await _analyze(container, data, body.options)


@router.post("/analyze/upload", response_model=AnalyzeResponse, responses=_ERRORS)
async def analyze_upload(
    container: ContainerDep,
    _key: AuthDep,
    file: Annotated[UploadFile, File(description="Page image (JPEG, PNG or WebP).")],
    options: Annotated[UploadOptions, Form()],
) -> AnalyzeResponse:
    """Analyze a page supplied as a multipart file upload.

    The processing options are form fields with the same names as in the JSON body.
    """
    limit = container.limits.max_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        msg = "Image exceeds the maximum allowed size."
        raise ImageTooLargeError(msg, {"max_bytes": limit})
    return await _analyze(container, data, options)


@router.get("/options", response_model=OptionsResponse, responses=_ERRORS)
async def options(container: ContainerDep, _key: AuthDep) -> OptionsResponse:
    """The processing options a request may override, with this server's default for each."""
    defaults = {name: getattr(container.settings, name) for name in OVERRIDABLE}
    return OptionsResponse(options=describe_options(defaults))


async def _store_feedback(
    container: ContainerDep, data: bytes, fields: FeedbackFields, response: Response
) -> FeedbackResponse:
    store = container.feedback
    assert store is not None  # noqa: S101  (checked by the caller before the image is read)
    labels = PageLabels(
        reading_direction=ReadingDirection(fields.reading_direction),
        verified=True,  # a person corrected it
        tags=[FEEDBACK_TAG],
        panels=[p.to_label() for p in fields.panels],
    )
    detected = fields.detected.model_dump(mode="json") if fields.detected else None

    def work() -> FeedbackResponse:
        decode_image(data, container.limits)  # same size, dimension and format checks as analyze
        outcome = store.save(fields.reference, data, labels, detected)
        return FeedbackResponse(
            reference=fields.reference,
            sample_id=outcome.sample_id,
            replaced=outcome.replaced,
            unchanged=outcome.unchanged,
            image_sha256=outcome.image_sha256,
            panels=len(fields.panels),
        )

    result = await run_in_threadpool(work)
    response.status_code = 200 if result.replaced else 201
    return result


def _require_feedback(container: ContainerDep) -> None:
    if container.feedback is None:
        msg = "Feedback is not enabled on this server (PANELS_FEEDBACK_DIR is not set)."
        raise FeedbackDisabledError(msg)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={**_ERRORS, 201: {"model": FeedbackResponse, "description": "Sample created."}},
)
async def feedback(
    body: FeedbackRequest, container: ContainerDep, _key: AuthDep, response: Response
) -> FeedbackResponse:
    """Store a corrected panel mapping with its page image as a labelled sample.

    `200` with `replaced: true` when the reference already had a sample, `201` for a new one.
    `501` when the server has no feedback folder configured.
    """
    _require_feedback(container)
    if body.image.url is not None:
        data = await container.fetcher.fetch(body.image.url)
    else:
        assert body.image.base64 is not None  # noqa: S101  (guaranteed by schema validator)
        data = decode_base64_image(body.image.base64, container.limits.max_bytes)
    return await _store_feedback(container, data, body, response)


@router.post(
    "/feedback/upload",
    response_model=FeedbackResponse,
    responses={**_ERRORS, 201: {"model": FeedbackResponse, "description": "Sample created."}},
)
async def feedback_upload(  # noqa: PLR0913, PLR0917
    container: ContainerDep,
    _key: AuthDep,
    response: Response,
    file: Annotated[UploadFile, File(description="Page image (JPEG, PNG or WebP).")],
    reference: Annotated[str, Form()],
    reading_direction: Annotated[str, Form()],
    panels: Annotated[str, Form(description="JSON list of normalized boxes, in reading order.")],
    detected: Annotated[str | None, Form(description="JSON object, as in the JSON body.")] = None,
) -> FeedbackResponse:
    """The same as `POST /v1/panels/feedback`, with the image as a multipart file upload.

    `panels` and `detected` are form fields carrying JSON.
    """
    _require_feedback(container)
    try:
        fields = FeedbackFields.model_validate(
            {
                "reference": reference,
                "reading_direction": reading_direction,
                "panels": parse_json_field(panels, "panels"),
                "detected": None if detected is None else parse_json_field(detected, "detected"),
            }
        )
    except ValueError as exc:  # ValidationError is a ValueError; a bad JSON field is one too
        if isinstance(exc, ValidationError):
            raise RequestValidationError(exc.errors()) from exc
        raise RequestValidationError(
            [{"type": "json_invalid", "loc": ("form",), "msg": str(exc)}]
        ) from exc
    limit = container.limits.max_bytes
    data = await file.read(limit + 1)
    if len(data) > limit:
        msg = "Image exceeds the maximum allowed size."
        raise ImageTooLargeError(msg, {"max_bytes": limit})
    return await _store_feedback(container, data, fields, response)


@router.delete("/feedback/{reference:path}", status_code=204, responses=_ERRORS)
async def delete_feedback(reference: str, container: ContainerDep, _key: AuthDep) -> Response:
    """Withdraw a sample that was sent by mistake. `404` when the reference has none."""
    _require_feedback(container)
    assert container.feedback is not None  # noqa: S101
    await run_in_threadpool(container.feedback.delete, reference)
    return Response(status_code=204)
