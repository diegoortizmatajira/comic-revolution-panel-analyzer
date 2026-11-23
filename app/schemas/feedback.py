"""HTTP models for the feedback endpoint (corrected panel mappings)."""

from __future__ import annotations

import json
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core import options as opt  # noqa: TC001  (pydantic needs them at runtime)
from app.evaluation.labels import PanelLabel
from app.schemas.panels import ImageSource  # noqa: TC001

MIN_PANELS = 1
MAX_PANELS = 64
MIN_SIDE = 0.01  # each panel is at least 1% of the page in both dimensions
_EPSILON = 1e-6

Unit = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class FeedbackBox(BaseModel):
    """A panel box normalized to the page (0..1, origin top-left)."""

    model_config = ConfigDict(extra="forbid")

    x: Unit
    y: Unit
    width: Unit
    height: Unit

    @model_validator(mode="after")
    def _valid(self) -> Self:
        if self.width < MIN_SIDE or self.height < MIN_SIDE:
            msg = "a panel must be at least 1% of the page in both dimensions"
            raise ValueError(msg)
        if self.x + self.width > 1 + _EPSILON or self.y + self.height > 1 + _EPSILON:
            msg = "a panel must lie inside the page"
            raise ValueError(msg)
        return self

    def to_label(self) -> PanelLabel:
        return PanelLabel(x=self.x, y=self.y, width=self.width, height=self.height)


class DetectedPanelIn(BaseModel):
    """A panel as the analyzer returned it (`bbox_normalized`)."""

    model_config = ConfigDict(extra="ignore")

    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    width: float = Field(allow_inf_nan=False)
    height: float = Field(allow_inf_nan=False)
    order: int | None = None
    confidence: float | None = Field(default=None, allow_inf_nan=False)


class DetectedIn(BaseModel):
    """What the analyzer returned for the page, so a sample records what was wrong."""

    model_config = ConfigDict(extra="ignore")

    version: str = Field(max_length=200)
    method: str = Field(max_length=100)
    panels: list[DetectedPanelIn] = Field(max_length=256)
    sequence_confidence: float | None = Field(default=None, allow_inf_nan=False)


class FeedbackFields(BaseModel):
    """Everything except the image, shared by the JSON and the multipart forms."""

    model_config = ConfigDict(extra="forbid")

    reference: str = Field(
        min_length=1,
        max_length=200,
        description="Opaque client-chosen id of the sample (for example book and page). Sending "
        "the same reference again replaces the earlier sample.",
    )
    reading_direction: opt.ReadingDirectionName
    panels: list[FeedbackBox] = Field(
        min_length=MIN_PANELS,
        max_length=MAX_PANELS,
        description="The corrected panels, in reading order.",
    )
    detected: DetectedIn | None = None


class FeedbackRequest(FeedbackFields):
    image: ImageSource


def parse_json_field(raw: str, name: str) -> Any:  # noqa: ANN401
    """A multipart form field that carries JSON; a bad value is a validation error."""
    try:
        return json.loads(raw)
    except ValueError as exc:
        msg = f"'{name}' is not valid JSON."
        raise ValueError(msg) from exc


class FeedbackResponse(BaseModel):
    reference: str
    sample_id: str = Field(description="Stable file stem of the sample in the feedback folder.")
    replaced: bool = Field(description="True when a sample with this reference already existed.")
    unchanged: bool = Field(description="True when the stored sample was already identical.")
    image_sha256: str
    panels: int
