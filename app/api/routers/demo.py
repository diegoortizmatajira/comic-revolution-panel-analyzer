"""Developer demo page: pick, upload or paste a page and see the detected panels overlaid."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.deps import ContainerDep  # noqa: TC001

router = APIRouter(tags=["demo"], include_in_schema=False)

_PAGE = Path(__file__).resolve().parent.parent.parent / "static" / "demo.html"
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _sample_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )


@router.get("/demo")
async def demo() -> FileResponse:
    return FileResponse(_PAGE, media_type="text/html")


class SampleOut(BaseModel):
    source: Literal["samples", "feedback", "fixtures"]
    name: str


def _folders(container: ContainerDep) -> dict[str, Path]:
    settings = container.settings
    folders = {"samples": settings.samples_dir}
    if settings.feedback_dir is not None:
        folders["feedback"] = settings.feedback_dir
    folders["fixtures"] = settings.fixtures_dir
    return folders


@router.get("/demo/samples")
async def list_samples(container: ContainerDep) -> list[SampleOut]:
    """Local pages (`samples`), stored feedback pages, then the committed test fixtures."""
    return [
        SampleOut(source=source, name=path.name)
        for source, folder in _folders(container).items()
        for path in _sample_files(folder)
    ]


@router.get("/demo/samples/{source}/{name}")
async def get_sample(source: str, name: str, container: ContainerDep) -> FileResponse:
    # Match against the listing instead of building a path from user input (no traversal).
    folder = _folders(container).get(source)
    if folder is not None:
        for path in _sample_files(folder):
            if path.name == name:
                return FileResponse(path)
    raise HTTPException(status_code=404, detail="Sample not found.")
