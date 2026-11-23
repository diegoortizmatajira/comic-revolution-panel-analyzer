from pathlib import Path

import pytest
from app.evaluation.import_feedback import import_samples, main
from app.evaluation.labels import discover

from tests.conftest import make_page

LABELS = (
    '{"version": 1, "reading_direction": "ltr", "verified": true, "tags": ["feedback"],'
    ' "notes": "", "panels": [{"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}]}'
)


def sample(folder: Path, stem: str, labels: str = LABELS) -> None:
    folder.mkdir(exist_ok=True)
    (folder / f"{stem}.png").write_bytes(make_page([(40, 40, 300, 300)]))
    (folder / f"{stem}.labels.json").write_text(labels)
    (folder / f"{stem}.feedback.json").write_text("{}")


def test_copies_and_the_harness_reads_them(tmp_path: Path) -> None:
    src, dest = tmp_path / "fb", tmp_path / "corpus"
    sample(src, "abc")
    result = import_samples(src, dest)
    assert result["added"] == ["abc"]
    assert not (dest / "feedback-abc.feedback.json").exists()
    [page] = discover([dest])
    assert page.id == "corpus/feedback-abc"
    assert "feedback" in page.tags


def test_second_run_changes_nothing_and_differences_need_overwrite(tmp_path: Path) -> None:
    src, dest = tmp_path / "fb", tmp_path / "corpus"
    sample(src, "abc")
    import_samples(src, dest)
    assert import_samples(src, dest)["unchanged"] == ["abc"]

    sample(src, "abc", LABELS.replace("0.5", "0.4"))
    assert import_samples(src, dest)["skipped"] == ["abc"]
    assert import_samples(src, dest, dry_run=True, overwrite=True)["updated"] == ["abc"]
    assert "0.5" in (dest / "feedback-abc.labels.json").read_text()
    assert import_samples(src, dest, overwrite=True)["updated"] == ["abc"]
    assert "0.4" in (dest / "feedback-abc.labels.json").read_text()


def test_incomplete_samples_are_ignored(tmp_path: Path) -> None:
    src = tmp_path / "fb"
    sample(src, "abc")
    (src / "abc.labels.json").unlink()
    assert not any(import_samples(src, tmp_path / "c").values())


def test_missing_source_folder(tmp_path: Path) -> None:
    assert main(["--source", str(tmp_path / "none"), "--dest", str(tmp_path)]) == 2


def test_main_reports_what_it_did(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    src, dest = tmp_path / "fb", tmp_path / "corpus"
    sample(src, "abc")
    args = ["--source", str(src), "--dest", str(dest)]
    assert main([*args, "--dry-run"]) == 0
    assert "1 would be added" in capsys.readouterr().out
    assert not dest.exists()
    assert main(args) == 0
    assert "1 added: abc" in capsys.readouterr().out
    sample(src, "abc", LABELS.replace("0.5", "0.4"))
    assert main(args) == 0
    assert "--overwrite" in capsys.readouterr().out
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--source", str(empty), "--dest", str(dest)]) == 0
    assert "No feedback samples" in capsys.readouterr().out
