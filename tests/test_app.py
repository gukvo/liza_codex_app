from pathlib import Path

from liza_codex_app.app import create_file, hello


def test_hello():
    assert hello() == "Hello"


def test_create_file(tmp_path: Path):
    out_dir = tmp_path / "out"
    path = create_file("note.txt", "hi", out_dir)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "hi"
