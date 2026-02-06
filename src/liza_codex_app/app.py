from __future__ import annotations

import platform
import sys
from pathlib import Path


def hello() -> str:
    return "Hello"


def create_file(filename: str, text: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(text, encoding="utf-8")
    return path


def system_info() -> dict[str, str]:
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cwd": str(Path.cwd()),
    }


def _menu() -> None:
    print(hello())
    while True:
        print()
        print("Create file")
        print("System info")
        print("Exit")
        choice = input("Choose an option: ").strip().lower()

        if choice in {"create file", "1", "create"}:
            filename = input("File name: ").strip()
            text = input("Text: ")
            path = create_file(filename, text, Path.cwd() / "out")
            print(f"Saved to {path}")
        elif choice in {"system info", "2", "system"}:
            info = system_info()
            print(f"Python version: {info['python_version']}")
            print(f"Platform: {info['platform']}")
            print(f"CWD: {info['cwd']}")
        elif choice in {"exit", "3", "quit", "q"}:
            return
        else:
            print("Unknown option, try again.")


def main() -> None:
    _menu()
