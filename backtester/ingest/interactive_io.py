"""Shared interactive I/O helpers for the notebook-driven pipeline (Colab or local).

Every prompt here is a plain `input()` call, which renders identically as an inline text
box in a Jupyter/Colab code cell and as a normal prompt in a terminal -- no notebook-
specific widget code needed. File uploads use `google.colab.files.upload()` when running
in Colab (detected via `in_colab()`), and fall back to asking for a local path otherwise,
so the same calling code works in both environments without a branch at every call site.
"""
from __future__ import annotations

import os


def in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def prompt_text(message: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default not in (None, "") else ""
    val = input(f"{message}{suffix}: ").strip()
    return val if val else (default or "")


def prompt_float(message: str, default: float) -> float:
    raw = prompt_text(message, default=str(default))
    try:
        return float(raw)
    except ValueError:
        print(f"  Could not parse '{raw}' as a number -- using default {default}.")
        return default


def prompt_int(message: str, default: int) -> int:
    raw = prompt_text(message, default=str(default))
    try:
        return int(raw)
    except ValueError:
        print(f"  Could not parse '{raw}' as an integer -- using default {default}.")
        return default


def prompt_yesno(message: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    val = input(f"{message}{suffix}: ").strip().lower()
    if not val:
        return default
    return val.startswith("y")


def prompt_choice(message: str, options: list[str], allow_other: bool = False) -> str:
    print(message)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    if allow_other:
        print(f"  {len(options) + 1}. (other -- type your own)")
    while True:
        raw = input("Choice: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(options):
                return options[idx - 1]
            if allow_other and idx == len(options) + 1:
                return prompt_text("Enter value")
        print("  Invalid choice, try again.")


def prompt_multiselect(message: str, options: list[str]) -> list[str]:
    """Returns the SUBSET of `options` the user picked, preserving `options` order."""
    if not options:
        return []
    print(message + "  (comma-separated numbers, or 'all', or blank for none)")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    raw = input("Selection: ").strip().lower()
    if raw == "all":
        return list(options)
    if not raw:
        return []
    picks = []
    for tok in raw.split(","):
        tok = tok.strip()
        if tok.isdigit():
            idx = int(tok)
            if 1 <= idx <= len(options):
                picks.append(options[idx - 1])
    return picks


def upload_file(prompt_message: str, save_dir: str) -> str:
    """Returns a local filesystem path to an uploaded (Colab) or manually-specified
    (local Jupyter/terminal) file. Empty string means the user declined."""
    os.makedirs(save_dir, exist_ok=True)
    if in_colab():
        from google.colab import files

        print(prompt_message)
        uploaded = files.upload()
        paths = []
        for name, content in uploaded.items():
            path = os.path.join(save_dir, name)
            with open(path, "wb") as f:
                f.write(content)
            paths.append(path)
        return paths[0] if paths else ""
    path = prompt_text(f"{prompt_message} (enter a local file path, or leave blank to skip)")
    if path and not os.path.exists(path):
        print(f"  Warning: '{path}' does not exist on disk.")
    return path
