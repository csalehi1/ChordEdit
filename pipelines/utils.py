"""Shared helpers for dataset pipeline scripts."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any


def bracket_diff(source: str, target: str) -> tuple[str, str]:
    """Wrap differing word spans in [brackets] for PIE-Bench-style prompts."""

    def wrap(words: list[str]) -> str:
        span = " ".join(words)
        # Use a regular expression to capture three parts of the string: leading non-word characters,
        # the main content in the middle (matched as little as possible), and trailing non-word characters.
        # ^(\W*) matches any number of non-word characters at the start of the string.
        # (.*?) captures the smallest possible middle section of the string while allowing multiline text.
        # (\W*)$ matches any number of non-word characters at the end of the string.
        # re.DOTALL makes the dot (.) match newline characters as well, so the middle section can span lines.
        match = re.match(r"^(\W*)(.*?)(\W*)$", span, flags=re.DOTALL)
        if not match or not match.group(2):
            return span
        lead, core, trail = match.groups()
        return f"{lead}[{core}]{trail}"

    source_words, target_words = source.split(), target.split()
    matcher = difflib.SequenceMatcher(a=source_words, b=target_words, autojunk=False)
    source_out, target_out = [], []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            source_out.extend(source_words[i1:i2])
            target_out.extend(target_words[j1:j2])
        else:
            if i2 > i1:
                source_out.append(wrap(source_words[i1:i2]))
            if j2 > j1:
                target_out.append(wrap(target_words[j1:j2]))
    return " ".join(source_out), " ".join(target_out)


def save_image(image: Any, path: Path, jpeg_quality: int | None = None) -> None:
    """Save a PIL image; None = lossless native format, int = JPEG at that quality."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if jpeg_quality is None:
        image.save(path)
        return
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    image.save(path, format="JPEG", quality=jpeg_quality)

