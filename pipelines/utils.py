"""Shared helpers for dataset pipeline scripts."""

from __future__ import annotations

import difflib
import re


def bracket_diff(source: str, target: str) -> tuple[str, str]:
    """Wrap differing word spans in [brackets] for PIE-Bench-style prompts."""

    def wrap(words: list[str]) -> str:
        span = " ".join(words)
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
