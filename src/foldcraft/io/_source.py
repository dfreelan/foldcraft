"""Uniform handling of path / handle / compressed inputs for the readers."""

from __future__ import annotations

import gzip
import io
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TextIO


def source_stem(path: str | os.PathLike[str]) -> str:
    """File name without directory, ``.gz`` and format suffix."""
    name = Path(path).name.removesuffix(".gz")
    return name.rsplit(".", 1)[0] if "." in name else name


@contextmanager
def open_text(source: str | os.PathLike[str] | TextIO) -> Iterator[tuple[TextIO, str]]:
    """Yield ``(handle, stem)`` for a path, a gzip path, or an open text handle.

    Handles passed in by the caller are not closed.
    """
    if hasattr(source, "read"):
        name = getattr(source, "name", "")
        yield source, source_stem(name) if isinstance(name, str) and name else ""  # type: ignore[misc]
        return
    path = Path(source)  # type: ignore[arg-type]
    with open(path, "rb") as probe:
        magic = probe.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as raw, io.TextIOWrapper(raw, encoding="utf-8", errors="replace") as fh:
            yield fh, source_stem(path)
    else:
        with open(path, encoding="utf-8", errors="replace") as fh:
            yield fh, source_stem(path)


def sniff_format(path: str | os.PathLike[str]) -> str:
    """Guess ``"pdb"`` or ``"mmcif"`` from the file extension, then the content."""
    name = Path(path).name.lower().removesuffix(".gz")
    if name.endswith((".cif", ".mmcif")):
        return "mmcif"
    if name.endswith((".pdb", ".ent", ".brk")):
        return "pdb"
    with open_text(path) as (fh, _):
        first = next((s for s in (line.strip() for line in fh) if s and not s.startswith("#")), "")
    return "mmcif" if first.lower().startswith(("data_", "loop_", "_")) else "pdb"


def write_text(text: str, target: str | os.PathLike[str] | TextIO) -> None:
    """Write *text* to a path or to a writable text handle (which is left open)."""
    if hasattr(target, "write"):
        target.write(text)  # type: ignore[union-attr]
    else:
        Path(target).write_text(text, encoding="utf-8")  # type: ignore[arg-type]
