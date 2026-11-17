"""Exception hierarchy; every library error derives from :class:`FoldcraftError`."""

from __future__ import annotations


class FoldcraftError(Exception):
    """Base class for every exception raised by foldcraft."""


class StructureError(FoldcraftError):
    """An inconsistent hierarchy or a failed lookup."""


class ParseError(FoldcraftError):
    """A coordinate file that cannot be interpreted; carries the 1-based line number and text."""

    def __init__(self, message: str, line_number: int | None = None, line: str | None = None) -> None:
        self.message, self.line_number, self.line = message, line_number, line
        location = f" (line {line_number})" if line_number is not None else ""
        detail = f": {line.rstrip()!r}" if line is not None else ""
        super().__init__(f"{message}{location}{detail}")


class PDBParseError(ParseError):
    """Malformed PDB-format records."""


class MMCIFParseError(ParseError):
    """Malformed mmCIF / STAR content."""


class ComparisonError(FoldcraftError):
    """Coordinate sets that cannot be superposed or compared."""
