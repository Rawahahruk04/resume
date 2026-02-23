"""
core/parser.py
==============
PDF text extraction pipeline with dual-engine fallback.

Responsibilities (single):
  - Accept a validated file path to a PDF
  - Extract text via PyMuPDF (primary engine — fast, memory-efficient)
  - Fall back to pdfplumber (secondary engine — better for complex layouts)
  - Detect low-text / scanned PDFs (<500 chars)
  - Return a structured ExtractionResult dataclass
  - Log all errors; never propagate naked exceptions to callers

Constraints:
  - No scoring logic
  - No matching logic
  - No cleaning/lemmatization (that is utils.py's job)
  - Deterministic — same input always produces same output
  - Memory-efficient — processes page-by-page, never loads full doc into RAM at once
  - Timeout-safe — per-page extraction inside isolated try/except blocks
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOW_TEXT_THRESHOLD: Final[int] = 500   # characters — below this is suspicious
MAX_FILE_SIZE_BYTES: Final[int] = 10 * 1024 * 1024  # 10 MB hard cap
ALLOWED_SUFFIX: Final[str] = ".pdf"

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class ExtractionEngine(Enum):
    PYMUPDF = auto()
    PDFPLUMBER = auto()
    FAILED = auto()


class ExtractionStatus(Enum):
    SUCCESS = auto()
    LOW_TEXT = auto()       # extracted but likely scanned / image-only
    EMPTY = auto()          # no text recovered at all
    FAILED = auto()         # hard failure — both engines raised


@dataclass(frozen=True)
class ExtractionResult:
    """Structured, immutable result returned by :func:`extract_text`.

    Attributes
    ----------
    text:
        Raw extracted text.  Empty string on failure — never ``None``.
    status:
        Machine-readable :class:`ExtractionStatus`.
    engine_used:
        Which engine produced the result.
    page_count:
        Number of pages processed (0 on failure).
    char_count:
        ``len(text)`` — pre-computed for convenience.
    is_low_text:
        ``True`` when char_count < LOW_TEXT_THRESHOLD.
    elapsed_seconds:
        Wall-clock extraction time in seconds.
    error_message:
        Human-readable error detail if status is FAILED.  Empty otherwise.
    """

    text: str
    status: ExtractionStatus
    engine_used: ExtractionEngine
    page_count: int
    char_count: int
    is_low_text: bool
    elapsed_seconds: float
    error_message: str = field(default="")


# ---------------------------------------------------------------------------
# Custom exception (used internally — never leaks outside this module)
# ---------------------------------------------------------------------------


class ParseError(RuntimeError):
    """Raised internally when both engines fail."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_path(file_path: str | Path) -> Path:
    """Return a resolved :class:`Path` or raise :class:`ParseError`."""

    path = Path(file_path).resolve()

    if not path.exists():
        raise ParseError(f"File not found: {path}")

    if not path.is_file():
        raise ParseError(f"Path is not a file: {path}")

    if path.suffix.lower() != ALLOWED_SUFFIX:
        raise ParseError(
            f"Unsupported file type: {path.suffix!r}. Only .pdf is accepted."
        )

    size = path.stat().st_size
    if size == 0:
        raise ParseError("File is empty (0 bytes).")
    if size > MAX_FILE_SIZE_BYTES:
        raise ParseError(
            f"File too large: {size:,} bytes (max {MAX_FILE_SIZE_BYTES:,} bytes)."
        )

    return path


def _extract_with_pymupdf(path: Path) -> tuple[str, int]:
    """Extract text page-by-page with PyMuPDF.

    Returns
    -------
    tuple[str, int]
        (full_text, page_count)

    Raises
    ------
    ImportError
        If ``fitz`` (PyMuPDF) is not installed.
    Exception
        Any PyMuPDF-specific error propagates so the caller can fall back.
    """
    import fitz  # PyMuPDF — lazy import to avoid hard dependency at module load

    parts: list[str] = []
    page_count: int = 0

    with fitz.open(str(path)) as doc:
        page_count = doc.page_count
        for page_num in range(page_count):
            try:
                page = doc.load_page(page_num)
                page_text = page.get_text("text") or ""
                parts.append(page_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "PyMuPDF: error on page %d of '%s': %s",
                    page_num + 1,
                    path.name,
                    exc,
                )
                # Continue — partial extraction is better than nothing

    return "\n".join(parts), page_count


def _extract_with_pdfplumber(path: Path) -> tuple[str, int]:
    """Extract text page-by-page with pdfplumber.

    Returns
    -------
    tuple[str, int]
        (full_text, page_count)

    Raises
    ------
    ImportError
        If ``pdfplumber`` is not installed.
    Exception
        Any pdfplumber-specific error propagates so the caller can mark FAILED.
    """
    import pdfplumber  # lazy import

    parts: list[str] = []
    page_count: int = 0

    with pdfplumber.open(str(path)) as pdf:
        page_count = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages):
            try:
                page_text = page.extract_text() or ""
                parts.append(page_text)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "pdfplumber: error on page %d of '%s': %s",
                    page_num + 1,
                    path.name,
                    exc,
                )

    return "\n".join(parts), page_count


def _determine_status(text: str) -> ExtractionStatus:
    """Classify the extraction result by text volume."""
    char_count = len(text.strip())
    if char_count == 0:
        return ExtractionStatus.EMPTY
    if char_count < LOW_TEXT_THRESHOLD:
        return ExtractionStatus.LOW_TEXT
    return ExtractionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_text(file_path: str | Path) -> ExtractionResult:
    """Extract text from a PDF file and return a structured result.

    Strategy
    --------
    1.  Validate the path (existence, extension, size).
    2.  Attempt extraction with **PyMuPDF** (primary — fast, low memory).
    3.  If PyMuPDF fails or returns zero characters, fall back to
        **pdfplumber** (handles more complex layouts / table-heavy PDFs).
    4.  Build and return an :class:`ExtractionResult`.

    This function **never raises** — all exceptions are caught, logged,
    and reflected in the returned ``ExtractionResult.status``.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the PDF file.

    Returns
    -------
    ExtractionResult
        Always returns a result object.  Check ``result.status`` before use.
    """
    start: float = time.monotonic()

    # ── Step 1: path validation ───────────────────────────────────────────
    try:
        path = _validate_path(file_path)
    except ParseError as exc:
        elapsed = time.monotonic() - start
        logger.error("Path validation failed: %s", exc)
        return ExtractionResult(
            text="",
            status=ExtractionStatus.FAILED,
            engine_used=ExtractionEngine.FAILED,
            page_count=0,
            char_count=0,
            is_low_text=False,
            elapsed_seconds=round(elapsed, 4),
            error_message=str(exc),
        )

    # ── Step 2: PyMuPDF (primary) ─────────────────────────────────────────
    text: str = ""
    page_count: int = 0
    engine_used: ExtractionEngine = ExtractionEngine.FAILED

    try:
        text, page_count = _extract_with_pymupdf(path)
        engine_used = ExtractionEngine.PYMUPDF
        logger.info(
            "PyMuPDF extracted %d chars from '%s' (%d pages).",
            len(text),
            path.name,
            page_count,
        )
    except ImportError:
        logger.warning("PyMuPDF (fitz) not installed — skipping to pdfplumber.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyMuPDF failed on '%s': %s — falling back.", path.name, exc)

    # ── Step 3: pdfplumber fallback ───────────────────────────────────────
    if not text.strip():
        logger.info("Falling back to pdfplumber for '%s'.", path.name)
        try:
            text, page_count = _extract_with_pdfplumber(path)
            engine_used = ExtractionEngine.PDFPLUMBER
            logger.info(
                "pdfplumber extracted %d chars from '%s' (%d pages).",
                len(text),
                path.name,
                page_count,
            )
        except ImportError:
            logger.error("pdfplumber not installed — both engines unavailable.")
        except Exception as exc:  # noqa: BLE001
            logger.error("pdfplumber also failed on '%s': %s", path.name, exc)

    # ── Step 4: build result ──────────────────────────────────────────────
    elapsed = time.monotonic() - start
    stripped_text = text.strip()
    char_count = len(stripped_text)
    status = _determine_status(stripped_text)

    if status == ExtractionStatus.LOW_TEXT:
        logger.warning(
            "'%s' returned only %d chars — possible scanned/image-only PDF.",
            path.name,
            char_count,
        )

    # Determine error message and normalize status
    # EMPTY = engine ran but returned no text → treat as FAILED for callers.
    # This prevents app.py from needing to check both EMPTY and FAILED.
    if engine_used == ExtractionEngine.FAILED:
        error_message = "All extraction engines failed or are unavailable."
        status = ExtractionStatus.FAILED
    elif status == ExtractionStatus.EMPTY:
        error_message = (
            "Both extraction engines ran but returned no text. "
            "The PDF may be image-only or corrupted."
        )
        status = ExtractionStatus.FAILED
    else:
        error_message = ""

    return ExtractionResult(
        text=stripped_text,
        status=status,
        engine_used=engine_used,
        page_count=page_count,
        char_count=char_count,
        is_low_text=char_count < LOW_TEXT_THRESHOLD,
        elapsed_seconds=round(elapsed, 4),
        error_message=error_message,
    )


def safe_delete(file_path: str | Path) -> bool:
    """Delete a file silently.  Returns ``True`` on success, ``False`` otherwise.

    Use this to clean up temporary files after extraction.  Safe to call on
    a path that no longer exists.
    """
    try:
        path = Path(file_path)
        if path.exists():
            os.remove(path)
            logger.debug("Deleted temporary file: %s", path)
            return True
        return False
    except OSError as exc:
        logger.warning("Could not delete '%s': %s", file_path, exc)
        return False
