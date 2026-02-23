"""
core/utils.py
=============
Text cleaning pipeline, input validation, and safe helper utilities.

Responsibilities (single):
  - HTML / script stripping
  - Unicode normalization
  - Stopword removal
  - Lemmatization (via NLTK WordNetLemmatizer)
  - Word count calculation
  - Length validation
  - Clamp helper (0–100)

Constraints:
  - Fully deterministic — no randomness
  - Defensive: handles empty / None / non-string input without crashing
  - Division-by-zero safe
  - No Flask, no I/O, no side-effects
"""

from __future__ import annotations

import re
import unicodedata
from html.parser import HTMLParser
from typing import Final

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------------------------------------------------------
# NLTK resource bootstrap (idempotent — safe to call at import time)
# ---------------------------------------------------------------------------
_NLTK_RESOURCES: Final[list[tuple[str, str]]] = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
]

for _path, _pkg in _NLTK_RESOURCES:
    try:
        nltk.data.find(_path)
    except LookupError:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception as _dl_exc:  # OSError on read-only disk, network error, etc.
            raise RuntimeError(
                f"Failed to download NLTK resource '{_pkg}'. "
                "On Render, set NLTK_DATA env var to a writable path, or pre-download "
                f"resources in the build step. Original error: {_dl_exc}"
            ) from _dl_exc

# ---------------------------------------------------------------------------
# Module-level singletons (instantiated once — thread-safe for read-only use)
# ---------------------------------------------------------------------------
_LEMMATIZER: Final[WordNetLemmatizer] = WordNetLemmatizer()
_STOPWORDS: Final[frozenset[str]] = frozenset(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------
MIN_TEXT_LENGTH: Final[int] = 20       # characters
MAX_TEXT_LENGTH: Final[int] = 100_000  # characters (~15,000 words)
MIN_WORD_COUNT: Final[int] = 5

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ValidationError(ValueError):
    """Raised when input fails validation checks."""


# ---------------------------------------------------------------------------
# Internal: HTML stripper
# ---------------------------------------------------------------------------


class _HTMLStripper(HTMLParser):
    """Minimal, dependency-free HTML tag stripper."""

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self._parts.append(data)

    def get_text(self) -> str:
        return " ".join(self._parts)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Return *value* clamped to [low, high].

    Deterministic — no randomness.  Division-by-zero cannot occur here.

    >>> clamp(120.0)
    100.0
    >>> clamp(-5.0)
    0.0
    >>> clamp(73.4)
    73.4
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"clamp() expects a numeric value, got {type(value).__name__!r}")
    return max(low, min(high, float(value)))


def strip_html(text: str) -> str:
    """Remove all HTML tags and inline <script>/<style> blocks from *text*.

    Returns plain text with tags replaced by spaces.  Safe on already-clean
    strings (no-op when no tags are present).
    """
    if not isinstance(text, str):
        return ""

    # Remove <script>…</script> and <style>…</style> blocks first
    text = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    stripper = _HTMLStripper()
    try:
        stripper.feed(text)
        return stripper.get_text()
    except Exception:
        # Fall back to a naive regex strip if the parser chokes on malformed HTML
        return re.sub(r"<[^>]+>", " ", text)


def normalize_unicode(text: str) -> str:
    """Normalize *text* to NFC unicode form and strip non-printable characters.

    Converts curly quotes, em-dashes, and other fancy unicode to their closest
    ASCII equivalents where possible, then removes remaining control chars.
    """
    if not isinstance(text, str):
        return ""

    # NFC normalization (canonical decomposition + canonical composition)
    text = unicodedata.normalize("NFC", text)

    # Replace common fancy punctuation with plain ASCII equivalents
    replacements: dict[str, str] = {
        "\u2018": "'",  # left single quotation mark
        "\u2019": "'",  # right single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en-dash
        "\u2014": "-",  # em-dash
        "\u2026": "...",  # ellipsis
        "\u00a0": " ",  # non-breaking space
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # Strip non-printable / control characters (keep newlines and tabs)
    text = "".join(
        ch for ch in text if unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t"
    )

    return text


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Filter *tokens* list, removing English stopwords.

    Returns a new list — original is not mutated.
    Tokens are compared case-insensitively against the stopword set.
    """
    if not tokens:
        return []
    return [t for t in tokens if t.lower() not in _STOPWORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Return a new list of lemmatized tokens using multi-POS strategy.

    INF-5 fix: tries verb, noun, then adjective POS tags.  Picks the first
    form that actually changes the token (e.g., 'managed' → 'manage' via
    verb POS, instead of staying 'managed' under noun POS).

    Uses NLTK WordNetLemmatizer — deterministic, no randomness.
    Empty or non-string tokens are skipped defensively.
    """
    if not tokens:
        return []

    result: list[str] = []
    for token in tokens:
        if not isinstance(token, str) or not token.strip():
            continue
        lower = token.lower()
        try:
            # Try verb first (most impactful for action words),
            # then noun, then adjective.
            lemma = lower
            for pos in ("v", "n", "a"):
                candidate = _LEMMATIZER.lemmatize(lower, pos=pos)
                if candidate != lower:
                    lemma = candidate
                    break
            result.append(lemma)
        except Exception:
            result.append(lower)
    return result


def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase alphabetic tokens (no punctuation or digits).

    Returns an empty list on empty / non-string input — never raises.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    return re.findall(r"[a-z]+", text.lower())


def word_count(text: str) -> int:
    """Return the number of whitespace-delimited words in *text*.

    Returns 0 on empty / non-string input.  Division-by-zero safe.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def validate_text_input(text: object, field_name: str = "input") -> str:
    """Validate that *text* is a non-empty string within allowed length bounds.

    Returns the stripped string on success.
    Raises :class:`ValidationError` with a descriptive message on failure.

    Parameters
    ----------
    text:
        The value to validate.
    field_name:
        Human-readable name used in error messages (e.g. ``"resume_text"``).
    """
    if text is None:
        raise ValidationError(f"{field_name}: must not be None.")

    if not isinstance(text, str):
        raise ValidationError(
            f"{field_name}: expected str, got {type(text).__name__!r}."
        )

    stripped = text.strip()

    if not stripped:
        raise ValidationError(f"{field_name}: must not be empty or whitespace-only.")

    if len(stripped) < MIN_TEXT_LENGTH:
        raise ValidationError(
            f"{field_name}: too short ({len(stripped)} chars). "
            f"Minimum is {MIN_TEXT_LENGTH} characters."
        )

    if len(stripped) > MAX_TEXT_LENGTH:
        raise ValidationError(
            f"{field_name}: too long ({len(stripped)} chars). "
            f"Maximum is {MAX_TEXT_LENGTH} characters."
        )

    wc = word_count(stripped)
    if wc < MIN_WORD_COUNT:
        raise ValidationError(
            f"{field_name}: too few words ({wc}). "
            f"Minimum is {MIN_WORD_COUNT} words."
        )

    return stripped


def clean_text(raw: str) -> str:
    """Full deterministic text cleaning pipeline.

    Steps (in order):
      1. Validate type (returns empty string for non-str rather than raising)
      2. Strip HTML tags and script/style blocks
      3. Normalize unicode
      4. Collapse whitespace
      5. Convert to lowercase

    This function is intentionally *non-raising* for use inside larger
    pipelines — callers that need strict validation should call
    :func:`validate_text_input` first.

    Returns
    -------
    str
        Cleaned, lowercase, whitespace-normalized string.
        Returns ``""`` if input cannot be meaningfully cleaned.
    """
    if not isinstance(raw, str):
        return ""

    text = strip_html(raw)
    text = normalize_unicode(text)

    # Collapse multiple whitespace characters (including newlines) into a single space
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stray punctuation clusters (keep hyphens within words, e.g. "full-stack")
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.lower()


def clean_and_tokenize(raw: str) -> list[str]:
    """Clean *raw* text, tokenize, remove stopwords, and lemmatize.

    This is the standard preprocessing pipeline consumed by
    ``matcher.py`` and ``scoring.py``.

    Returns
    -------
    list[str]
        Final list of meaningful, lemmatized tokens.
        Returns ``[]`` on empty or non-string input.
    """
    cleaned = clean_text(raw)
    if not cleaned:
        return []

    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)

    return tokens
