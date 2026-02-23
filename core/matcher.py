"""
core/matcher.py
===============
TF-IDF vectorization and cosine similarity scoring.

Responsibilities (single):
  - Vectorize resume text and job description using TF-IDF
  - Compute cosine similarity between the two vectors
  - Enforce feature limits to prevent memory explosion
  - Enforce JD length cap before vectorization
  - Return a deterministic similarity score clamped to [0.0, 100.0]

Constraints:
  - No scoring formula logic (that lives in scoring.py)
  - No text cleaning (that lives in utils.py)
  - No PDF I/O (that lives in parser.py)
  - Fully deterministic — same inputs always produce the same score
  - Safe on empty / whitespace-only strings
  - Never raises to callers — all edge cases return 0.0
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.utils import clamp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guard constants
# ---------------------------------------------------------------------------

# Maximum TF-IDF features — prevents sparse matrix from blowing up RAM.
# At 512 features, a 2×512 float64 matrix is ~8 KB — trivially small.
MAX_FEATURES: Final[int] = 512

# Hard cap on JD character length fed into the vectorizer.
# Prevents a malicious or runaway JD from stalling sklearn.
MAX_JD_CHARS: Final[int] = 15_000

# Hard cap on resume character length.
MAX_RESUME_CHARS: Final[int] = 30_000

# TF-IDF n-gram range — unigrams + bigrams capture phrase context.
NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)

# Score multiplier: raw cosine similarity is [0.0, 1.0] → scale to [0.0, 100.0]
SCORE_SCALE: Final[float] = 100.0

# INF-7 fix: minimum token count for full cosine similarity weight.
# Below this, a linear penalty is applied to reduce noise on short documents.
MIN_MEANINGFUL_TOKENS: Final[int] = 200

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SimilarityResult:
    """Immutable result from :func:`compute_similarity`.

    Attributes
    ----------
    score:
        Cosine similarity scaled to [0.0, 100.0].  Always a valid float.
    raw_cosine:
        Unscaled cosine similarity in [0.0, 1.0].
    resume_token_count:
        Number of space-delimited tokens in the (possibly truncated) resume text.
    jd_token_count:
        Number of space-delimited tokens in the (possibly truncated) JD text.
    was_truncated:
        ``True`` if either input was truncated before vectorization.
    """

    score: float
    raw_cosine: float
    resume_token_count: int
    jd_token_count: int
    was_truncated: bool


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_token_count(text: str) -> int:
    """Return the number of whitespace-delimited tokens.  Division-by-zero safe."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def _truncate(text: str, max_chars: int, label: str) -> tuple[str, bool]:
    """Hard-truncate *text* to *max_chars* characters.

    Returns
    -------
    tuple[str, bool]
        (possibly_truncated_text, was_truncated)
    """
    if len(text) > max_chars:
        logger.warning(
            "%s truncated from %d to %d characters before vectorization.",
            label,
            len(text),
            max_chars,
        )
        return text[:max_chars], True
    return text, False


def _is_meaningful(text: str) -> bool:
    """Return True if *text* has at least one non-whitespace character."""
    return isinstance(text, str) and bool(text.strip())


def _zero_result(resume_text: str, jd_text: str, was_truncated: bool) -> SimilarityResult:
    """Return a safe zero-score result for degenerate inputs."""
    return SimilarityResult(
        score=0.0,
        raw_cosine=0.0,
        resume_token_count=_safe_token_count(resume_text),
        jd_token_count=_safe_token_count(jd_text),
        was_truncated=was_truncated,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_similarity(resume_text: str, jd_text: str) -> SimilarityResult:
    """Compute TF-IDF cosine similarity between *resume_text* and *jd_text*.

    Both inputs are expected to be **pre-cleaned** strings (output of
    ``utils.clean_and_tokenize`` joined with spaces, or any cleaned text).
    This function does **not** clean or lemmatize — that is ``utils.py``'s job.

    Strategy
    --------
    1. Guard against empty / non-string inputs → return score=0.0 immediately.
    2. Truncate inputs to per-type character caps (memory safety).
    3. Fit a shared :class:`TfidfVectorizer` on both documents simultaneously
       so the feature space is identical (deterministic).
    4. Compute cosine similarity between the two resulting vectors.
    5. Scale to [0.0, 100.0] and clamp (handles floating-point rounding).

    This function **never raises**.  All failures are logged and reflected as
    ``score=0.0`` in the returned :class:`SimilarityResult`.

    Parameters
    ----------
    resume_text:
        Cleaned resume text (space-joined tokens or full cleaned string).
    jd_text:
        Cleaned job description text.

    Returns
    -------
    SimilarityResult
        Always returns a result.  Check ``result.score`` — it is always a
        finite float in [0.0, 100.0].
    """
    # ── Guard: type and emptiness ─────────────────────────────────────────
    if not _is_meaningful(resume_text):
        logger.warning("compute_similarity: resume_text is empty or None — returning 0.0")
        return _zero_result(resume_text or "", jd_text or "", False)

    if not _is_meaningful(jd_text):
        logger.warning("compute_similarity: jd_text is empty or None — returning 0.0")
        return _zero_result(resume_text, jd_text or "", False)

    # ── Truncation (memory safety) ────────────────────────────────────────
    resume_text, truncated_resume = _truncate(resume_text, MAX_RESUME_CHARS, "resume_text")
    jd_text, truncated_jd = _truncate(jd_text, MAX_JD_CHARS, "jd_text")
    was_truncated = truncated_resume or truncated_jd

    resume_tokens = _safe_token_count(resume_text)
    jd_tokens = _safe_token_count(jd_text)

    # ── Guard: post-truncation emptiness ─────────────────────────────────
    if resume_tokens == 0 or jd_tokens == 0:
        logger.warning(
            "compute_similarity: text became empty after truncation — returning 0.0"
        )
        return _zero_result(resume_text, jd_text, was_truncated)

    # ── TF-IDF vectorization ──────────────────────────────────────────────
    try:
        vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            # Determinism-critical settings:
            strip_accents="unicode",   # consistent unicode handling
            analyzer="word",
            sublinear_tf=True,         # log(1+tf) — reduces impact of high-freq terms
            dtype=np.float64,           # explicit dtype — float64 is stable across sklearn versions
        )

        # Fit on both documents at once so shared vocabulary is deterministic
        tfidf_matrix = vectorizer.fit_transform([resume_text, jd_text])

    except ValueError as exc:
        # TfidfVectorizer raises ValueError when vocabulary is empty
        # (e.g., both docs contain only stop words after internal filtering)
        logger.warning("TF-IDF vectorization produced empty vocabulary: %s", exc)
        return _zero_result(resume_text, jd_text, was_truncated)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during TF-IDF vectorization: %s", exc)
        return _zero_result(resume_text, jd_text, was_truncated)

    # ── Cosine similarity ─────────────────────────────────────────────────
    try:
        # tfidf_matrix is shape (2, n_features)
        # cosine_similarity returns a (2, 2) matrix; [0, 1] is what we want
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        raw_cosine: float = float(similarity_matrix[0, 0])
    except Exception as exc:  # noqa: BLE001
        logger.error("Cosine similarity computation failed: %s", exc)
        return _zero_result(resume_text, jd_text, was_truncated)

    # ── Guard: NaN / Inf (defensive — sklearn should not produce these) ───
    if not math.isfinite(raw_cosine):
        logger.warning(
            "Non-finite cosine similarity value (%s) — clamping to 0.0", raw_cosine
        )
        raw_cosine = 0.0

    # ── INF-7 fix: length penalty for short documents ─────────────────────
    # Short documents (<200 tokens) produce noisy cosine similarity.
    # Scale down proportionally to the shorter document's length.
    shorter_doc_tokens = min(resume_tokens, jd_tokens)
    if shorter_doc_tokens < MIN_MEANINGFUL_TOKENS:
        length_factor = shorter_doc_tokens / MIN_MEANINGFUL_TOKENS
        raw_cosine = raw_cosine * length_factor
        logger.info(
            "Applied short-doc penalty: factor=%.3f (shorter_tokens=%d, min=%d)",
            length_factor,
            shorter_doc_tokens,
            MIN_MEANINGFUL_TOKENS,
        )

    # ── Scale and clamp ───────────────────────────────────────────────────
    scaled = raw_cosine * SCORE_SCALE
    final_score = clamp(scaled, 0.0, 100.0)

    logger.info(
        "Similarity computed: raw_cosine=%.4f  score=%.2f  "
        "resume_tokens=%d  jd_tokens=%d  truncated=%s",
        raw_cosine,
        final_score,
        resume_tokens,
        jd_tokens,
        was_truncated,
    )

    return SimilarityResult(
        score=final_score,
        raw_cosine=round(raw_cosine, 6),
        resume_token_count=resume_tokens,
        jd_token_count=jd_tokens,
        was_truncated=was_truncated,
    )
