"""
core/scoring.py
===============
Weighted ATS scoring formula.

Responsibilities (single):
  - Compute five sub-scores from pre-processed token data
  - Apply a fixed weighted formula to produce a final ATS score
  - Clamp and round the final score to an integer in [0, 100]
  - Return a fully structured, immutable ScoreBreakdown

Sub-scores:
  keyword_similarity   (from matcher.py output)         weight: 0.40
  skill_coverage       (matched skills / total JD skills) weight: 0.25
  section_completeness (detected resume sections)        weight: 0.15
  length_quality       (token count vs. ideal range)     weight: 0.10
  action_verbs_score   (action verbs present in resume)  weight: 0.10

Constraints:
  - No parsing (parser.py handles that)
  - No TF-IDF / vectorization (matcher.py handles that)
  - Deterministic — same inputs always produce the same output
  - Division-by-zero safe throughout
  - No randomness
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Final

from core.utils import clamp

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formula weights — must sum to 1.0
# ---------------------------------------------------------------------------

W_KEYWORD_SIMILARITY: Final[float] = 0.40
W_SKILL_COVERAGE: Final[float] = 0.25
W_SECTION_COMPLETENESS: Final[float] = 0.15
W_LENGTH_QUALITY: Final[float] = 0.10
W_ACTION_VERBS: Final[float] = 0.10

_WEIGHT_SUM: Final[float] = (
    W_KEYWORD_SIMILARITY
    + W_SKILL_COVERAGE
    + W_SECTION_COMPLETENESS
    + W_LENGTH_QUALITY
    + W_ACTION_VERBS
)
assert abs(_WEIGHT_SUM - 1.0) < 1e-9, f"Weights must sum to 1.0, got {_WEIGHT_SUM}"

# ---------------------------------------------------------------------------
# Ideal resume length (word count)
# Resumes below MIN are too thin; above MAX have diminishing returns.
# ---------------------------------------------------------------------------

IDEAL_MIN_WORDS: Final[int] = 300
IDEAL_MAX_WORDS: Final[int] = 800

# ---------------------------------------------------------------------------
# Standard resume section keywords (lowercase)
# ---------------------------------------------------------------------------

SECTION_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "experience":     ("experience", "work experience", "employment", "work history", "professional experience"),
    "education":      ("education", "academic background", "qualifications", "degree", "university", "college"),
    "skills":         ("skills", "technical skills", "core competencies", "competencies", "expertise", "proficiencies"),
    "summary":        ("summary", "objective", "profile", "about me", "professional summary", "career objective"),
    "projects":       ("projects", "project experience", "personal projects", "key projects"),
    "certifications": ("certifications", "certificates", "credentials", "licenses"),
    "achievements":   ("achievements", "accomplishments", "awards", "honors", "recognition"),
}

MAX_SECTION_SCORE: Final[int] = len(SECTION_KEYWORDS)  # 7
assert MAX_SECTION_SCORE > 0, "SECTION_KEYWORDS must not be empty"
assert IDEAL_MAX_WORDS > 0, "IDEAL_MAX_WORDS must be > 0 to avoid division by zero"

# ---------------------------------------------------------------------------
# Non-skill tokens — common words that survive NLTK stopword removal but carry
# no skill signal.  Filtered from skill_coverage to prevent cross-domain
# inflation (INF-2 fix).
# ---------------------------------------------------------------------------

NON_SKILL_TOKENS: Final[frozenset[str]] = frozenset({
    "team", "work", "working", "system", "process", "service", "support",
    "project", "company", "role", "position", "ability", "strong", "good",
    "well", "year", "experience", "new", "using", "used", "use", "also",
    "including", "provide", "ensure", "make", "help", "need", "time",
    "high", "level", "area", "part", "key", "required", "must", "one",
    "two", "three", "first", "based", "across", "within", "would",
    "related", "various", "multiple", "current", "best", "day",
    "environment", "client", "customer", "business", "management",
    "solution", "tool", "knowledge", "understanding", "opportunity",
    "responsible", "requirement", "result", "plan", "report",
})

MIN_JD_TOKENS_FOR_SKILL_SCORE: Final[int] = 10  # INF-4 fix

# ---------------------------------------------------------------------------
# Action verbs — strong resume language indicators
# ---------------------------------------------------------------------------

ACTION_VERBS: Final[frozenset[str]] = frozenset({
    # Leadership / Management
    "led", "managed", "directed", "supervised", "oversaw", "coordinated",
    "spearheaded", "orchestrated", "established", "founded", "championed",
    # Achievement / Delivery
    "delivered", "achieved", "accomplished", "exceeded", "surpassed",
    "generated", "increased", "improved", "reduced", "saved", "cut",
    "boosted", "accelerated", "streamlined", "optimized", "maximized",
    # Building / Creating
    "built", "developed", "designed", "architected", "engineered", "created",
    "launched", "implemented", "deployed", "shipped", "produced", "authored",
    # Collaboration
    "collaborated", "partnered", "facilitated", "mentored", "coached", "trained",
    "advised", "consulted", "negotiated", "persuaded", "presented",
    # Analysis / Research
    "analyzed", "researched", "evaluated", "assessed", "audited", "diagnosed",
    "identified", "investigated", "reviewed", "measured", "monitored",
    # Transformation
    "transformed", "restructured", "revamped", "reengineered", "modernized",
    "automated", "integrated", "migrated", "consolidated", "standardized",
})

STRONG_ACTION_VERB_THRESHOLD: Final[int] = 8   # ≥8 unique verbs → full score

# ---------------------------------------------------------------------------
# Grade mapping
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS: Final[list[tuple[int, str, str]]] = [
    (85, "A",  "Excellent match — strong ATS compatibility."),
    (70, "B",  "Good match — minor gaps to address."),
    (55, "C",  "Moderate match — noticeable alignment gaps."),
    (40, "D",  "Weak match — significant tailoring needed."),
    (0,  "F",  "Poor match — resume needs major revision for this role."),
]


def _grade(score: int) -> tuple[str, str]:
    """Return (letter_grade, description) for an integer score in [0, 100].

    The threshold list includes a 0-threshold entry for 'F', so this loop
    always returns before exhaustion.  The trailing return is a safety net
    for type-checkers only and should never be reached at runtime.
    """
    for threshold, letter, description in _GRADE_THRESHOLDS:
        if score >= threshold:
            return letter, description
    # Safety net: score < 0 after clamping would be a bug — log and return F
    logger.error("_grade() received out-of-range score: %d — defaulting to F", score)
    return "F", _GRADE_THRESHOLDS[-1][2]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreBreakdown:
    """Immutable, structured ATS scoring result.

    All sub-scores are in [0.0, 100.0].
    ``final_score`` is an integer in [0, 100].
    """

    # Sub-scores (float, 0–100)
    keyword_similarity: float
    skill_coverage: float
    section_completeness: float
    length_quality: float
    action_verbs_score: float

    # Final output
    final_score: int
    grade: str
    grade_description: str

    # Diagnostics (for frontend display)
    matched_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    detected_sections: list[str] = field(default_factory=list)
    missing_sections: list[str] = field(default_factory=list)
    matched_action_verbs: list[str] = field(default_factory=list)
    resume_word_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for JSON responses."""
        return {
            "final_score": self.final_score,
            "grade": self.grade,
            "grade_description": self.grade_description,
            "sub_scores": {
                "keyword_similarity": round(self.keyword_similarity, 2),
                "skill_coverage": round(self.skill_coverage, 2),
                "section_completeness": round(self.section_completeness, 2),
                "length_quality": round(self.length_quality, 2),
                "action_verbs_score": round(self.action_verbs_score, 2),
            },
            "diagnostics": {
                "matched_skills": self.matched_skills,
                "missing_skills": self.missing_skills,
                "detected_sections": self.detected_sections,
                "missing_sections": self.missing_sections,
                "matched_action_verbs": self.matched_action_verbs,
                "resume_word_count": self.resume_word_count,
            },
        }


# ---------------------------------------------------------------------------
# Sub-score calculators (pure functions — no side effects)
# ---------------------------------------------------------------------------


def _score_skill_coverage(
    resume_tokens: list[str],
    jd_tokens: list[str],
) -> tuple[float, list[str], list[str]]:
    """Compute skill coverage as (matched / total_jd_skills) × 100.

    INF-2 fix: non-skill common tokens are excluded before matching.
    INF-4 fix: returns 0.0 if JD has fewer than MIN_JD_TOKENS_FOR_SKILL_SCORE
    unique tokens to prevent inflation from trivial JDs.

    Division-by-zero safe: returns 0.0 if JD has no tokens.

    Returns
    -------
    tuple[float, list[str], list[str]]
        (score_0_to_100, matched_skills, missing_skills)
    """
    if not jd_tokens:
        return 0.0, [], []

    # INF-2 fix: filter out non-skill common tokens from both sets
    resume_set = frozenset(resume_tokens) - NON_SKILL_TOKENS
    jd_set = frozenset(jd_tokens) - NON_SKILL_TOKENS

    if not jd_set:
        return 0.0, [], []

    # INF-4 fix: require minimum JD complexity for meaningful skill scoring
    if len(jd_set) < MIN_JD_TOKENS_FOR_SKILL_SCORE:
        logger.warning(
            "JD has only %d unique skill tokens (min %d) — skill_coverage set to 0.",
            len(jd_set),
            MIN_JD_TOKENS_FOR_SKILL_SCORE,
        )
        return 0.0, [], sorted(jd_set)

    matched = sorted(resume_set & jd_set)
    missing = sorted(jd_set - resume_set)

    # Safe division — jd_set is guaranteed non-empty here
    raw = len(matched) / len(jd_set)
    return clamp(raw * 100.0), matched, missing


def _score_section_completeness(resume_text_lower: str) -> tuple[float, list[str], list[str]]:
    """Detect standard resume sections via word-boundary keyword matching.

    INF-1 fix: uses regex word-boundary matching instead of substring `in`.
    Keywords must appear as standalone words/phrases near line starts or
    followed by colons/dashes to qualify as section headers, not incidental
    mentions in body text.

    Score = (sections_found / MAX_SECTION_SCORE) × 100

    Returns
    -------
    tuple[float, list[str], list[str]]
        (score_0_to_100, detected_section_names, missing_section_names)
    """
    detected: list[str] = []
    missing: list[str] = []

    for section_name, keywords in SECTION_KEYWORDS.items():
        # INF-1 fix: require keyword at line start (with optional leading whitespace)
        # OR followed by a colon/dash — typical section header patterns.
        found = any(
            re.search(
                r'(?:^|\n)\s*' + re.escape(kw) + r'\s*(?::|\-|–|—|\n|$)',
                resume_text_lower,
            )
            for kw in keywords
        )
        if found:
            detected.append(section_name)
        else:
            missing.append(section_name)

    # Safe division — MAX_SECTION_SCORE is a compile-time constant > 0
    raw = len(detected) / MAX_SECTION_SCORE
    return clamp(raw * 100.0), detected, missing


def _score_length_quality(word_count: int) -> float:
    """Score resume length against the ideal [IDEAL_MIN_WORDS, IDEAL_MAX_WORDS] range.

    Scoring logic:
      - Within ideal range                 → 100.0
      - Below minimum (too short)          → linear decay toward 0
      - Above maximum (too long, > 1.5×)   → linear decay, floor at 40.0

    Division-by-zero safe: IDEAL_MIN_WORDS and IDEAL_MAX_WORDS are constants > 0.
    """
    if word_count <= 0:
        return 0.0

    if IDEAL_MIN_WORDS <= word_count <= IDEAL_MAX_WORDS:
        return 100.0

    if word_count < IDEAL_MIN_WORDS:
        # Linear scale from 0 (at 0 words) to 100 (at IDEAL_MIN_WORDS)
        return clamp((word_count / IDEAL_MIN_WORDS) * 100.0)

    # Above ideal max — decay from 100 toward 40 as length doubles
    excess_ratio = (word_count - IDEAL_MAX_WORDS) / IDEAL_MAX_WORDS  # 0 → ∞
    score = 100.0 - (excess_ratio * 60.0)   # drops 60 pts over 1× the ideal range
    return clamp(score, 40.0, 100.0)


def _score_action_verbs(resume_tokens: list[str]) -> tuple[float, list[str]]:
    """Count unique action verbs in resume tokens.

    Score = min(unique_verb_count / STRONG_ACTION_VERB_THRESHOLD, 1.0) × 100

    Returns
    -------
    tuple[float, list[str]]
        (score_0_to_100, matched_verb_list)
    """
    if not resume_tokens:
        return 0.0, []

    token_set = frozenset(resume_tokens)
    matched = sorted(token_set & ACTION_VERBS)
    unique_count = len(matched)

    # Safe division — STRONG_ACTION_VERB_THRESHOLD is a constant > 0
    raw = min(unique_count / STRONG_ACTION_VERB_THRESHOLD, 1.0)
    return clamp(raw * 100.0), matched


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_score(
    *,
    keyword_similarity: float,
    resume_tokens: list[str],
    jd_tokens: list[str],
    resume_text_lower: str,
    resume_word_count: int,
) -> ScoreBreakdown:
    """Apply the weighted ATS scoring formula and return a :class:`ScoreBreakdown`.

    Parameters
    ----------
    keyword_similarity:
        Cosine similarity score from ``matcher.compute_similarity().score``.
        Expected range [0.0, 100.0] — will be clamped defensively.
    resume_tokens:
        Lemmatized, stopword-filtered tokens from the resume
        (output of ``utils.clean_and_tokenize``).
    jd_tokens:
        Lemmatized, stopword-filtered tokens from the JD.
    resume_text_lower:
        Full lowercased resume text (used for section keyword matching).
        Pass ``""`` if unavailable — section score will be 0.
    resume_word_count:
        Raw word count of the unfiltered resume text (used for length scoring).

    Returns
    -------
    ScoreBreakdown
        Structured, immutable result.  Always valid — never raises.
    """
    # ── Defensive clamp on incoming keyword_similarity ───────────────────
    kw_sim = clamp(float(keyword_similarity) if isinstance(keyword_similarity, (int, float)) else 0.0)

    # ── Sub-score: skill coverage ─────────────────────────────────────────
    skill_cov, matched_skills, missing_skills = _score_skill_coverage(
        resume_tokens or [],
        jd_tokens or [],
    )

    # ── Sub-score: section completeness ──────────────────────────────────
    section_score, detected_sections, missing_sections = _score_section_completeness(
        resume_text_lower if isinstance(resume_text_lower, str) else ""
    )

    # ── Sub-score: length quality ─────────────────────────────────────────
    length_score = _score_length_quality(
        int(resume_word_count) if isinstance(resume_word_count, (int, float)) else 0
    )

    # ── Sub-score: action verbs ───────────────────────────────────────────
    verbs_score, matched_verbs = _score_action_verbs(resume_tokens or [])

    # ── Weighted formula ──────────────────────────────────────────────────
    raw_final = (
        (kw_sim       * W_KEYWORD_SIMILARITY) +
        (skill_cov    * W_SKILL_COVERAGE)     +
        (section_score * W_SECTION_COMPLETENESS) +
        (length_score * W_LENGTH_QUALITY)     +
        (verbs_score  * W_ACTION_VERBS)
    )

    final_clamped = clamp(raw_final, 0.0, 100.0)
    final_int = int(round(final_clamped))
    letter_grade, grade_desc = _grade(final_int)

    logger.info(
        "ATS score: %d (%s) | kw_sim=%.1f  skill_cov=%.1f  "
        "sections=%.1f  length=%.1f  verbs=%.1f",
        final_int,
        letter_grade,
        kw_sim,
        skill_cov,
        section_score,
        length_score,
        verbs_score,
    )

    return ScoreBreakdown(
        keyword_similarity=round(kw_sim, 2),
        skill_coverage=round(skill_cov, 2),
        section_completeness=round(section_score, 2),
        length_quality=round(length_score, 2),
        action_verbs_score=round(verbs_score, 2),
        final_score=final_int,
        grade=letter_grade,
        grade_description=grade_desc,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        detected_sections=detected_sections,
        missing_sections=missing_sections,
        matched_action_verbs=matched_verbs,
        resume_word_count=int(resume_word_count) if isinstance(resume_word_count, (int, float)) else 0,
    )
