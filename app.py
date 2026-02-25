"""
app.py
======
Flask application entry point — routing ONLY.

Responsibilities (single):
  - Create the Flask app and configure it for production
  - Register all routes (/, /build-resume, /analyze)
  - Attach rate limiting, file size limits, structured error handlers
  - Configure dual logging (access + error)
  - Delegate ALL business logic to core/ modules via orchestration helpers
  - Auto-delete uploaded temp files in finally blocks
  - No scoring, no parsing, no cleaning logic here

Production notes:
  - Render: set PORT env var; gunicorn reads wsgi.py → app
  - Never runs with debug=True
  - Compatible with gunicorn: `gunicorn wsgi:app --workers 2 --timeout 20`

Audit fixes applied:
  - C1: Timeout thread cancellation via threading.Event
  - C2: Rate limiter documented as per-worker; TODO Redis backend
  - C3: Auto-purge expired IPs every 100 requests
  - C4: Request data copied to locals before thread boundary
  - M1: SECRET_KEY fails loudly in production
  - M6: Stale uploads cleaned on startup
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import threading
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Callable

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from core.matcher import SimilarityResult, compute_similarity
from core.parser import ExtractionStatus, extract_text, safe_delete
from core.scoring import ScoreBreakdown, calculate_score
from core.utils import (
    ValidationError,
    clean_and_tokenize,
    clean_text,
    validate_text_input,
    word_count,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR: Path = Path(__file__).resolve().parent
LOG_DIR: Path = BASE_DIR / "logs"
UPLOAD_DIR: Path = BASE_DIR / "uploads"

LOG_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Startup: purge stale uploads from previous crashes (M6)
# ---------------------------------------------------------------------------


def _cleanup_stale_uploads() -> None:
    """Delete any leftover .pdf files in UPLOAD_DIR from a previous crash."""
    for f in UPLOAD_DIR.glob("*.pdf"):
        try:
            f.unlink()
        except OSError:
            pass


_cleanup_stale_uploads()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_CONTENT_LENGTH: int = 3 * 1024 * 1024   # 3 MB
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"pdf"})
REQUEST_TIMEOUT_SECONDS: int = 15

RATE_LIMIT_WINDOW: int = 60        # seconds
RATE_LIMIT_MAX_REQUESTS: int = 5   # per IP per window per worker
# NOTE (C2): This rate limiter is per-process. With N gunicorn workers, the
#   effective limit is N × RATE_LIMIT_MAX_REQUESTS per IP. For true
#   cross-worker limiting, migrate to flask-limiter with a Redis backend.

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    """Set up rotating file handlers for access and error logs."""

    log_format = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # Error log — WARNING and above
    error_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "error.log",
        maxBytes=5 * 1024 * 1024,   # 5 MB per file
        backupCount=3,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(log_format)

    # Access log — INFO and above (request lifecycle events)
    access_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "access.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    access_handler.setLevel(logging.INFO)
    access_handler.setFormatter(log_format)

    # Root logger — attach both handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(access_handler)

    # Console handler for local development (filtered out by gunicorn in prod)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_format)
    root_logger.addHandler(console)


_configure_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory rate limiter — per-worker, thread-safe, auto-purging (C2, C3)
# ---------------------------------------------------------------------------

_rate_store: dict[str, list[float]] = {}
_rate_lock = threading.Lock()
_rate_check_count: int = 0
_PURGE_INTERVAL: int = 100  # full sweep every N checks


def _is_rate_limited(ip: str) -> bool:
    """Return True if *ip* has exceeded RATE_LIMIT_MAX_REQUESTS in the last window.

    Every _PURGE_INTERVAL calls, a full sweep removes all expired IPs to prevent
    unbounded _rate_store growth (C3 fix).
    """
    global _rate_check_count
    now = time.monotonic()
    cutoff = now - RATE_LIMIT_WINDOW

    with _rate_lock:
        _rate_check_count += 1

        # C3 fix: periodic full purge of stale IPs
        if _rate_check_count % _PURGE_INTERVAL == 0:
            stale_ips = [
                k for k, v in _rate_store.items()
                if not any(t > cutoff for t in v)
            ]
            for k in stale_ips:
                del _rate_store[k]

        timestamps = _rate_store.get(ip, [])
        timestamps = [t for t in timestamps if t > cutoff]

        if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
            _rate_store[ip] = timestamps
            return True

        timestamps.append(now)
        _rate_store[ip] = timestamps
        return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error(message: str, status: int = 400) -> tuple:
    """Return a structured JSON error response."""
    return jsonify({"success": False, "error": message}), status


def _allowed_file(filename: str) -> bool:
    """Return True if *filename* has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _save_upload_bytes(file_bytes: bytes, original_filename: str) -> Path:
    """Save raw file bytes to UPLOAD_DIR with a UUID filename.

    Replaces the old _save_upload that took a FileStorage — this version works
    with pre-read bytes, making it safe to call from any thread (C4 fix).
    """
    suffix = Path(secure_filename(original_filename)).suffix.lower()
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(file_bytes)
    return dest


def _get_client_ip() -> str:
    """Return the sanitized real client IP, respecting X-Forwarded-For from Render's proxy.

    Newline characters are stripped to prevent log injection.
    """
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.remote_addr or "unknown"
    return ip.replace("\n", "").replace("\r", "").replace("\t", "")[:45]


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    app = Flask(__name__)

    # ── Security / production config ──────────────────────────────────────
    secret = os.environ.get("SECRET_KEY")
    if not secret and os.environ.get("RENDER"):  # M1 fix: fail loudly in prod
        raise RuntimeError(
            "SECRET_KEY environment variable is required in production. "
            "Set it in your Render dashboard."
        )
    app.config["SECRET_KEY"] = secret or os.urandom(32)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["DEBUG"] = False
    app.config["TESTING"] = False
    app.config["PROPAGATE_EXCEPTIONS"] = False

    # ── Before-request: rate limiting ─────────────────────────────────────
    @app.before_request
    def check_rate_limit():
        if request.endpoint in ("index", "health"):
            return
        ip = _get_client_ip()
        if _is_rate_limited(ip):
            logger.warning("Rate limit exceeded for IP: %s", ip)
            return _error(
                "Too many requests. You are limited to 5 requests per minute. Please wait.",
                429,
            )

    # ── After-request: security headers + access logging ────────────────
    @app.after_request
    def after_request_hooks(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' cdn.tailwindcss.com cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' fonts.googleapis.com cdn.tailwindcss.com; "
            "font-src 'self' fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self';"
        )
        if os.environ.get("RENDER"):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        logger.info(
            "%s %s %s → %d",
            _get_client_ip(),
            request.method,
            request.path,
            response.status_code,
        )
        return response

    # ── Structured error handlers ─────────────────────────────────────────
    @app.errorhandler(400)
    def bad_request(e):
        return _error("Bad request. Please check your input.", 400)

    @app.errorhandler(404)
    def not_found(e):
        return _error("Endpoint not found.", 404)

    @app.errorhandler(413)
    def file_too_large(e):
        return _error(
            "File too large. Maximum allowed size is 3 MB. Please upload a smaller PDF.",
            413,
        )

    @app.errorhandler(429)
    def rate_limited(e):
        return _error("Too many requests. Please slow down.", 429)

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception("Unhandled internal error: %s", e)
        return _error("An unexpected error occurred. Please try again later.", 500)

    # ── Routes ────────────────────────────────────────────────────────────

    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")

    @app.route("/builder", methods=["GET"])
    def builder():
        return render_template("builder.html")

    @app.route("/result", methods=["GET"])
    def result_page():
        # This route allows direct access or preview of the result template
        return render_template("result.html", analysis=None)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "service": "AI Resume Builder"}), 200

    @app.route("/build-resume", methods=["POST"])
    def build_resume():
        """
        Accept JSON with resume field names/values and return
        a structured resume data payload for frontend rendering.
        """
        # C4 fix: read body in main thread
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return _error("Request body must be valid JSON.")

        # ── Run logic in timeout-guarded thread ───────────────────────────
        result_container: list = []
        exception_container: list = []
        cancel_event = threading.Event()

        def _build_worker():
            try:
                required_fields = ["name", "summary", "skills"]
                missing = [f for f in required_fields if not data.get(f, "").strip()]
                if missing:
                    result_container.append(_error(f"Missing required fields: {', '.join(missing)}."))
                    return

                try:
                    validated_summary = validate_text_input(data.get("summary", ""), "summary")
                    validated_skills = validate_text_input(data.get("skills", ""), "skills")
                except ValidationError as exc:
                    result_container.append(_error(str(exc)))
                    return

                # C1 check
                if cancel_event.is_set():
                    return

                resume_payload = {
                    "success": True,
                    "resume": {
                        "name": clean_text(str(data.get("name", ""))),
                        "email": clean_text(str(data.get("email", ""))),
                        "phone": str(data.get("phone", "")).strip(),
                        "location": clean_text(str(data.get("location", ""))),
                        "summary": validated_summary,
                        "experience": clean_text(str(data.get("experience", ""))),
                        "education": clean_text(str(data.get("education", ""))),
                        "skills": validated_skills,
                    },
                }
                result_container.append((jsonify(resume_payload), 200))
                logger.info("Resume built for: %s", data.get("name", "unknown"))

            except Exception as exc:
                logger.exception("Unhandled error during resume building: %s", exc)
                exception_container.append(exc)

        thread = threading.Thread(target=_build_worker, daemon=True)
        thread.start()
        thread.join(timeout=REQUEST_TIMEOUT_SECONDS)

        if thread.is_alive():
            cancel_event.set()
            return _error("Request timed out.", 503)

        if exception_container:
            return _error("Internal server error.", 500)

        if result_container:
            return result_container[0]

        return _error("Processing failed.", 500)

    @app.route("/analyze", methods=["POST"])
    def analyze():
        """
        Accept a PDF resume + job description text.
        Return a full ATS compatibility score breakdown.

        C4 fix: all request data is read in the main thread BEFORE entering
        the timeout-guarded worker thread. The worker receives plain Python
        objects (bytes, str) — no Flask proxy access needed.
        """
        # ── 1. Validate file presence (main thread — safe) ────────────────
        if "resume" not in request.files:
            return _error("No resume file uploaded. Please attach a PDF.")

        resume_file = request.files["resume"]
        if not resume_file.filename:
            return _error("Uploaded file has no filename.")
        if not _allowed_file(resume_file.filename):
            return _error("Only PDF files are accepted.")

        # C4 fix: read ALL request data into local variables NOW (main thread)
        original_filename: str = resume_file.filename
        file_bytes: bytes = resume_file.read()
        jd_text_raw: str = request.form.get("jd_text", "")

        # ── 2. Validate JD text (still in main thread) ────────────────────
        try:
            jd_text_validated = validate_text_input(jd_text_raw, "jd_text")
        except ValidationError as exc:
            return _error(str(exc))

        # ── 3. Run pipeline in timeout-guarded thread ─────────────────────
        result_container: list = []
        exception_container: list = []
        cancel_event = threading.Event()

        def _analysis_worker():
            """Runs in a daemon thread — receives only plain Python objects."""
            tmp_path = None
            try:
                # 3a. Save file from bytes (no Flask proxy needed)
                tmp_path = _save_upload_bytes(file_bytes, original_filename)

                # 3b. Check cancellation before expensive extraction
                if cancel_event.is_set():
                    return

                # 3c. Extract PDF text
                extraction = extract_text(tmp_path)

                if extraction.status == ExtractionStatus.FAILED:
                    result_container.append(
                        _error(
                            f"Could not read PDF: {extraction.error_message or 'Unknown error'}. "
                            "Please ensure the file is a valid, text-based PDF."
                        )
                    )
                    return

                if extraction.is_low_text:
                    result_container.append(
                        _error(
                            "The uploaded PDF appears to be scanned or image-only "
                            f"(only {extraction.char_count} characters extracted). "
                            "Please upload a text-based PDF."
                        )
                    )
                    return

                # C1 fix: check cancellation before TF-IDF (most CPU-heavy step)
                if cancel_event.is_set():
                    return

                # 3d. Clean and tokenize
                resume_tokens = clean_and_tokenize(extraction.text)
                jd_tokens = clean_and_tokenize(jd_text_validated)

                resume_text_lower = extraction.text.lower()
                resume_wc = word_count(extraction.text)

                # 3e. Compute similarity
                resume_clean = " ".join(resume_tokens)
                jd_clean = " ".join(jd_tokens)
                similarity: SimilarityResult = compute_similarity(resume_clean, jd_clean)

                # C1 fix: check cancellation before scoring
                if cancel_event.is_set():
                    return

                # 3f. Calculate ATS score
                breakdown: ScoreBreakdown = calculate_score(
                    keyword_similarity=similarity.score,
                    resume_tokens=resume_tokens,
                    jd_tokens=jd_tokens,
                    resume_text_lower=resume_text_lower,
                    resume_word_count=resume_wc,
                )

                logger.info(
                    "Analysis complete | score=%d grade=%s pages=%d wc=%d",
                    breakdown.final_score,
                    breakdown.grade,
                    extraction.page_count,
                    resume_wc,
                )

                result_container.append((
                    jsonify({
                        "success": True,
                        "extraction": {
                            "page_count": extraction.page_count,
                            "char_count": extraction.char_count,
                            "engine_used": extraction.engine_used.name,
                            "elapsed_seconds": extraction.elapsed_seconds,
                        },
                        "analysis": breakdown.to_dict(),
                    }),
                    200,
                ))

            except Exception as exc:
                logger.exception("Unhandled error during analysis: %s", exc)
                exception_container.append(exc)

            finally:
                if tmp_path is not None:
                    safe_delete(tmp_path)

        # ── 4. Execute with timeout ───────────────────────────────────────
        thread = threading.Thread(target=_analysis_worker, daemon=True)
        thread.start()
        thread.join(timeout=REQUEST_TIMEOUT_SECONDS)

        if thread.is_alive():
            cancel_event.set()  # C1 fix: signal worker to stop
            logger.error("Analysis timed out after %ds", REQUEST_TIMEOUT_SECONDS)
            return _error("Request timed out. Please try a smaller file.", 503)

        if exception_container:
            return _error("Analysis failed due to an internal error. Please try again.", 500)

        if result_container:
            # Multi-page flow: render the result dashboard template
            # Note: You can also return JSON if the client asks for it, 
            # but for this SaaS demo we return the full result page.
            resp, status = result_container[0]
            if status == 200:
                # Extract the data from the JSON response object to pass to template
                data = resp.get_json()
                return render_template("result.html", analysis=data["analysis"])
            return resp, status

        return _error("Analysis produced no result. Please try again.", 500)

    return app


# ---------------------------------------------------------------------------
# Application instance (imported by wsgi.py)
# ---------------------------------------------------------------------------

app = create_app()

# ---------------------------------------------------------------------------
# Local development entry (never reached in production via gunicorn)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask dev server on port %d (NOT for production use)", port)
    app.run(host="0.0.0.0", port=port, debug=False)
