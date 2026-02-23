"""
wsgi.py
=======
Gunicorn / WSGI entry point for Render deployment.

Usage:
    gunicorn wsgi:app --workers 2 --threads 2 --timeout 20 --bind 0.0.0.0:$PORT
"""

from app import app  # noqa: F401 — exported for gunicorn

if __name__ == "__main__":
    app.run()
