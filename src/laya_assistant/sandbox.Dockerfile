FROM python:3.13-slim
# Built once (`ensure_image`), so a session container starts in about a second and never pip-installs on the hot path.
RUN pip install --no-cache-dir pytest && mkdir -p /workspace
WORKDIR /workspace
