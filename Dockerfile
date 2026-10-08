# Content-moderation model — for AWS EC2 (or any Docker host).
#
# The 67 MB int8 model is baked into the image at build time and verified
# against the same size + sha256 the service checks at runtime, so a
# container comes up already able to score text — no first-request download
# and no volume to keep in sync.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py bake_model.py ./
RUN python bake_model.py

EXPOSE 8000

# One process is enough: scoring is CPU-bound, and FastAPI runs the sync
# /moderate handler in a threadpool, so the event loop stays free. The
# concurrency cap keeps a burst from stacking up on a 2-vCPU box.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "4"]
