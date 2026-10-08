"""Download the model-v3 release assets into the image at build time.

It reuses main.py's own size + sha256 constants, so an image can never be
built around a wrong or half-downloaded model. After this runs, a container
starts with the model already on disk: no 67 MB download on the first
request, and no volume to keep in sync.

A transient network hiccup should not break a build, so each file is retried
a few times; if it still cannot be fetched *and* verified, the build fails
loudly rather than shipping a broken model.

    python bake_model.py
"""
from __future__ import annotations

import time

import main

FILES = [
    (main.TOKENIZER_CONFIG_PATH, main.TOKENIZER_CONFIG_SIZE, main.TOKENIZER_CONFIG_SHA256),
    (main.TOKENIZER_PATH, main.TOKENIZER_SIZE, main.TOKENIZER_SHA256),
    (main.MODEL_PATH, main.MODEL_SIZE, main.MODEL_SHA256),
]

ATTEMPTS = 3


def fetch(path: str, size: int, sha256: str) -> None:
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            main._download_verified(path, size, sha256)
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            print("  attempt %d/%d failed for %s: %s" % (attempt, ATTEMPTS, path, exc))
            time.sleep(2 * attempt)
    raise RuntimeError("could not fetch %s after %d attempts: %s" % (path, ATTEMPTS, last))


def main_entry() -> None:
    for path, size, sha in FILES:
        if main._file_ok(path, size, sha):
            print("already present and verified:", path)
            continue
        print("fetching %s (%d bytes) ..." % (path, size))
        fetch(path, size, sha)
        print("verified (size + sha256 ok):", path)
    print("model files ready")


if __name__ == "__main__":
    main_entry()
