"""Shared helpers: environment, JSONL and table IO, disk caches, and the rate gate.

The rate gate is the only subtle piece. The LiteLLM budget is enforced per model and per key
rather than per connection, so concurrent workers share one allowance. Without a shared gate
each worker discovers a 429 separately and sleeps its own Retry-After, which wastes the window
and keeps the limiter tripped: eight workers once measured slower than one. A single "not
before" timestamp per client means the first 429 pauses everyone and the retry wave is spread.
"""

import csv
import json
import os
import random
import threading
import time
from pathlib import Path

import requests
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

RETRYABLE = (429, 500, 502, 503, 504)
MAX_BACKOFF = 60.0


def env(*names: str) -> list[str]:
    """Read required settings from the repository-root .env, with os.environ taking priority.

    Two lines in that file put spaces around `=`, so shell `source` fails on it; dotenv does not.
    """
    values = {**dotenv_values(ROOT / ".env"), **os.environ}
    missing = [n for n in names if not values.get(n)]
    if missing:
        raise SystemExit(f"missing in {ROOT / '.env'}: {', '.join(missing)}")
    return [values[n] for n in names]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def read_table(path: Path) -> list[dict]:
    """Read a parquet or CSV source file as a list of dicts."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    with path.open() as handle:
        return list(csv.DictReader(handle))


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


class DiskCache:
    """A JSON dict cache that survives interrupts.

    Several steps make hundreds of model calls, and a failure near the end of one should cost
    the current item rather than the whole run.
    """

    def __init__(self, path: Path, flush_every: int = 25):
        self.path = Path(path)
        self.flush_every = flush_every
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}
        self._since_flush = 0

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def get(self, key: str, default=None):
        return self.data.get(key, default)

    def set(self, key: str, value) -> None:
        self.data[key] = value
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        self.path.write_text(json.dumps(self.data))
        self._since_flush = 0

    def __len__(self) -> int:
        return len(self.data)


class RateGate:
    """A shared backoff window for one API client. See the module docstring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                delay = self._next_allowed - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(delay, 5.0))

    def hold_off(self, seconds: float) -> None:
        with self._lock:
            self._next_allowed = max(self._next_allowed, time.monotonic() + seconds)

    @staticmethod
    def backoff(response, attempt: int, base: float) -> float:
        """Honour Retry-After when the server sends it, else exponential backoff with jitter."""
        if response is not None:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    return min(float(retry_after), MAX_BACKOFF)
                except ValueError:
                    pass
        return min(base * (2**attempt) + random.uniform(0, 1), MAX_BACKOFF)


def post_with_retry(url, body, key, gate: RateGate, retries=8, timeout=120):
    """POST JSON with shared-gate backoff. Raises on the final attempt."""
    for attempt in range(retries):
        response = None
        gate.wait()
        try:
            response = requests.post(
                url,
                timeout=timeout,
                json=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            if response.status_code in RETRYABLE:
                raise requests.HTTPError(str(response.status_code))
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError):
            if attempt == retries - 1:
                raise
            rate_limited = response is not None and response.status_code == 429
            wait = RateGate.backoff(response, attempt, 5.0 if rate_limited else 1.0)
            if rate_limited:
                gate.hold_off(wait)
            time.sleep(wait * random.uniform(0.9, 1.3))
    raise RuntimeError("unreachable")


def cluster_by_cosine(vectors, threshold: float):
    """Average-linkage agglomerative clustering on unit-normalised vectors.

    Used both to collapse near-duplicate behaviours and to group them for splitting, so the
    two share one definition of "near-duplicate" and cannot drift apart.
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    X = np.asarray(vectors, dtype=np.float32)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric="cosine",
        linkage="average",
    ).fit_predict(X)
    return labels, X
