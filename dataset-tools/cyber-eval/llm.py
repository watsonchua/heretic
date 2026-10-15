"""LiteLLM gateway client: embeddings for clustering, and the twin generator.

Paired with systemone.py, which wraps the classifier. Keeping them apart keeps the one
model-free boundary visible: diagnostics.py imports neither.

The generator must never be the judge. glm-5.2 scores responses and gives the second opinion
during calibration, so a model that wrote a twin is never asked to grade it.
"""

import json
import re

from common import DiskCache, RateGate, env, post_with_retry  # noqa: F401

EMBED_MODEL = "azure/text-embedding-3-large"
CHAT_MODEL = "gemini-3.8-flash-global"  # writes the harmless twins
JUDGE_MODEL = "glm-5.2"  # judges responses, and second opinion during calibration

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")
_gate = RateGate()


def credentials() -> tuple[str, str]:
    base, key = env("LITELLM_BASE_URL", "LITELLM_API_KEY")
    return base.rstrip("/"), key


def _post(path, body, url=None, key=None, timeout=120):
    if url is None:
        url, key = credentials()
    return post_with_retry(f"{url}{path}", body, key, _gate, timeout=timeout)


def embed(texts, model=EMBED_MODEL, batch=256, url=None, key=None, progress=False):
    """Embed a list of strings, preserving order. Returns a list of float vectors."""
    if url is None:
        url, key = credentials()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        data = _post("/v1/embeddings", {"model": model, "input": chunk}, url, key)["data"]
        out += [e["embedding"] for e in sorted(data, key=lambda x: x["index"])]
        if progress:
            print(f"  embedded {len(out)}/{len(texts)}", flush=True)
    return out


def chat(messages, model=CHAT_MODEL, temperature=0.7, url=None, key=None, **kwargs):
    body = {"model": model, "temperature": temperature, "messages": messages, **kwargs}
    return _post("/v1/chat/completions", body, url, key)["choices"][0]["message"]["content"]


def chat_json(messages, **kwargs):
    """Both chat models sometimes wrap JSON in ```json fences; strip them before parsing."""
    raw = chat(messages, **kwargs)
    try:
        return json.loads(FENCE.sub("", raw.strip()))
    except json.JSONDecodeError:
        return None


def judge(system, user, model=JUDGE_MODEL, temperature=0, retries=3, **kwargs):
    """Structured judgement from the judge model. Returns the parsed dict, or None."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    for _ in range(retries):
        result = chat_json(messages, model=model, temperature=temperature, **kwargs)
        if isinstance(result, dict):
            return result
    return None
