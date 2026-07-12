"""LLM-powered steps for the paper-collecter pipeline.
Supports BOTH OpenAI-compatible (/v1/chat/completions) and Anthropic-compatible
(/anthropic/v1/messages) endpoints. Auto-detects from AI_BASE_URL.
Python stdlib only — no pip install required.
"""
import os
import json
import urllib.request

AI_BASE = os.environ.get("AI_BASE_URL", "https://api.deepseek.com/anthropic")
AI_KEY  = os.environ.get("AI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "deepseek-v4-pro")

# Auto-detect endpoint type
_IS_ANTHROPIC = "/anthropic" in AI_BASE

print(f"[llm] DEBUG: base={AI_BASE[:50]} model={AI_MODEL} key_ok={'yes' if AI_KEY else 'MISSING!'} anthropic={_IS_ANTHROPIC}")


def _chat(system: str, user: str, temperature: float = 0.3, max_tokens: int = 4096) -> str:
    """Single-turn chat. Returns response text or '' on failure."""
    if not AI_KEY:
        print("[llm] AI_API_KEY not set — skipping LLM step")
        return ""

    if _IS_ANTHROPIC:
        # Anthropic Messages API format
        url = AI_BASE.rstrip("/") + "/v1/messages"
        body = {
            "model": AI_MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": AI_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
    else:
        # OpenAI-compatible chat/completions format
        url = AI_BASE.rstrip("/") + "/chat/completions"
        body = {
            "model": AI_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {AI_KEY}",
            "Content-Type": "application/json",
        }

    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw_bytes = r.read()
            data = json.loads(raw_bytes)
            print(f"[llm] HTTP {r.status}, resp keys: {list(data.keys())[:5]}, "
                  f"content_len={len(json.dumps(data))}")
        if _IS_ANTHROPIC:
            blocks = data.get("content", [])
            text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
            print(f"[llm] anthropic parsed: {len(blocks)} blocks, text_len={len(text)}")
            if not text and not blocks:
                # Maybe the response uses a different format?
                print(f"[llm] WARNING: no content blocks found, raw keys: {list(data.keys())}")
                print(f"[llm] raw sample: {str(data)[:300]}")
            return text
        else:
            text = data["choices"][0]["message"]["content"].strip()
            return text
    except Exception as e:
        print(f"[llm] API call failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# Step 1 — Query expansion
# ---------------------------------------------------------------------------

EXPAND_SYSTEM = """You are a research librarian. Given a research keyword and its domain,
generate 2-3 English search queries that maximise recall across academic
search engines (arXiv, Crossref, Semantic Scholar). Use synonyms, full forms,
and adjacent sub-topics. Return ONLY valid JSON, no other text.

Format:
{"queries": ["query1", "query2", "query3"]}"""

def expand_queries(keyword: str, domain: str = "physics") -> list[str]:
    """Return a list of 2-3 expanded search queries for a keyword."""
    user = f"Keyword: {keyword}\nDomain: {domain}"
    raw = _chat(EXPAND_SYSTEM, user, temperature=0.4, max_tokens=256)
    if not raw:
        return [keyword]
    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        return json.loads(raw)["queries"]
    except (json.JSONDecodeError, KeyError):
        print(f"[llm] expand_queries parse failed, falling back to raw keyword")
        return [keyword]


# ---------------------------------------------------------------------------
# Step 3 — Filter relevance & write Chinese summaries
# ---------------------------------------------------------------------------

FILTER_SYSTEM = """You are a Chinese-speaking condensed-matter physicist curating a daily
literature digest. You receive a JSON array of candidate papers. For EACH
item decide:

1. Is it genuinely about physics / condensed matter / quantum science?
   Drop blockchain, finance, pure CS, medical, random GitHub repos with no
   physics connection.

2. For KEPT items, write a concise Chinese summary:
   - "tldr": 一句话核心 (<=60字)
   - "method": 方法简述 (<=80字), leave empty for GitHub repos
   - "contributions": [核心贡献1, 核心贡献2] (1-3 items), leave empty for GitHub repos

Return ONLY valid JSON (no markdown fences, no other text) — an array of the kept items:

[{
  "title": "原样保留",
  "tldr": "...",
  "method": "...",
  "contributions": ["...", "..."]
}, ...]

Sort papers (arXiv, Crossref) first, GitHub repos last."""


def _dumb_filter(candidates: list[dict]) -> list[dict]:
    """No-LLM fallback: keep only academic sources (arXiv/Crossref), drop GitHub."""
    kept = [c for c in candidates if c.get("source") in ("arXiv", "Crossref", "Semantic Scholar")]
    print(f"[llm] dumb filter: kept {len(kept)}/{len(candidates)} (academic sources only)")
    return [{**c, "tldr": (c.get("abstract", "") or "")[:80],
             "method": "", "contributions": []} for c in kept]


def filter_and_summarize(candidates: list[dict]) -> list[dict]:
    """Filter candidates for physics relevance and add Chinese summaries."""
    if not candidates:
        return []

    slim = []
    for c in candidates:
        slim.append({
            "source": c.get("source", ""),
            "topic": c.get("topic", ""),
            "title": c.get("title", ""),
            "abstract": (c.get("abstract", "") or "")[:600],
            "venue": c.get("venue", ""),
        })
    user = json.dumps(slim, ensure_ascii=False, indent=2)
    raw = _chat(FILTER_SYSTEM, user, temperature=0.3, max_tokens=4096)
    if not raw:
        print("[llm] filter API failed — using dumb filter (academic sources only, no GitHub spam)")
        return _dumb_filter(candidates)

    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        kept = json.loads(raw)
    except json.JSONDecodeError:
        print("[llm] filter parse failed — using dumb filter")
        return _dumb_filter(candidates)

    out = []
    for item in kept:
        title = item.get("title", "")
        match = next((c for c in candidates if c.get("title") == title), None)
        if match:
            out.append({**match, **item})
        else:
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Step 4 — Hot-topic synthesis (optional)
# ---------------------------------------------------------------------------

TRENDS_SYSTEM = """You are a condensed-matter physics research strategist.
Given a list of curated papers, cluster them into 2-3 coarse sub-fields
and write a <=80 char Chinese summary of what's trending. Return ONLY valid JSON:

{"top": [{"name": "子领域名", "delta": <int: paper count>,
          "summary": "<=80字趋势总结",
          "papers": ["paper title 1", "paper title 2"]}, ...]}"""

def synthesize_trends(curated: list[dict]) -> dict:
    if len(curated) < 3:
        return {"top": []}

    slim = [{"topic": p.get("topic", ""), "title": p.get("title", ""),
             "tldr": p.get("tldr", "")} for p in curated]
    user = json.dumps(slim, ensure_ascii=False, indent=2)
    raw = _chat(TRENDS_SYSTEM, user, temperature=0.4, max_tokens=1024)
    if not raw:
        return {"top": []}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"top": []}
