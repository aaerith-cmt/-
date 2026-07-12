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
            if isinstance(blocks, str):
                text = blocks
            elif isinstance(blocks, list):
                # Skip thinking blocks, extract text blocks
                text = ""
                for b in blocks:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "text":
                        text += b.get("text", "")
                # Fallback: if no text block, try thinking block
                if not text:
                    for b in blocks:
                        if isinstance(b, dict) and b.get("type") == "thinking":
                            text += b.get("thinking", "") or b.get("text", "")
            else:
                text = str(blocks)
            print(f"[llm] anthropic parsed: {len(blocks) if isinstance(blocks, list) else 1} blocks, text_len={len(text)}")
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
    raw = _chat(EXPAND_SYSTEM, user, temperature=0.4, max_tokens=4096)
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

FILTER_SYSTEM = """你是凝聚态物理领域的中国研究员，正在策划每日文献简报。你会收到一个候选论文的JSON数组。对每篇文章：

1. 判断是否真正与物理学/凝聚态物理/量子科学相关。删除以下内容：区块链、金融科技、纯计算机科学、医学研究、与物理无关的GitHub仓库。

2. 对保留的文章，在阅读完整摘要后，用中文撰写详细摘要：
   - "tldr": 用一句话（<=80字）概括这篇论文的核心发现或创新点
   - "method": 简要描述研究方法或技术路线（<=100字），GitHub仓库留空
   - "contributions": 列出2-3个具体核心贡献（每条<=50字），要具体不要空洞，GitHub仓库留空

重要：不要只是翻译标题！要基于摘要内容提炼真正的科学贡献。
如果没有足够信息写method和contributions，可以基于摘要合理推断。

只返回有效JSON（不要markdown代码块，不要其他文字）——保留项目的数组：

[{
  "title": "保持原标题",
  "tldr": "用中文写的一句话说清楚论文做了什么",
  "method": "用了什么方法/技术",
  "contributions": ["贡献1", "贡献2"]
}, ...]

论文（arXiv、Crossref）排在前面，GitHub仓库排在最后。"""


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
    raw = _chat(FILTER_SYSTEM, user, temperature=0.3, max_tokens=8192)
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
    raw = _chat(TRENDS_SYSTEM, user, temperature=0.4, max_tokens=4096)
    if not raw:
        return {"top": []}
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"top": []}
