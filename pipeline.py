#!/usr/bin/env python3
"""Cloud paper-collecter pipeline — runs end-to-end without human intervention.
Designed for GitHub Actions cron job.

Steps:
  1. Expand queries via LLM (DeepSeek) → state/queries.json
  2. Fetch candidates from all enabled sources     → state/candidates.json
  3. Filter + Chinese summaries via LLM            → state/curated.json
  4. Hot-topic synthesis via LLM (optional)        → state/trends.json
  5. Render Markdown + HTML digests                → digests/YYYY-MM-DD.{md,html}
  6. Push to WeChat / email / Telegram             (reads env vars)

Usage:
  python pipeline.py
  python pipeline.py --fetch-only      # just fetch, no LLM
  python pipeline.py --no-llm          # skip AI steps (use raw candidates)

Env vars needed:
  AI_BASE_URL, AI_API_KEY, AI_MODEL     — for LLM steps
  SERVERCHAN_KEY or WECHAT_WEBHOOK      — for push notifications
"""

import os
import sys
import json
import argparse

# Ensure we can import from scripts/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "scripts"))

import common as C
from llm import expand_queries, filter_and_summarize, synthesize_trends


def ensure_dirs():
    os.makedirs(C.STATE, exist_ok=True)
    os.makedirs(C.DIGESTS, exist_ok=True)


def step_expand(cfg: dict, use_llm: bool = True) -> dict:
    """Step 1: Read keywords from config, expand via LLM, write queries.json."""
    keywords = cfg.get("keywords", [])
    domain = cfg.get("domain", "physics")

    if use_llm:
        print("[pipeline] Step 1: Expanding queries via LLM...")
        queries = {}
        for kw in keywords:
            expanded = expand_queries(kw, domain)
            queries[kw] = expanded
            print(f"  {kw} → {expanded}")
    else:
        print("[pipeline] Step 1: Using raw keywords (--no-llm)")
        queries = {kw: [kw] for kw in keywords}

    C.save_json(os.path.join(C.STATE, "queries.json"), queries)
    return queries


def step_fetch(cfg: dict):
    """Step 2: Fetch candidates from all enabled sources. Runs fetch.py logic."""
    print("[pipeline] Step 2: Fetching candidates...")

    # Import fetch module dynamically
    import importlib
    fetch = importlib.import_module("fetch")

    enabled = cfg.get("sources", {})
    n = cfg.get("max_per_source", 15)
    lookback = cfg.get("lookback_days", 5)
    feeds = cfg.get("rss_feeds", [])

    queries = C.load_json(os.path.join(C.STATE, "queries.json"), None)
    if not queries:
        queries = {kw: [kw] for kw in cfg.get("keywords", [])}

    seen = set(C.load_json(os.path.join(C.STATE, "seen.json"), []))
    by_key, candidates = {}, []

    for kw, qlist in queries.items():
        for q in qlist:
            for name, fn in fetch.SOURCES.items():
                if enabled.get(name, True):
                    for it in fn(q, n):
                        it["topic"] = kw
                        if not fetch._recent(it.get("_d"), lookback):
                            continue
                        k = C.dedup_key(it)
                        if k in by_key or k in seen:
                            continue
                        by_key[k] = it
            if enabled.get("RSS", True) and feeds:
                for it in fetch.src_rss(q, feeds, n):
                    it["topic"] = kw
                    k = C.dedup_key(it)
                    if k not in by_key and k not in seen:
                        by_key[k] = it

    candidates = list(by_key.values())
    for it in candidates:
        it.pop("_d", None)
    candidates.sort(key=lambda x: x.get("published", ""), reverse=True)

    C.save_json(os.path.join(C.STATE, "candidates.json"), candidates)
    print(f"[pipeline] {len(candidates)} new candidates → state/candidates.json")
    return candidates


def step_curate(candidates: list, use_llm: bool = True) -> list:
    """Step 3 + 4: Filter, summarize, synthesize trends."""
    if not candidates:
        print("[pipeline] Step 3: No candidates — skipping curation.")
        C.save_json(os.path.join(C.STATE, "curated.json"), [])
        C.save_json(os.path.join(C.STATE, "trends.json"), {"top": []})
        return []

    if use_llm:
        print(f"[pipeline] Step 3: Filtering & summarizing {len(candidates)} candidates via LLM...")
        curated = filter_and_summarize(candidates)
    else:
        print("[pipeline] Step 3: Using raw candidates (--no-llm)")
        curated = [{**c, "tldr": (c.get("abstract", "") or "")[:80],
                    "method": "", "contributions": []} for c in candidates]

    print(f"[pipeline] Kept {len(curated)} / {len(candidates)} candidates")
    C.save_json(os.path.join(C.STATE, "curated.json"), curated)

    if use_llm and curated:
        print("[pipeline] Step 4: Synthesizing hot topics...")
        trends = synthesize_trends(curated)
    else:
        trends = {"top": []}

    C.save_json(os.path.join(C.STATE, "trends.json"), trends)
    return curated


def step_render():
    """Step 5: Render digest markdown + HTML."""
    print("[pipeline] Step 5: Rendering digest...")
    import importlib
    render = importlib.import_module("render")
    render.main()


def step_notify():
    """Step 6: Push to configured channels."""
    print("[pipeline] Step 6: Sending notifications...")
    import importlib
    notify = importlib.import_module("notify")
    notify.main()


def main():
    parser = argparse.ArgumentParser(description="Cloud paper-collecter pipeline")
    parser.add_argument("--fetch-only", action="store_true",
                        help="Only fetch candidates, skip curation & render")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM steps (use raw queries & candidates)")
    parser.add_argument("--skip-notify", action="store_true",
                        help="Skip push notifications")
    args = parser.parse_args()

    ensure_dirs()
    cfg = C.config()
    use_llm = not args.no_llm

    # Step 1: Expand queries
    step_expand(cfg, use_llm=use_llm)

    # Step 2: Fetch
    candidates = step_fetch(cfg)

    if args.fetch_only:
        print("[pipeline] --fetch-only: stopping after fetch.")
        return

    if not candidates:
        print("[pipeline] 暂无新文献 — nothing to render or push.")
        C.save_json(os.path.join(C.STATE, "curated.json"), [])
        C.save_json(os.path.join(C.STATE, "trends.json"), {"top": []})
        return

    # Step 3 + 4: Curate
    step_curate(candidates, use_llm=use_llm)

    # Step 5: Render
    step_render()

    # Step 6: Notify
    if not args.skip_notify:
        step_notify()

    print(f"[pipeline] ✅ Done. Digest at: digests/{C.today()}.html")


if __name__ == "__main__":
    main()
