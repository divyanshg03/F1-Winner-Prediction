"""Pull F1 race-winner prediction-market prices (Polymarket public APIs; no key needed).

Run this from a network that can reach polymarket.com (it is blocked on some ISPs):
    python scripts/fetch_market_odds.py
Writes data/raw/odds/polymarket_f1.csv with columns:
    event, market, outcome, token_id, end_date, ts_utc, price      (price = implied win probability, 0-1)
and prints a discovery summary. NOTE: written from the documented API shape and not yet run against the
live service; if a field name differs, the summary printout will show it. Send that output back.
"""
import csv
import json
import re
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "odds"
OUT.mkdir(parents=True, exist_ok=True)
GAMMA, CLOB = "https://gamma-api.polymarket.com", "https://clob.polymarket.com"
PAT = re.compile(r"grand prix|formula 1|\bf1\b", re.I)


def get(url, **params):
    for i in range(4):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(3 * (i + 1)); continue
        r.raise_for_status(); time.sleep(0.25); return r.json()
    raise RuntimeError(url)


def find_events():
    seen = {}
    for q in ["Formula 1 Grand Prix winner", "F1 Grand Prix winner", "Grand Prix", "F1 race winner"]:
        for status in ("closed", "active"):
            try:
                d = get(f"{GAMMA}/public-search", q=q, limit_per_type=100, events_status=status)
            except Exception as e:  # noqa: BLE001
                print("search failed", q, status, repr(e)[:80]); continue
            for e in d.get("events", []):
                if PAT.search(e.get("title", "")) and re.search(r"win", e.get("title", ""), re.I):
                    seen[e["id"]] = e
    return list(seen.values())


def main():
    events = find_events()
    print(f"{len(events)} candidate events")
    rows = []
    for e in events:
        for m in e.get("markets", []):
            try:
                tokens = json.loads(m["clobTokenIds"]) if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds") or []
            except Exception:  # noqa: BLE001
                continue
            if not tokens:
                continue
            name = m.get("groupItemTitle") or m.get("question")
            try:
                hist = get(f"{CLOB}/prices-history", market=tokens[0], interval="max", fidelity=60).get("history", [])
            except Exception as ex:  # noqa: BLE001
                print("  no history:", e.get("title"), name, repr(ex)[:60]); continue
            for h in hist:
                rows.append([e.get("title"), m.get("question"), name, tokens[0], str(e.get("endDate"))[:10], h["t"], h["p"]])
        print(f"  {e.get('title')} | end {str(e.get('endDate'))[:10]} | markets {len(e.get('markets', []))}")
    with open(OUT / "polymarket_f1.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f); w.writerow(["event", "market", "outcome", "token_id", "end_date", "ts_utc", "price"]); w.writerows(rows)
    print(f"wrote {len(rows)} price points -> {OUT / 'polymarket_f1.csv'}")


if __name__ == "__main__":
    main()
