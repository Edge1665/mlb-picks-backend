# main.py — real lineups + recent rates + optional sportsbook odds + scoring
from __future__ import annotations

import os
import math
import asyncio
import datetime as dt
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load your model feature builder (pandas-free version recommended)
MISSING_MODEL_ARTIFACTS = False
IMPORT_ERROR_DETAIL = ""
try:
    from feature_builder import build_today_features  # expects list[dict] -> list[dict]
except Exception as e:
    MISSING_MODEL_ARTIFACTS = True
    IMPORT_ERROR_DETAIL = str(e)

# -------- App & CORS --------
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app = FastAPI(title="MLB Picks API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Helpers: MLB schedule/lineups (no roster fallback) =========
# --- MLB StatsAPI lineup fetch (with roster fallback) ---
# Endpoints:
#   Schedule:  https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD
#   Boxscore:  https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore
#   Roster:    https://statsapi.mlb.com/api/v1/teams/{teamId}/roster?rosterType=active

async def _get_game_pks_for_date(client: httpx.AsyncClient, date_str: str) -> List[int]:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str}
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    pks: List[int] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            pk = g.get("gamePk")
            if pk:
                pks.append(int(pk))
    return pks

def _extract_lineup_from_boxscore_json(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for side in ("home", "away"):
        team_name = js.get("teams", {}).get(side, {}).get("team", {}).get("name", "")
        players = js.get("teams", {}).get(side, {}).get("players", {}) or {}
        for _, pl in players.items():
            bo = pl.get("battingOrder")
            if not bo:
                continue
            try:
                lineup_spot = int(bo) // 100
                if not (1 <= lineup_spot <= 9):
                    continue
            except Exception:
                continue
            person = pl.get("person", {}) or {}
            pid = person.get("id")
            name = person.get("fullName") or person.get("boxscoreName") or "Unknown"
            if not pid:
                continue
            out.append({
                "playerId": int(pid),
                "playerName": str(name),
                "team": team_name,
                "lineupSpot": lineup_spot
            })
    # dedupe
    seen = set(); dedup: List[Dict[str, Any]] = []
    for r in out:
        if r["playerId"] in seen: continue
        seen.add(r["playerId"]); dedup.append(r)
    return dedup

def _teams_from_boxscore(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    teams: List[Dict[str, Any]] = []
    for side in ("home", "away"):
        blk = js.get("teams", {}).get(side, {}) or {}
        team = blk.get("team", {}) or {}
        tid = team.get("id"); tname = team.get("name")
        if tid and tname:
            teams.append({"teamId": int(tid), "teamName": str(tname)})
    return teams

async def _fetch_active_roster(client: httpx.AsyncClient, team_id: int, team_name: str) -> List[Dict[str, Any]]:
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
    params = {"rosterType": "active"}
    r = await client.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return []
    out: List[Dict[str, Any]] = []
    try:
        data = r.json()
        for item in data.get("roster", []) or []:
            person = item.get("person", {}) or {}
            pid = person.get("id"); name = person.get("fullName") or "Unknown"
            if not pid: continue
            out.append({
                "playerId": int(pid),
                "playerName": str(name),
                "team": team_name,
                "lineupSpot": None  # unknown before lineups
            })
    except Exception:
        return []
    return out

async def _fetch_boxscore_with_fallback(client: httpx.AsyncClient, game_pk: int) -> List[Dict[str, Any]]:
    # Try posted lineups
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    r = await client.get(url, timeout=20)
    if r.status_code != 200:
        return []
    try:
        js = r.json()
    except Exception:
        return []

    lineup = _extract_lineup_from_boxscore_json(js)
    if lineup:
        return lineup

    # Fallback to active rosters (before lineups post)
    teams = _teams_from_boxscore(js)
    if not teams:
        return []
    res = await asyncio.gather(*(asyncio.create_task(_fetch_active_roster(client, t["teamId"], t["teamName"])) for t in teams), return_exceptions=True)
    players: List[Dict[str, Any]] = []
    for r in res:
        if isinstance(r, Exception): continue
        players.extend(r)

    # dedupe
    seen = set(); dedup: List[Dict[str, Any]] = []
    for r in players:
        if r["playerId"] in seen: continue
        seen.add(r["playerId"]); dedup.append(r)
    return dedup

async def fetch_lineups_for_today(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    if not date_str:
        date_str = dt.date.today().isoformat()
    async with httpx.AsyncClient() as client:
        pks = await _get_game_pks_for_date(client, date_str)
        if not pks: return []
        res = await asyncio.gather(*(asyncio.create_task(_fetch_boxscore_with_fallback(client, pk)) for pk in pks), return_exceptions=True)
    players: List[Dict[str, Any]] = []
    for r in res:
        if isinstance(r, Exception): continue
        players.extend(r)
    # final dedupe
    seen = set(); dedup: List[Dict[str, Any]] = []
    for r in players:
        if r["playerId"] in seen: continue
        seen.add(r["playerId"]); dedup.append(r)
    return dedup

# ========= Helpers: last-30-day hitting rates (per player) =========
async def fetch_recent_rates(players: List[dict], date_iso: str, window_days: int = 30) -> Dict[int, Dict[str, float | None]]:
    end_dt = dt.date.fromisoformat(date_iso)
    start_dt = end_dt - dt.timedelta(days=window_days)
    start, end = start_dt.isoformat(), end_dt.isoformat()
    ids = sorted({int(p["playerId"]) for p in players})

    async with httpx.AsyncClient() as client:
        async def one(pid: int):
            url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
            params = {"stats": "byDateRange", "group": "hitting", "startDate": start, "endDate": end}
            try:
                r = await client.get(url, params=params, timeout=15)
                r.raise_for_status()
                js = r.json()
                splits = (js.get("stats") or [{}])[0].get("splits") or []
                if not splits:
                    return pid, 0, 0, 0
                stat = splits[0].get("stat") or {}
                pa   = int(stat.get("plateAppearances") or 0)
                hits = int(stat.get("hits") or 0)
                hr   = int(stat.get("homeRuns") or 0)
                return pid, pa, hits, hr
            except Exception:
                return pid, 0, 0, 0

        results = await asyncio.gather(*(one(pid) for pid in ids))
    out: Dict[int, Dict[str, float | None]] = {}
    for pid, pa, hits, hr in results:
        hr_rate  = (hr/pa)   if pa > 0 else None
        hit_rate = (hits/pa) if pa > 0 else None
        out[pid] = {"pa": float(pa), "hr_rate": hr_rate, "hit_rate": hit_rate}
    return out

def apply_recent_rates(players: List[dict], rates: Dict[int, Dict[str, float | None]]) -> List[dict]:
    enriched = []
    for pl in players:
        pid = int(pl["playerId"])
        r = rates.get(pid, {})
        hr_rate  = r.get("hr_rate")
        hit_rate = r.get("hit_rate")
        if hr_rate is None:  hr_rate  = 0.02   # conservative fallback
        if hit_rate is None: hit_rate = 0.24
        pl2 = dict(pl)
        pl2["hr_rate_rolling"]  = float(hr_rate)
        pl2["hit_rate_rolling"] = float(hit_rate)
        pl2["recent_pa"] = int(r.get("pa") or 0)
        enriched.append(pl2)
    return enriched

# ========= Helpers: odds + math =========
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()  # optional

def american_to_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    try:
        o = int(odds)
    except Exception:
        return None
    if o > 0:
        return 100.0 / (o + 100.0)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return None

def prob_to_american(p: float) -> int:
    p = max(1e-6, min(0.999999, p))
    if p >= 0.5:
        return -int(round((p*100) / (1 - p)))
    else:
        return int(round(((1 - p) * 100) / p))

async def fetch_market_odds(date_iso: str) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """
    Optional odds fetch. Returns a mapping: (playerId, market) -> {'odds': int, 'book': str}
    Markets: 'HR', 'H1', 'H2'  (HR Anytime, 1+ Hits, 2+ Hits)
    If THE_ODDS_API_KEY not set, returns {} and we skip odds (app still works).
    """
    if not ODDS_API_KEY:
        return {}

    # Placeholder: many free tiers don't expose player props reliably.
    # This function is structured so you can add a real provider later.
    # For now, return empty dict so the rest of the pipeline works.
    return {}

def score_pick(model_prob: float, market_prob: Optional[float], recent_pa: int, market: str) -> Tuple[float, float]:
    """
    Returns (edge, score_1_to_10).
    edge = model_prob - market_prob (if market odds exist) else None-treated as 0 for score calc.
    Score uses edge * confidence, with different weights per market.
    """
    # Confidence from sample size (last 30 days); saturate at 30 PA
    conf = min(1.0, max(0.2, recent_pa / 30.0))  # floor to avoid zeroing early

    # Market-specific weight (HR is rarer → edges carry more)
    weight = {"HR": 0.40, "H1": 0.25, "H2": 0.35}.get(market, 0.30)

    if market_prob is None:
        edge = None
        eff_edge = 0.0
    else:
        edge = model_prob - market_prob
        eff_edge = edge

    # Base 5.0, add/subtract based on (edge% * weight * confidence)
    edge_pct = (eff_edge * 100.0)
    raw = 5.0 + (edge_pct * weight * conf) / 1.0
    score = max(1.0, min(10.0, raw))
    # round to 0.1 increments
    score = round(score * 10.0) / 10.0
    return (0.0 if edge is None else edge, score)

# ========= Build payload =========
async def compute_markets_payload(date: Optional[str] = None) -> List[Dict[str, Any]]:
    if date is None:
        date = dt.date.today().isoformat()

    if MISSING_MODEL_ARTIFACTS:
        return [{
            "date": date,
            "playerId": -1,
            "playerName": "Model not loaded",
            "team": "",
            "lineupSpot": None,
            "hr_prob_pa_model": 0.0,
            "hit_prob_pa_model": 0.0,
            "hr_anytime_prob": 0.0,
            "hits_1plus_prob": 0.0,
            "hits_2plus_prob": 0.0,
            "error": f"feature_builder import failed: {IMPORT_ERROR_DETAIL}",
        }]

    players = await fetch_lineups_for_today(date)
    if not players:
        return []  # lineups not posted yet

    # Enrich with last-30-day rates
    rates = await fetch_recent_rates(players, date, window_days=30)
    players = apply_recent_rates(players, rates)

    # Model predictions (per-PA -> per-game markets)
    preds = build_today_features(date, players)  # list[dict]

    # Optional sportsbook odds
    odds_map = await fetch_market_odds(date)  # {(playerId, 'HR'): {'odds': -105, 'book':'X'}, ...}

    # Build final records with edges & scores
    out: List[Dict[str, Any]] = []
    for rec in preds:
        pid = int(rec["playerId"])
        recent_pa = int(next((p.get("recent_pa", 0) for p in players if int(p["playerId"]) == pid), 0))

        # Markets: HR, H1 (1+ Hits), H2 (2+ Hits)
        hr_p  = float(rec["hr_anytime_prob"])
        h1_p  = float(rec["hits_1plus_prob"])
        h2_p  = float(rec["hits_2plus_prob"])

        hr_odds = odds_map.get((pid, "HR"), {}).get("odds")
        h1_odds = odds_map.get((pid, "H1"), {}).get("odds")
        h2_odds = odds_map.get((pid, "H2"), {}).get("odds")

        hr_market_p = american_to_prob(hr_odds)
        h1_market_p = american_to_prob(h1_odds)
        h2_market_p = american_to_prob(h2_odds)

        hr_edge, hr_score = score_pick(hr_p, hr_market_p, recent_pa, "HR")
        h1_edge, h1_score = score_pick(h1_p, h1_market_p, recent_pa, "H1")
        h2_edge, h2_score = score_pick(h2_p, h2_market_p, recent_pa, "H2")

        rec_out = {
            "date": rec["date"],
            "playerId": pid,
            "playerName": rec["playerName"],
            "team": rec["team"],
            "lineupSpot": rec.get("lineupSpot"),

            # Model probabilities
            "hr_anytime_prob": round(hr_p, 4),
            "hits_1plus_prob": round(h1_p, 4),
            "hits_2plus_prob": round(h2_p, 4),

            # Optional market odds (None if not available)
            "hr_market_odds": hr_odds,
            "h1_market_odds": h1_odds,
            "h2_market_odds": h2_odds,
            "hr_market_prob": None if hr_market_p is None else round(hr_market_p, 4),
            "h1_market_prob": None if h1_market_p is None else round(h1_market_p, 4),
            "h2_market_prob": None if h2_market_p is None else round(h2_market_p, 4),

            # Edge (model - market) where market exists
            "hr_edge": round(hr_edge, 4) if hr_market_p is not None else None,
            "h1_edge": round(h1_edge, 4) if h1_market_p is not None else None,
            "h2_edge": round(h2_edge, 4) if h2_market_p is not None else None,

            # Scores 1.0–10.0 (0.1 steps). Uses edge if market exists, else confidence-weighted baseline.
            "hr_score": hr_score,
            "h1_score": h1_score,
            "h2_score": h2_score,

            # Extras for UI/context
            "recent_pa": recent_pa,
            "fair_hr_american": prob_to_american(hr_p),
            "fair_h1_american": prob_to_american(h1_p),
            "fair_h2_american": prob_to_american(h2_p),
        }
        out.append(rec_out)

    return out

# ========= Routes =========
@app.get("/health")
async def health():
    return {"ok": True, "time": dt.datetime.utcnow().isoformat()}

@app.get("/markets")
async def markets(date: str | None = None):
    if not date:
        date = dt.date.today().isoformat()
    return await compute_markets_payload(date)
