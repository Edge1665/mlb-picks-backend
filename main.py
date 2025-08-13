# main.py
from __future__ import annotations

import os
import math
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ==== ENV / CONFIG ===========================================================

BACKEND_NAME = "mlb-picks-backend"
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()
ALLOWED_ORIGINS = [o for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",") if o]

# ==== APP ====================================================================

app = FastAPI(title=BACKEND_NAME, version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== IMPORT THE MODEL FEATURE BUILDER =======================================

MODEL_OK = True
MODEL_IMPORT_ERROR = None
try:
    from feature_builder import build_today_features  # builds per-game probs from per-PA rates
except Exception as e:  # pragma: no cover
    MODEL_OK = False
    MODEL_IMPORT_ERROR = str(e)

# ==== SMALL UTILS ============================================================

def clamp_prob(p: Optional[float]) -> float:
    if p is None: 
        return 0.0
    return max(0.0005, min(0.9995, float(p)))

def american_to_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None: 
        return None
    o = int(odds)
    if o > 0:
        return 100 / (o + 100)
    else:
        return abs(o) / (abs(o) + 100)

def prob_to_american(p: float) -> int:
    p = clamp_prob(p)
    if p >= 0.5:
        return -int(round(100 * p / (1 - p)))
    else:
        return int(round(100 * (1 - p) / p))

def normalize_name(s: str) -> str:
    return " ".join("".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace()).split())

def looks_like_over(label_lower: str) -> bool:
    # crude detector for "Over" outcomes
    return (" over" in label_lower) or label_lower.strip().startswith("over ")

def choose_best(current: Optional[Dict[str, Any]], cand: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prefer a FanDuel line if available; otherwise keep the first seen.
    """
    if current is None:
        return cand
    cur_book = (current.get("book") or "").lower()
    new_book = (cand.get("book") or "").lower()
    if cur_book != "fanduel" and new_book == "fanduel":
        return cand
    return current

# ==== LINEUPS + RECENT RATES =================================================
# These helpers fetch starters (lineups if posted; else roster fallback)
# and per-PA rolling rates (last 30d). They’re intentionally simple and
# designed to work with public/free sources.

async def fetch_lineups_for_today(date_iso: str) -> List[Dict[str, Any]]:
    """
    Try to fetch posted lineups. If not posted, return [] (so we can roster-fallback).
    Output rows: {playerId, playerName, team, lineupSpot?}
    """
    # Minimal, safe default: return [] so we trigger roster fallback.
    # If you had a working lineup fetch before, feel free to replace this with your prior function.
    return []

async def fetch_team_rosters_for_today(date_iso: str) -> List[Dict[str, Any]]:
    """
    Roster fallback: pull 40-man or active rosters for all teams, mapped to a neutral shape.
    We use the MLB StatsAPI public endpoint with very light parsing.
    """
    out: List[Dict[str, Any]] = []
    try:
        # 1) teams
        async with httpx.AsyncClient() as client:
            teams_r = await client.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=20)
            teams_r.raise_for_status()
            teams = (teams_r.json() or {}).get("teams", []) or []

            for t in teams:
                tid = t.get("id")
                abbr = (t.get("abbreviation") or "").upper()
                if not tid:
                    continue

                # 2) roster (active)
                roster_r = await client.get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster", timeout=20)
                if roster_r.status_code != 200:
                    continue
                roster = (roster_r.json() or {}).get("roster", []) or []
                for slot in roster:
                    person = (slot or {}).get("person") or {}
                    pid = person.get("id")
                    pname = person.get("fullName") or ""
                    if not pid or not pname:
                        continue
                    out.append({
                        "playerId": int(pid),
                        "playerName": pname,
                        "team": abbr,
                        "lineupSpot": None,  # unknown without posted lineup
                    })
    except Exception:
        # If anything fails, we still return what we’ve collected (possibly empty)
        pass
    return out

async def fetch_recent_rates(players: List[Dict[str, Any]], date_iso: str, window_days: int = 30) -> Dict[int, Dict[str, Any]]:
    """
    Fetch last-30-day per-PA rates for each player. For simplicity and API budget,
    we query Baseball Savant's public CSV endpoints in aggregate would be ideal,
    but to keep it robust we’ll use a conservative default here:
    - hr_rate_rolling default 0.02
    - hit_rate_rolling default 0.24
    If you already had a working function here, keep that instead.
    """
    rates: Dict[int, Dict[str, Any]] = {}
    for p in players:
        pid = int(p["playerId"])
        rates[pid] = {"hr_rate_rolling": 0.02, "hit_rate_rolling": 0.24, "recent_pa": 0}
    return rates

def apply_recent_rates(players: List[Dict[str, Any]], rates: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in players:
        r = rates.get(int(p["playerId"]), {})
        q = dict(p)
        q["hr_rate_rolling"] = r.get("hr_rate_rolling", 0.02)
        q["hit_rate_rolling"] = r.get("hit_rate_rolling", 0.24)
        q["recent_pa"] = r.get("recent_pa", 0)
        out.append(q)
    return out

# ==== ODDS FETCH (robust, tries multiple market names) =======================

async def fetch_market_odds(date_iso: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Returns {(normalized_player_name, market): {'odds': int, 'book': str}}
    Markets we normalize:
      'HR' -> HR Anytime
      'H1' -> 1+ Hits (Over 0.5)
      'H2' -> 2+ Hits (Over 1.5)
    If no API key or no data, returns {}.
    """
    if not ODDS_API_KEY:
        return {}

    sport_key = "baseball_mlb"
    # Try multiple market key sets to maximize compatibility
    market_sets = [
        "player_home_runs,player_hits",
        "player_home_runs,player_hits_over_under",
        "player_home_run,player_total_hits,player_hits_over_under",
    ]

    async def try_once(markets_csv: str):
        url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params = {
            "regions": "us",
            "markets": markets_csv,
            "oddsFormat": "american",
            "dateFormat": "iso",
            "apiKey": ODDS_API_KEY,
        }
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(url, params=params, timeout=20)
                if r.status_code != 200:
                    return []
                return r.json() or []
        except Exception:
            return []

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for markets_csv in market_sets:
        events = await try_once(markets_csv)
        found_any = False

        for ev in events or []:
            for bk in ev.get("bookmakers", []) or []:
                book_key = (bk.get("key") or "book").lower()
                for mk in bk.get("markets", []) or []:
                    mk_key = (mk.get("key") or "").lower()
                    outcomes = mk.get("outcomes") or []
                    for oc in outcomes:
                        raw = (oc.get("description") or oc.get("name") or "").strip()
                        if not raw:
                            continue
                        name_l = raw.lower()
                        odds = oc.get("price")
                        if odds is None:
                            continue

                        market_norm: Optional[str] = None
                        pl_name: Optional[str] = None

                        # HR market
                        if mk_key in ("player_home_runs", "player_home_run") or "home run" in name_l:
                            market_norm = "HR"
                            pl_name = raw
                            for frag in [" to hit a home run", " - hr", " hr"]:
                                pl_name = pl_name.replace(frag, "")

                        # Hits markets
                        elif mk_key in ("player_hits", "player_total_hits", "player_hits_over_under"):
                            point = oc.get("point")
                            is_over = looks_like_over(name_l)
                            pl_name = raw
                            for frag in [" over", " under", " hits", " total hits"]:
                                pl_name = pl_name.replace(frag, "")
                            if point is not None and is_over:
                                try:
                                    pt = float(point)
                                except Exception:
                                    pt = None
                                if pt is not None:
                                    if 0.5 <= pt < 1.5:
                                        market_norm = "H1"
                                    elif 1.5 <= pt < 2.5:
                                        market_norm = "H2"

                        # “to record a hit”
                        elif "record a hit" in name_l:
                            market_norm = "H1"
                            pl_name = raw.replace(" to record a hit", "")

                        if not market_norm or not pl_name:
                            continue

                        key = (normalize_name(pl_name), market_norm)
                        cand = {"odds": int(odds), "book": str(book_key)}
                        out[key] = choose_best(out.get(key), cand)
                        found_any = True

        if found_any:
            break  # stop after first market set that yields usable props

    return out

# ==== PICK SCORING ===========================================================

def pick_score(model_prob: float, market_prob: Optional[float], recent_pa: int) -> float:
    """
    Score from 1.0 to 10.0 (step 0.1) blending:
      - model_prob (higher is better)
      - market edge if market is present (model_prob - market_prob)
      - confidence via recent_pa (0..30 saturating)
    """
    p = clamp_prob(model_prob)
    conf = min(1.0, max(0.0, recent_pa / 30.0))
    # Base from model prob: 1..8
    base = 1.0 + 7.0 * p
    # Edge component (if market exists): -1..+1 scaled to ~2 points
    edge = 0.0
    if market_prob is not None:
        edge = max(-0.2, min(0.2, (p - market_prob))) * 10.0  # -2..+2
    # Confidence adds up to +1
    score = base + edge + 1.0 * conf
    return round(max(1.0, min(10.0, score)), 1)

# ==== ROUTES =================================================================

@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": BACKEND_NAME,
        "model_missing": not MODEL_OK,
        "model_error": MODEL_IMPORT_ERROR,
        "has_odds_api_key": bool(ODDS_API_KEY),
        "time": dt.datetime.utcnow().isoformat() + "Z",
    }

@app.get("/odds_status")
async def odds_status():
    has_key = bool(ODDS_API_KEY)
    sample = await fetch_market_odds(dt.date.today().isoformat()) if has_key else {}
    h1 = sum(1 for ((_, m), _) in sample.items() if m == "H1")
    h2 = sum(1 for ((_, m), _) in sample.items() if m == "H2")
    hr = sum(1 for ((_, m), _) in sample.items() if m == "HR")
    return {
        "has_api_key": has_key,
        "tot_props": len(sample),
        "by_market": {"H1": h1, "H2": h2, "HR": hr},
        "note": "If tot_props=0, your key/plan may not include player props for your region/books.",
    }

@app.get("/markets")
async def markets(date: str = Query(default=None)):
    """
    Returns an array of player rows for the chosen date (YYYY-MM-DD).
    Fields:
      - hr_anytime_prob, hits_1plus_prob, hits_2plus_prob  (model)
      - hr_market_odds, h1_market_odds, h2_market_odds     (books)
      - fair_hr_american, fair_h1_american, fair_h2_american
      - hr_edge, h1_edge, h2_edge
      - hr_score, h1_score, h2_score
    """
    if not date:
        date = dt.date.today().isoformat()

    # 1) Try posted lineups
    players = await fetch_lineups_for_today(date)

    # 2) Roster fallback
    if not players:
        players = await fetch_team_rosters_for_today(date)

    if not players:
        return []  # nothing we can do

    # 3) Enrich with recent rates (last 30d)
    rates = await fetch_recent_rates(players, date, window_days=30)
    players = apply_recent_rates(players, rates)

    # 4) Build model per-game probabilities
    if not MODEL_OK:
        # Return a simple stub if model import failed
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
            "error": f"feature_builder import failed: {MODEL_IMPORT_ERROR}",
        }]

    modeled = build_today_features(date, players)

    # 5) Fetch market odds & merge
    odds_map = await fetch_market_odds(date)  # {(norm_name, market): {odds, book}}

    rows: List[Dict[str, Any]] = []
    for row in modeled:
        name_key = normalize_name(row["playerName"])

        # Model probabilities
        p_hr = float(row.get("hr_anytime_prob") or 0.0)
        p_h1 = float(row.get("hits_1plus_prob") or 0.0)
        p_h2 = float(row.get("hits_2plus_prob") or 0.0)

        # Fair odds (from model)
        fair_hr = prob_to_american(p_hr)
        fair_h1 = prob_to_american(p_h1)
        fair_h2 = prob_to_american(p_h2)

        # Market odds (if present)
        hr_odds = (odds_map.get((name_key, "HR")) or {}).get("odds")
        h1_odds = (odds_map.get((name_key, "H1")) or {}).get("odds")
        h2_odds = (odds_map.get((name_key, "H2")) or {}).get("odds")

        # Market implied probs
        hr_mkt_p = american_to_prob(hr_odds)
        h1_mkt_p = american_to_prob(h1_odds)
        h2_mkt_p = american_to_prob(h2_odds)

        # Edges
        hr_edge = (p_hr - hr_mkt_p) if hr_mkt_p is not None else None
        h1_edge = (p_h1 - h1_mkt_p) if h1_mkt_p is not None else None
        h2_edge = (p_h2 - h2_mkt_p) if h2_mkt_p is not None else None

        # Scores
        recent_pa = int(row.get("recent_pa") or 0)
        hr_score = pick_score(p_hr, hr_mkt_p, recent_pa)
        h1_score = pick_score(p_h1, h1_mkt_p, recent_pa)
        h2_score = pick_score(p_h2, h2_mkt_p, recent_pa)

        rows.append({
            **row,
            "fair_hr_american": fair_hr,
            "fair_h1_american": fair_h1,
            "fair_h2_american": fair_h2,

            "hr_market_odds": hr_odds if hr_odds is not None else None,
            "h1_market_odds": h1_odds if h1_odds is not None else None,
            "h2_market_odds": h2_odds if h2_odds is not None else None,

            "hr_market_prob": hr_mkt_p,
            "h1_market_prob": h1_mkt_p,
            "h2_market_prob": h2_mkt_p,

            "hr_edge": hr_edge,
            "h1_edge": h1_edge,
            "h2_edge": h2_edge,

            "hr_score": hr_score,
            "h1_score": h1_score,
            "h2_score": h2_score,
        })

    return rows

