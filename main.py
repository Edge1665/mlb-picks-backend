# main.py
from __future__ import annotations

import os
import math
import asyncio
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

# ===================== ENV / CONFIG =====================

BACKEND_NAME = "mlb-picks-backend"
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()
ALLOWED_ORIGINS = [o for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",") if o]

# ===================== APP ==============================

app = FastAPI(title=BACKEND_NAME, version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== MODEL IMPORT =====================

MODEL_OK = True
MODEL_IMPORT_ERROR = None
try:
    from feature_builder import build_today_features  # uses per-PA rates -> per-game probs
except Exception as e:
    MODEL_OK = False
    MODEL_IMPORT_ERROR = str(e)

# ===================== UTILS ============================

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
    return abs(o) / (abs(o) + 100)

def prob_to_american(p: float) -> int:
    p = clamp_prob(p)
    if p >= 0.5:
        return -int(round(100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))

def normalize_name(s: str) -> str:
    return " ".join("".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace()).split())

def looks_like_over(label_lower: str) -> bool:
    return (" over" in label_lower) or label_lower.strip().startswith("over ")

def choose_best(current: Optional[Dict[str, Any]], cand: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer a FanDuel line if present
    if current is None:
        return cand
    if (current.get("book") or "").lower() != "fanduel" and (cand.get("book") or "").lower() == "fanduel":
        return cand
    return current

async def _http_get_json(client: httpx.AsyncClient, url: str, params: dict | None = None):
    try:
        r = await client.get(url, params=params, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# ===================== LINEUPS / ROSTER =================

async def fetch_lineups_for_today(date_iso: str) -> List[Dict[str, Any]]:
    """
    Placeholder for posted lineups. Return [] so we fall back to rosters.
    You can replace this later with a real lineup source.
    """
    return []

async def fetch_team_rosters_for_today(date_iso: str) -> List[Dict[str, Any]]:
    """
    Roster fallback: active rosters for all MLB teams via StatsAPI (free).
    Returns: {playerId, playerName, team, lineupSpot=None}
    """
    out: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient() as client:
            teams_r = await client.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=20)
            teams_r.raise_for_status()
            teams = (teams_r.json() or {}).get("teams", []) or []

            for t in teams:
                tid = t.get("id")
                abbr = (t.get("abbreviation") or "").upper()
                if not tid:
                    continue

                roster_r = await client.get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster", timeout=20)
                if roster_r.status_code != 200:
                    continue
                for slot in (roster_r.json() or {}).get("roster", []) or []:
                    person = (slot or {}).get("person") or {}
                    pid = person.get("id")
                    pname = person.get("fullName") or ""
                    if not pid or not pname:
                        continue
                    out.append({
                        "playerId": int(pid),
                        "playerName": pname,
                        "team": abbr,
                        "lineupSpot": None,  # unknown until lineups post
                    })
    except Exception:
        pass
    return out

# ===================== RECENT RATES (FREE, S2D) =========

async def fetch_recent_rates(players: List[Dict[str, Any]], date_iso: str, window_days: int = 30) -> Dict[int, Dict[str, Any]]:
    """
    Fetch *season-to-date* per-PA rates from MLB StatsAPI (free) so players differ:
      hr_rate_rolling := HR / PA
      hit_rate_rolling := H / PA
      recent_pa := PA (S2D)
    Fallbacks: 0.02 HR/PA, 0.24 Hit/PA, 0 PA.
    """
    try:
        season = int(date_iso[:4])
    except Exception:
        season = dt.date.today().year

    base_url = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    sem = asyncio.Semaphore(12)

    async def fetch_one(pid: int):
        url = base_url.format(pid=pid)
        params = {"stats": "season", "group": "hitting", "season": str(season), "gameType": "R"}
        async with sem:
            async with httpx.AsyncClient() as client:
                data = await _http_get_json(client, url, params)

        hr_rate, hit_rate, pa = 0.02, 0.24, 0
        try:
            splits = (((data or {}).get("stats") or [])[0].get("splits") or [])
            if splits:
                stat = splits[0].get("stat") or {}
                pa = int(stat.get("plateAppearances") or 0)
                hr = int(stat.get("homeRuns") or 0)
                h = int(stat.get("hits") or 0)
                if pa > 0:
                    hr_rate = max(0.0005, min(0.9995, hr / pa))
                    hit_rate = max(0.0005, min(0.9995, h / pa))
        except Exception:
            pass
        return pid, {"hr_rate_rolling": hr_rate, "hit_rate_rolling": hit_rate, "recent_pa": pa}

    tasks = [fetch_one(int(p["playerId"])) for p in players if p.get("playerId")]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    rates: Dict[int, Dict[str, Any]] = {}
    for res in results:
        if isinstance(res, Exception) or not res:
            continue
        pid, row = res
        rates[int(pid)] = row

    for p in players:
        pid = int(p["playerId"])
        if pid not in rates:
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

# ===================== ODDS (robust parsing) ============

async def fetch_market_odds(date_iso: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    {(normalized_player_name, market): {'odds': int, 'book': str}}
    Markets:
      'HR' = HR Anytime
      'H1' = 1+ Hit (Over 0.5)
      'H2' = 2+ Hits (Over 1.5)
    Returns {} if no API key or no data.
    """
    if not ODDS_API_KEY:
        return {}

    sport_key = "baseball_mlb"
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
                    for oc in mk.get("outcomes") or []:
                        raw = (oc.get("description") or oc.get("name") or "").strip()
                        if not raw:
                            continue
                        name_l = raw.lower()
                        odds = oc.get("price")
                        if odds is None:
                            continue

                        market_norm: Optional[str] = None
                        pl_name: Optional[str] = None

                        # HR
                        if mk_key in ("player_home_runs", "player_home_run") or "home run" in name_l:
                            market_norm = "HR"
                            pl_name = raw
                            for frag in [" to hit a home run", " - hr", " hr"]:
                                pl_name = pl_name.replace(frag, "")

                        # Hits variants
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
            break

    return out

# ===================== SCORING ==========================

def pick_score(model_prob: float, market_prob: Optional[float], recent_pa: int) -> float:
    """
    1.0..10.0 (step 0.1). Blends model prob, market edge (if any), and confidence via recent PA.
    """
    p = clamp_prob(model_prob)
    conf = min(1.0, max(0.0, recent_pa / 30.0))  # saturate ~30 PA
    base = 1.0 + 7.0 * p          # 1..8
    edge = 0.0
    if market_prob is not None:
        edge = max(-0.2, min(0.2, (p - market_prob))) * 10.0  # -2..+2
    score = base + edge + 1.0 * conf
    return round(max(1.0, min(10.0, score)), 1)

# ===================== ROUTES ===========================

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
        "note": "If tot_props=0, your key/plan/region likely lacks player props. Model works; edges will be null.",
    }

@app.get("/markets")
async def markets(date: str = Query(default=None)):
    """
    Returns an array of player rows for the chosen date (YYYY-MM-DD).

    Model fields:
      - hr_prob_pa_model, hit_prob_pa_model (per-PA)
      - n_pa_est (estimated PAs used)
      - hr_anytime_prob, hits_1plus_prob, hits_2plus_prob (per-game)

    Market fields (if provider returns props):
      - hr_market_odds, h1_market_odds, h2_market_odds (American)
      - hr_market_prob, h1_market_prob, h2_market_prob
      - fair_hr_american, fair_h1_american, fair_h2_american
      - hr_edge, h1_edge, h2_edge
      - hr_score, h1_score, h2_score
    """
    if not date:
        date = dt.date.today().isoformat()

    # 1) Lineups if posted
    players = await fetch_lineups_for_today(date)

    # 2) Roster fallback
    if not players:
        players = await fetch_team_rosters_for_today(date)
    if not players:
        return []

    # 3) Season-to-date rates (free) so values vary
    rates = await fetch_recent_rates(players, date, window_days=30)
    players = apply_recent_rates(players, rates)

    # 4) Build per-game probabilities from per-PA rates
    if not MODEL_OK:
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

    # 5) Odds (may be empty if plan lacks player props)
    odds_map = await fetch_market_odds(date)  # {(norm_name, market): {odds, book}}

    rows: List[Dict[str, Any]] = []
    for row in modeled:
        name_key = normalize_name(row["playerName"])

        p_hr = float(row.get("hr_anytime_prob") or 0.0)
        p_h1 = float(row.get("hits_1plus_prob") or 0.0)
        p_h2 = float(row.get("hits_2plus_prob") or 0.0)

        fair_hr = prob_to_american(p_hr)
        fair_h1 = prob_to_american(p_h1)
        fair_h2 = prob_to_american(p_h2)

        hr_odds = (odds_map.get((name_key, "HR")) or {}).get("odds")
        h1_odds = (odds_map.get((name_key, "H1")) or {}).get("odds")
        h2_odds = (odds_map.get((name_key, "H2")) or {}).get("odds")

        hr_mkt_p = american_to_prob(hr_odds)
        h1_mkt_p = american_to_prob(h1_odds)
        h2_mkt_p = american_to_prob(h2_odds)

        hr_edge = (p_hr - hr_mkt_p) if hr_mkt_p is not None else None
        h1_edge = (p_h1 - h1_mkt_p) if h1_mkt_p is not None else None
        h2_edge = (p_h2 - h2_mkt_p) if h2_mkt_p is not None else None

        recent_pa = int(row.get("recent_pa") or 0)
        hr_score = pick_score(p_hr, hr_mkt_p, recent_pa)
        h1_score = pick_score(p_h1, h1_mkt_p, recent_pa)
        h2_score = pick_score(p_h2, h2_mkt_p, recent_pa)

        rows.append({
            **row,

            # Clear game-level aliases for the frontend (use these in UI)
            "hr_game_prob": p_hr,
            "h1_game_prob": p_h1,   # 1+ hit
            "h2_game_prob": p_h2,   # 2+ hits

            # Fair odds from the model
            "fair_hr_american": fair_hr,
            "fair_h1_american": fair_h1,
            "fair_h2_american": fair_h2,

            # Market odds, if present
            "hr_market_odds": hr_odds if hr_odds is not None else None,
            "h1_market_odds": h1_odds if h1_odds is not None else None,
            "h2_market_odds": h2_odds if h2_odds is not None else None,

            # Market implied probabilities (may be null if no odds returned)
            "hr_market_prob": hr_mkt_p,
            "h1_market_prob": h1_mkt_p,
            "h2_market_prob": h2_mkt_p,

            # Edges (model minus market), may be null if no market
            "hr_edge": hr_edge,
            "h1_edge": h1_edge,
            "h2_edge": h2_edge,

            # 1.0–10.0 pick scores
            "hr_score": hr_score,
            "h1_score": h1_score,
            "h2_score": h2_score,
        })


    return rows

