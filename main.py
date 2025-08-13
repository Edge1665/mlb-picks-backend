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
from fastapi.responses import JSONResponse

# ===================== ENV / CONFIG =====================

BACKEND_NAME = "mlb-picks-backend"
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()
ALLOWED_ORIGINS = [o for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",") if o]

# Simple park factors (HR & Hit). 1.00 = neutral. (Defaults apply when missing.)
PARK_FACTORS: Dict[int, Dict[str, float]] = {
    # venue_id: {"hr": ..., "hit": ...}
    3309: {"hr": 1.08, "hit": 1.02},  # Yankee Stadium
    2395: {"hr": 1.03, "hit": 1.00},  # Fenway Park
    2681: {"hr": 0.92, "hit": 0.98},  # T-Mobile Park
    2680: {"hr": 0.95, "hit": 0.99},  # Oracle Park
    4169: {"hr": 1.10, "hit": 1.03},  # Citizens Bank Park
    4034: {"hr": 1.04, "hit": 1.01},  # Busch Stadium
    4919: {"hr": 1.11, "hit": 1.03},  # Great American Ball Park
    2889: {"hr": 0.90, "hit": 0.97},  # Petco Park
    3313: {"hr": 1.05, "hit": 1.02},  # Dodger Stadium
    3839: {"hr": 1.06, "hit": 1.02},  # Wrigley Field
    # add more over time as needed…
}

# ===================== APP ==============================

app = FastAPI(title=BACKEND_NAME, version="1.4.0")

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
    from feature_builder import build_today_features
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
    if current is None:
        return cand
    if (current.get("book") or "").lower() != "fanduel" and (cand.get("book") or "").lower() == "fanduel":
        return cand
    return current

def innings_str_to_float(ip_str: str) -> float:
    """
    Convert MLB innings string like '123.2' to float innings:
    .0 -> +0.0; .1 -> +1/3; .2 -> +2/3
    """
    try:
        if "." not in ip_str:
            return float(int(ip_str))
        a, b = ip_str.split(".")
        base = float(int(a))
        frac = 0.0
        if b == "1":
            frac = 1.0 / 3.0
        elif b == "2":
            frac = 2.0 / 3.0
        return base + frac
    except Exception:
        return 0.0

# ===================== LINEUPS / ROSTER =================

async def fetch_lineups_for_today(date_iso: str) -> List[Dict[str, Any]]:
    # Placeholder (return [] to force roster fallback until you add a lineup source)
    return []

async def fetch_team_rosters_for_today(date_iso: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=25)
            r.raise_for_status()
            teams = (r.json() or {}).get("teams", []) or []
            for t in teams:
                tid = t.get("id")
                abbr = (t.get("abbreviation") or "").upper()
                if not tid:
                    continue
                rr = await client.get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster", timeout=25)
                if rr.status_code != 200:
                    continue
                for slot in (rr.json() or {}).get("roster", []) or []:
                    person = (slot or {}).get("person") or {}
                    pid = person.get("id")
                    pname = person.get("fullName") or ""
                    if not pid or not pname:
                        continue
                    out.append({
                        "playerId": int(pid),
                        "playerName": pname,
                        "team": abbr,
                        "lineupSpot": None,
                    })
    except Exception:
        pass
    return out

# ===================== MATCHUPS (pitcher + park) =========

async def fetch_schedule_matchups(date_iso: str) -> Dict[str, Dict[str, Any]]:
    """
    Returns map keyed by TEAM_ABBR to:
      {
        'opp_pitcher_id': int | None,
        'venue_id': int | None
      }
    for both away and home teams for all games on date.
    """
    out: Dict[str, Dict[str, Any]] = {}
    headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get(
                f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_iso}&hydrate=probablePitcher,team,venue",
                timeout=25
            )
            if r.status_code != 200:
                return out
            dates = (r.json() or {}).get("dates", []) or []
            for d in dates:
                for g in d.get("games", []) or []:
                    venue = (g.get("venue") or {}).get("id")
                    away = (g.get("teams", {}).get("away", {}) or {})
                    home = (g.get("teams", {}).get("home", {}) or {})
                    for side in ("away", "home"):
                        part = g.get("teams", {}).get(side, {}) or {}
                        team = (part.get("team") or {})
                        abbr = (team.get("abbreviation") or "").upper()
                        pitcher = (part.get("probablePitcher") or {})
                        pid = pitcher.get("id")
                        if abbr:
                            out[abbr] = {"opp_pitcher_id": None, "venue_id": venue}
                    # link opposing pitcher id per team
                    away_abbr = (away.get("team") or {}).get("abbreviation") or ""
                    home_abbr = (home.get("team") or {}).get("abbreviation") or ""
                    away_pitch = (home.get("probablePitcher") or {}).get("id")  # away faces home pitcher
                    home_pitch = (away.get("probablePitcher") or {}).get("id")  # home faces away pitcher
                    if away_abbr:
                        out[away_abbr.upper()]["opp_pitcher_id"] = away_pitch
                    if home_abbr:
                        out[home_abbr.upper()]["opp_pitcher_id"] = home_pitch
    except Exception:
        pass
    return out

async def fetch_pitcher_rates(pitcher_ids: List[int], date_iso: str) -> Dict[int, Dict[str, float]]:
    """
    For each pitcher id, compute {hr9, h9}. Uses season pitching stats.
    """
    result: Dict[int, Dict[str, float]] = {}
    if not pitcher_ids:
        return result
    try:
        season = int(date_iso[:4])
    except Exception:
        season = dt.date.today().year

    headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}
    async with httpx.AsyncClient(headers=headers) as client:
        # batch hydrate
        ids_csv = ",".join(str(x) for x in pitcher_ids)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_csv}&hydrate=stats(group=[pitching],type=[season],season={season})"
        r = await client.get(url, timeout=30)
        if r.status_code != 200:
            return result
        people = (r.json() or {}).get("people", []) or []
        for p in people:
            pid = p.get("id")
            hr9, h9 = 1.1, 8.1  # mild league-ish defaults
            try:
                stats = (p.get("stats") or [])
                if stats:
                    splits = (stats[0].get("splits") or [])
                    if splits:
                        st = splits[0].get("stat") or {}
                        ip = innings_str_to_float(str(st.get("inningsPitched", "0")))
                        hr = float(st.get("homeRuns", 0) or 0)
                        hits = float(st.get("hits", 0) or 0)
                        if ip > 0:
                            hr9 = 9.0 * (hr / ip)
                            h9 = 9.0 * (hits / ip)
            except Exception:
                pass
            if pid:
                result[int(pid)] = {"hr9": hr9, "h9": h9}
    return result

def park_multiplier(venue_id: Optional[int]) -> Tuple[float, float]:
    if not venue_id:
        return 1.00, 1.00
    pf = PARK_FACTORS.get(int(venue_id), {"hr": 1.00, "hit": 1.00})
    return float(pf.get("hr", 1.00)), float(pf.get("hit", 1.00))

def pitcher_multiplier(hr9: float, h9: float) -> Tuple[float, float]:
    # Compare to league-ish baselines; clamp to 0.7..1.3 range
    league_hr9 = 1.10
    league_h9 = 8.30
    m_hr = max(0.7, min(1.3, hr9 / league_hr9))
    m_hit = max(0.7, min(1.3, h9 / league_h9))
    return m_hr, m_hit

# ===================== RECENT RATES (FREE, S2D) =========

async def fetch_recent_rates(players: List[Dict[str, Any]], date_iso: str, window_days: int = 30) -> Dict[int, Dict[str, Any]]:
    # season S2D per-PA from batting
    try:
        season = int(date_iso[:4])
    except Exception:
        season = dt.date.today().year

    base_url = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}

    async def fetch_one(client: httpx.AsyncClient, pid: int):
        params = {"stats": "season", "group": "hitting", "season": str(season), "gameType": "R"}
        try:
            r = await client.get(base_url.format(pid=pid), params=params, timeout=25)
            if r.status_code != 200:
                data = None
            else:
                data = r.json()
        except Exception:
            data = None

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

    rates: Dict[int, Dict[str, Any]] = {}
    seen: set[int] = set()
    async with httpx.AsyncClient(headers=headers) as client:
        # modest concurrency
        sem = asyncio.Semaphore(16)
        async def runner(pid: int):
            async with sem:
                return await fetch_one(client, pid)
        tasks = []
        for p in players:
            pid = int(p.get("playerId") or 0)
            if pid and pid not in seen:
                seen.add(pid)
                tasks.append(asyncio.create_task(runner(pid)))
        results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception) or not res:
            continue
        pid, row = res
        rates[int(pid)] = row

    # ensure all present
    for
