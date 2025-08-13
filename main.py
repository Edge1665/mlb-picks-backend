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

# ----------------- ENV -----------------
BACKEND_NAME = "mlb-picks-backend"
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()
ALLOWED_ORIGINS = [o for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",") if o]

# Park factors (mini table; default 1.00)
PARK_FACTORS: Dict[int, Dict[str, float]] = {
    3309: {"hr": 1.08, "hit": 1.02},  # Yankee Stadium
    2395: {"hr": 1.03, "hit": 1.00},  # Fenway
    2681: {"hr": 0.92, "hit": 0.98},  # T-Mobile
    2680: {"hr": 0.95, "hit": 0.99},  # Oracle
    4169: {"hr": 1.10, "hit": 1.03},  # Citizens Bank
    4034: {"hr": 1.04, "hit": 1.01},  # Busch
    4919: {"hr": 1.11, "hit": 1.03},  # Great American
    2889: {"hr": 0.90, "hit": 0.97},  # Petco
    3313: {"hr": 1.05, "hit": 1.02},  # Dodger
    3839: {"hr": 1.06, "hit": 1.02},  # Wrigley
}

# ----------------- APP -----------------
app = FastAPI(title=BACKEND_NAME, version="1.4.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- MODEL IMPORT -----------------
MODEL_OK = True
MODEL_IMPORT_ERROR = None
try:
    from feature_builder import build_today_features  # expects per-PA -> per-game logic
except Exception as e:
    MODEL_OK = False
    MODEL_IMPORT_ERROR = str(e)

# ----------------- HELPERS -----------------
def clamp_prob(p: Optional[float]) -> float:
    if p is None:
        return 0.0
    return max(0.0005, min(0.9995, float(p)))

def american_to_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None:
        return None
    o = int(odds)
    return 100 / (o + 100) if o > 0 else abs(o) / (abs(o) + 100)

def prob_to_american(p: float) -> int:
    p = clamp_prob(p)
    return -int(round(100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))

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
    try:
        if "." not in ip_str:
            return float(int(ip_str))
        a, b = ip_str.split(".")
        base = float(int(a))
        frac = 1.0/3.0 if b == "1" else (2.0/3.0 if b == "2" else 0.0)
        return base + frac
    except Exception:
        return 0.0

def park_multiplier(venue_id: Optional[int]) -> Tuple[float, float]:
    if not venue_id:
        return 1.00, 1.00
    pf = PARK_FACTORS.get(int(venue_id), {"hr": 1.00, "hit": 1.00})
    return float(pf.get("hr", 1.00)), float(pf.get("hit", 1.00))

def pitcher_multiplier(hr9: float, h9: float) -> Tuple[float, float]:
    league_hr9, league_h9 = 1.10, 8.30
    m_hr = max(0.7, min(1.3, hr9 / league_hr9))
    m_hit = max(0.7, min(1.3, h9 / league_h9))
    return m_hr, m_hit

def pick_score(model_prob: float, market_prob: Optional[float], recent_pa: int) -> float:
    p = clamp_prob(model_prob)
    conf = min(1.0, max(0.0, recent_pa / 30.0))
    base = 1.0 + 7.0 * p      # 1..8
    edge = (max(-0.2, min(0.2, (p - market_prob))) * 10.0) if market_prob is not None else 0.0
    return round(max(1.0, min(10.0, base + edge + 1.0 * conf)), 1)

# ----------------- DATA FETCH -----------------
async def fetch_lineups_for_today(date_iso: str) -> List[Dict[str, Any]]:
    # Stub until you wire a lineup API. Forces roster fallback.
    return []

async def fetch_team_rosters_for_today(date_iso: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=25)
            if r.status_code != 200:
                return out
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
                    out.append({"playerId": int(pid), "playerName": pname, "team": abbr, "lineupSpot": None})
    except Exception:
        pass
    return out

async def fetch_schedule_matchups(date_iso: str) -> Dict[str, Dict[str, Any]]:
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
            for d in (r.json() or {}).get("dates", []) or []:
                for g in d.get("games", []) or []:
                    venue = (g.get("venue") or {}).get("id")
                    away = (g.get("teams", {}).get("away", {}) or {})
                    home = (g.get("teams", {}).get("home", {}) or {})
                    away_abbr = (away.get("team") or {}).get("abbreviation") or ""
                    home_abbr = (home.get("team") or {}).get("abbreviation") or ""
                    away_pitch = (home.get("probablePitcher") or {}).get("id")  # away faces home pitcher
                    home_pitch = (away.get("probablePitcher") or {}).get("id")  # home faces away pitcher
                    if away_abbr:
                        out[away_abbr.upper()] = {"opp_pitcher_id": away_pitch, "venue_id": venue}
                    if home_abbr:
                        out[home_abbr.upper()] = {"opp_pitcher_id": home_pitch, "venue_id": venue}
    except Exception:
        pass
    return out

async def fetch_pitcher_rates(pitcher_ids: List[int], date_iso: str) -> Dict[int, Dict[str, float]]:
    res: Dict[int, Dict[str, float]] = {}
    if not pitcher_ids:
        return res
    try:
        season = int(date_iso[:4])
    except Exception:
        season = dt.date.today().year
    headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}
    ids_csv = ",".join(str(x) for x in pitcher_ids)
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_csv}&hydrate=stats(group=[pitching],type=[season],season={season})"
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get(url, timeout=30)
            if r.status_code != 200:
                return res
            for p in (r.json() or {}).get("people", []) or []:
                pid = p.get("id")
                hr9, h9 = 1.1, 8.1
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
                    res[int(pid)] = {"hr9": hr9, "h9": h9}
    except Exception:
        pass
    return res

async def fetch_recent_rates(players: List[Dict[str, Any]], date_iso: str, window_days: int = 30) -> Dict[int, Dict[str, Any]]:
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
            data = r.json() if r.status_code == 200 else None
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

async def fetch_market_odds(date_iso: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
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
        params = {"regions": "us", "markets": markets_csv, "oddsFormat": "american", "dateFormat": "iso", "apiKey": ODDS_API_KEY}
        headers = {"User-Agent": "mlb-picks/1.0 (+https://mlb-picks.app)"}
        try:
            async with httpx.AsyncClient(headers=headers) as client:
                r = await client.get(url, params=params, timeout=25)
                return r.json() if r.status_code == 200 else []
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
                        if mk_key in ("player_home_runs", "player_home_run") or "home run" in name_l:
                            market_norm = "HR"
                            pl_name = raw.replace(" to hit a home run", "").replace(" - hr", "").replace(" hr", "")
                        elif mk_key in ("player_hits", "player_total_hits", "player_hits_over_under"):
                            point = oc.get("point")
                            is_over = looks_like_over(name_l)
                            pl_name = raw.replace(" over", "").replace(" under", "").replace(" hits", "").replace(" total hits", "")
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

# ----------------- ROUTES -----------------
@app.get("/health")
async def health():
    return {
        "ok": True,
        "service": BACKEND_NAME,
        "version": "1.4.1",
        "model_missing": not MODEL_OK,
        "model_error": MODEL_IMPORT_ERROR,
        "has_odds_api_key": bool(ODDS_API_KEY),
        "time": dt.datetime.utcnow().isoformat() + "Z",
    }

@app.get("/matchup_status")
async def matchup_status(date: str | None = None):
    if not date:
        date = dt.date.today().isoformat()
    m = await fetch_schedule_matchups(date)
    pitchers = [v["opp_pitcher_id"] for v in m.values() if v.get("opp_pitcher_id")]
    uniq = sorted(set(int(x) for x in pitchers))
    rates = await fetch_pitcher_rates(uniq, date)
    return {"date": date, "teams": len(m), "with_pitcher": len(uniq), "pitcher_rates": len(rates)}

@app.get("/rates_status")
async def rates_status(date: str | None = None):
    if not date:
        date = dt.date.today().isoformat()
    players = await fetch_team_rosters_for_today(date)
    rates = await fetch_recent_rates(players, date, window_days=30)
    vals_hr = [r["hr_rate_rolling"] for r in rates.values()]
    vals_hit = [r["hit_rate_rolling"] for r in rates.values()]
    def summarize(xs: List[float]):
        if not xs:
            return {"min": None, "max": None, "avg": None, "n": 0}
        return {"min": min(xs), "max": max(xs), "avg": sum(xs)/len(xs), "n": len(xs)}
    return {"date": date, "players": len(players), "rates": {
        "hr_rate_rolling": summarize(vals_hr), "hit_rate_rolling": summarize(vals_hit)
    }}

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
        "note": "If tot_props=0, your key/plan/region likely lacks player props. Model still works; edges will be null.",
    }

@app.get("/debug_summary")
async def debug_summary(date: str | None = None):
    if not date:
        date = dt.date.today().isoformat()
    players = await fetch_team_rosters_for_today(date)
    rates = await fetch_recent_rates(players, date, window_days=30)
    enriched = apply_recent_rates(players, rates)
    if not MODEL_OK:
        return {"model_loaded": False, "error": MODEL_IMPORT_ERROR}
    modeled = build_today_features(date, enriched)
    def col(xs, key):
        vs = [float(x.get(key) or 0.0) for x in xs]
        return {"min": min(vs, default=None), "max": max(vs, default=None), "avg": (sum(vs)/len(vs) if vs else None), "n": len(vs)}
    return {
        "model_loaded": True, "date": date, "counts": len(modeled),
        "hr_pa": col(modeled, "hr_prob_pa_model"),
        "hit_pa": col(modeled, "hit_prob_pa_model"),
        "hr_game": col(modeled, "hr_anytime_prob"),
        "h1_game": col(modeled, "hits_1plus_prob"),
        "h2_game": col(modeled, "hits_2plus_prob"),
    }

@app.get("/markets")
async def markets(date: str = Query(default=None)):
    if not date:
        date = dt.date.today().isoformat()

    # 1) Lineups (stub) or 2) roster fallback
    players = await fetch_lineups_for_today(date)
    if not players:
        players = await fetch_team_rosters_for_today(date)
    if not players:
        return JSONResponse(content=[], headers={"Cache-Control": "no-store"})

    # 3) Season-to-date per-PA batter rates
    rates = await fetch_recent_rates(players, date, window_days=30)
    players = apply_recent_rates(players, rates)

    # 4) Matchups & park factors
    team_map = await fetch_schedule_matchups(date)
    pitcher_ids = sorted(set(v["opp_pitcher_id"] for v in team_map.values() if v.get("opp_pitcher_id")))
    pitch_rates = await fetch_pitcher_rates(pitcher_ids, date)

    # inject matchup multipliers onto each player row
    adj_players = []
    for p in players:
        team = (p.get("team") or "").upper()
        m = team_map.get(team, {})
        venue_id = m.get("venue_id")
        opp_pid = m.get("opp_pitcher_id")
        hr_pa = float(p.get("hr_rate_rolling") or 0.02)
        hit_pa = float(p.get("hit_rate_rolling") or 0.24)

        pf_hr, pf_hit = park_multiplier(venue_id)
        pm_hr, pm_hit = 1.0, 1.0
        if opp_pid and opp_pid in pitch_rates:
            pr = pitch_rates[opp_pid]
            pm_hr, pm_hit = pitcher_multiplier(pr.get("hr9", 1.1), pr.get("h9", 8.3))

        adj = dict(p)
        adj["hr_rate_rolling"] = clamp_prob(hr_pa * pf_hr * pm_hr)
        adj["hit_rate_rolling"] = clamp_prob(hit_pa * pf_hit * pm_hit)
        adj["venue_id"] = venue_id
        adj["opp_pitcher_id"] = opp_pid
        adj_players.append(adj)

    # 5) Build per-game probabilities
    if not MODEL_OK:
        return JSONResponse(
            content=[{
                "date": date, "playerId": -1, "playerName": "Model not loaded", "team": "", "lineupSpot": None,
                "hr_prob_pa_model": 0.0, "hit_prob_pa_model": 0.0,
                "hr_anytime_prob": 0.0, "hits_1plus_prob": 0.0, "hits_2plus_prob": 0.0,
                "error": f"feature_builder import failed: {MODEL_IMPORT_ERROR}",
            }],
            headers={"Cache-Control": "no-store"},
        )

    modeled = build_today_features(date, adj_players)

    # 6) Odds (optional)
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
            "hr_game_prob": p_hr, "h1_game_prob": p_h1, "h2_game_prob": p_h2,
            "fair_hr_american": fair_hr, "fair_h1_american": fair_h1, "fair_h2_american": fair_h2,
            "hr_market_odds": hr_odds if hr_odds is not None else None,
            "h1_market_odds": h1_odds if h1_odds is not None else None,
            "h2_market_odds": h2_odds if h2_odds is not None else None,
            "hr_market_prob": hr_mkt_p, "h1_market_prob": h1_mkt_p, "h2_market_prob": h2_mkt_p,
            "hr_edge": hr_edge, "h1_edge": h1_edge, "h2_edge": h2_edge,
            "hr_score": hr_score, "h1_score": h1_score, "h2_score": h2_score,
        })

    return JSONResponse(content=rows, headers={"Cache-Control": "no-store"})
