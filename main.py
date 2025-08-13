# main.py  (v1.5.0)
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

BACKEND_NAME = "mlb-picks-backend"
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()
ALLOWED_ORIGINS = [o for o in (os.getenv("ALLOWED_ORIGINS") or "*").split(",") if o]

# Small park list (defaults 1.00 when unknown)
PARK_FACTORS: Dict[int, Dict[str, float]] = {
    3309: {"hr": 1.08, "hit": 1.02}, 2395: {"hr": 1.03, "hit": 1.00}, 2681: {"hr": 0.92, "hit": 0.98},
    2680: {"hr": 0.95, "hit": 0.99}, 4169: {"hr": 1.10, "hit": 1.03}, 4034: {"hr": 1.04, "hit": 1.01},
    4919: {"hr": 1.11, "hit": 1.03}, 2889: {"hr": 0.90, "hit": 0.97}, 3313: {"hr": 1.05, "hit": 1.02},
    3839: {"hr": 1.06, "hit": 1.02},
}

# Ballpark coords for weather (subset; unknown venues fall back to no weather adj)
VENUE_COORDS: Dict[int, Tuple[float, float]] = {
    3309: (40.8296, -73.9262),  # Yankee
    2395: (42.3467, -71.0972),  # Fenway
    4169: (39.9057, -75.1665),  # Citizens Bank
    4919: (39.0976, -84.5068),  # Great American
    3839: (41.9484, -87.6553),  # Wrigley
    3313: (34.0739, -118.2400), # Dodger
    2680: (37.7786, -122.3893), # Oracle
    2681: (47.5914, -122.3325), # T-Mobile
    2889: (32.7076, -117.1570), # Petco
    4034: (38.6226, -90.1928),  # Busch
}

app = FastAPI(title=BACKEND_NAME, version="1.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if "*" in ALLOWED_ORIGINS else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_OK = True
MODEL_IMPORT_ERROR = None
try:
    # expects to return per-game hr_anytime_prob / hits_1plus_prob / hits_2plus_prob from per-PA rates
    from feature_builder import build_today_features
except Exception as e:
    MODEL_OK = False
    MODEL_IMPORT_ERROR = str(e)

def clamp_prob(p: Optional[float]) -> float:
    if p is None: return 0.0
    return max(0.0005, min(0.9995, float(p)))

def american_to_prob(odds: Optional[int]) -> Optional[float]:
    if odds is None: return None
    o = int(odds)
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def prob_to_american(p: float) -> int:
    p = clamp_prob(p)
    return -int(round(100*p/(1-p))) if p>=0.5 else int(round(100*(1-p)/p))

def normalize_name(s: str) -> str:
    return " ".join("".join(ch for ch in s.lower().strip() if ch.isalnum() or ch.isspace()).split())

def looks_like_over(s: str) -> bool:
    s = s.strip().lower()
    return s.startswith("over ") or " over" in s

def choose_best(cur: Optional[Dict[str, Any]], cand: Dict[str, Any]) -> Dict[str, Any]:
    if cur is None: return cand
    if (cur.get("book","").lower()!="fanduel") and (cand.get("book","").lower()=="fanduel"):
        return cand
    return cur

def innings_str_to_float(ip_str: str) -> float:
    try:
        if "." not in ip_str: return float(int(ip_str))
        a,b = ip_str.split("."); base=float(int(a)); frac=0.0
        if b=="1": frac=1/3
        elif b=="2": frac=2/3
        return base+frac
    except Exception:
        return 0.0

def park_multiplier(venue_id: Optional[int]) -> Tuple[float,float]:
    if not venue_id: return 1.0,1.0
    pf = PARK_FACTORS.get(int(venue_id), {"hr":1.0,"hit":1.0})
    return float(pf.get("hr",1.0)), float(pf.get("hit",1.0))

def pitcher_multiplier(hr9: float, h9: float) -> Tuple[float,float]:
    league_hr9, league_h9 = 1.10, 8.30
    m_hr = max(0.7, min(1.3, hr9/league_hr9))
    m_hit = max(0.7, min(1.3, h9/league_h9))
    return m_hr, m_hit

def temp_wind_hr_multiplier(temp_f: float, wind_mph: float) -> float:
    # gentle bounds: ~±8% total
    # temperature: +4% at 90F, -4% at 40F (linear around 65F)
    t_adj = max(-0.04, min(0.04, (temp_f - 65.0) * 0.0016))
    # wind speed (no direction model w/o park orientation): up to +4% at ~20 mph
    w_adj = max(0.0, min(0.04, wind_mph * 0.002))
    return 1.0 + t_adj + w_adj

def pick_score(model_prob: float, market_prob: Optional[float], recent_pa: int) -> float:
    p = clamp_prob(model_prob)
    conf = min(1.0, max(0.0, recent_pa/30.0))
    base = 1.0 + 7.0*p         # 1..8 weighted by model prob
    edge = ( (p - market_prob) * 10.0 ) if market_prob is not None else 0.0
    edge = max(-2.0, min(2.0, edge))  # cap edge contribution
    return round(max(1.0, min(10.0, base + edge + 1.0*conf)), 1)

# ---------- Lineups / Rosters ----------
async def fetch_lineups_for_today(date_iso: str) -> List[Dict[str, Any]]:
    # still stubbed; we rely on roster fallback
    return []

async def fetch_team_rosters_for_today(date_iso: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    headers = {"User-Agent": "mlb-picks/1.0"}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=25)
            if r.status_code != 200: return out
            for t in (r.json() or {}).get("teams", []) or []:
                tid = t.get("id"); abbr=(t.get("abbreviation") or "").upper()
                if not tid: continue
                rr = await client.get(f"https://statsapi.mlb.com/api/v1/teams/{tid}/roster", timeout=25)
                if rr.status_code != 200: continue
                for slot in (rr.json() or {}).get("roster", []) or []:
                    person = (slot or {}).get("person") or {}
                    pid = person.get("id"); pname = person.get("fullName") or ""
                    if not pid or not pname: continue
                    out.append({"playerId": int(pid), "playerName": pname, "team": abbr, "lineupSpot": None})
    except Exception:
        pass
    return out

# ---------- Schedule / Matchups / Parks ----------
async def fetch_schedule_matchups(date_iso: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    headers = {"User-Agent": "mlb-picks/1.0"}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get(
                f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_iso}&hydrate=probablePitcher,team,venue",
                timeout=25
            )
            if r.status_code != 200: return out
            for d in (r.json() or {}).get("dates", []) or []:
                for g in d.get("games", []) or []:
                    venue = (g.get("venue") or {}).get("id")
                    away = (g.get("teams", {}).get("away", {}) or {})
                    home = (g.get("teams", {}).get("home", {}) or {})
                    away_abbr = (away.get("team") or {}).get("abbreviation") or ""
                    home_abbr = (home.get("team") or {}).get("abbreviation") or ""
                    away_pitch = (home.get("probablePitcher") or {}).get("id")
                    home_pitch = (away.get("probablePitcher") or {}).get("id")
                    if away_abbr: out[away_abbr.upper()] = {"opp_pitcher_id": away_pitch, "venue_id": venue}
                    if home_abbr: out[home_abbr.upper()] = {"opp_pitcher_id": home_pitch, "venue_id": venue}
    except Exception:
        pass
    return out

async def fetch_pitcher_rates(pitcher_ids: List[int], date_iso: str) -> Dict[int, Dict[str, float]]:
    res: Dict[int, Dict[str, float]] = {}
    if not pitcher_ids: return res
    try: season = int(date_iso[:4])
    except Exception: season = dt.date.today().year
    headers = {"User-Agent": "mlb-picks/1.0"}
    ids_csv = ",".join(str(x) for x in pitcher_ids)
    url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_csv}&hydrate=stats(group=[pitching],type=[season],season={season})"
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get(url, timeout=30)
            if r.status_code != 200: return res
            for p in (r.json() or {}).get("people", []) or []:
                pid = p.get("id"); hr9, h9 = 1.1, 8.1
                try:
                    stats = (p.get("stats") or [])
                    if stats:
                        splits = (stats[0].get("splits") or [])
                        if splits:
                            st = splits[0].get("stat") or {}
                            ip = innings_str_to_float(str(st.get("inningsPitched","0")))
                            hr = float(st.get("homeRuns",0) or 0)
                            hits = float(st.get("hits",0) or 0)
                            if ip>0:
                                hr9 = 9.0*(hr/ip)
                                h9  = 9.0*(hits/ip)
                except Exception:
                    pass
                if pid: res[int(pid)] = {"hr9": hr9, "h9": h9}
    except Exception:
        pass
    return res

# ---------- Batter Rates (season + recent) ----------
async def fetch_season_rates(players: List[Dict[str, Any]], date_iso: str) -> Dict[int, Dict[str, Any]]:
    try: season = int(date_iso[:4])
    except Exception: season = dt.date.today().year
    base = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    headers = {"User-Agent": "mlb-picks/1.0"}
    async def one(client: httpx.AsyncClient, pid: int):
        params = {"stats": "season", "group": "hitting", "season": str(season), "gameType": "R"}
        try:
            r = await client.get(base.format(pid=pid), params=params, timeout=20)
            data = r.json() if r.status_code==200 else None
        except Exception:
            data = None
        pa=0; hr_rate=0.02; hit_rate=0.24
        try:
            splits=(((data or {}).get("stats") or [])[0].get("splits") or [])
            if splits:
                st=splits[0].get("stat") or {}
                pa=int(st.get("plateAppearances") or 0)
                hr=int(st.get("homeRuns") or 0)
                h=int(st.get("hits") or 0)
                if pa>0:
                    hr_rate=max(0.0005,min(0.9995,hr/pa))
                    hit_rate=max(0.0005,min(0.9995,h/pa))
        except Exception: pass
        return pid, {"hr_rate_season": hr_rate, "hit_rate_season": hit_rate, "season_pa": pa}
    res: Dict[int, Dict[str, Any]] = {}
    seen:set[int]=set()
    async with httpx.AsyncClient(headers=headers) as client:
        sem=asyncio.Semaphore(16)
        async def run(pid:int):
            async with sem: return await one(client,pid)
        tasks=[]
        for p in players:
            pid=int(p.get("playerId") or 0)
            if pid and pid not in seen: seen.add(pid); tasks.append(asyncio.create_task(run(pid)))
        for r in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(r, tuple): res[int(r[0])] = r[1]
    return res

async def fetch_recent15_rates(players: List[Dict[str, Any]], date_iso: str, last_n: int = 15) -> Dict[int, Dict[str, Any]]:
    """Use game logs (up to last 15 games) for a simple recent-form rate."""
    try: season = int(date_iso[:4])
    except Exception: season = dt.date.today().year
    base = "https://statsapi.mlb.com/api/v1/people/{pid}/stats"
    headers = {"User-Agent":"mlb-picks/1.0"}
    async def one(client:httpx.AsyncClient, pid:int):
        params = {"stats":"gameLog","group":"hitting","season":str(season),"gameType":"R"}
        try:
            r = await client.get(base.format(pid=pid), params=params, timeout=20)
            data = r.json() if r.status_code==200 else None
        except Exception:
            data=None
        pa_sum=0; hr_sum=0; h_sum=0; n=0
        try:
            gl=((data or {}).get("stats") or [])[0].get("splits") or []
            for split in gl[:last_n]:
                st=split.get("stat") or {}
                pa = int(st.get("plateAppearances") or 0)
                hr = int(st.get("homeRuns") or 0)
                h  = int(st.get("hits") or 0)
                pa_sum+=pa; hr_sum+=hr; h_sum+=h; n+=1
        except Exception: pass
        hr_rate = max(0.0005, min(0.9995, (hr_sum/pa_sum))) if pa_sum>0 else None
        hit_rate = max(0.0005, min(0.9995, (h_sum/pa_sum))) if pa_sum>0 else None
        return pid, {"hr_rate_recent": hr_rate, "hit_rate_recent": hit_rate, "recent_pa": pa_sum}
    res: Dict[int, Dict[str, Any]] = {}
    seen:set[int]=set()
    async with httpx.AsyncClient(headers=headers) as client:
        sem=asyncio.Semaphore(12)
        async def run(pid:int):
            async with sem: return await one(client,pid)
        tasks=[]
        for p in players:
            pid=int(p.get("playerId") or 0)
            if pid and pid not in seen: seen.add(pid); tasks.append(asyncio.create_task(run(pid)))
        for r in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(r, tuple): res[int(r[0])] = r[1]
    return res

def apply_batter_rates(players: List[Dict[str, Any]],
                       season: Dict[int, Dict[str, Any]],
                       recent: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out=[]
    for p in players:
        pid=int(p["playerId"])
        s = season.get(pid, {})
        r = recent.get(pid, {})
        hr_s = s.get("hr_rate_season", 0.02);  hit_s = s.get("hit_rate_season", 0.24)
        hr_r = r.get("hr_rate_recent");        hit_r = r.get("hit_rate_recent")
        r_pa = int(r.get("recent_pa") or 0)
        # blend weights: 70/30, but scale recent weight by min(1, recent_pa/30)
        w_recent = 0.3 * min(1.0, r_pa/30.0) if hr_r is not None and hit_r is not None else 0.0
        w_season = 1.0 - w_recent
        hr_blend  = clamp_prob(w_season*hr_s  + w_recent*(hr_r  if hr_r  is not None else hr_s))
        hit_blend = clamp_prob(w_season*hit_s + w_recent*(hit_r if hit_r is not None else hit_s))
        q = dict(p)
        q["hr_rate_rolling"] = hr_blend
        q["hit_rate_rolling"] = hit_blend
        q["recent_pa"] = r_pa
        out.append(q)
    return out

# ---------- Weather ----------
async def fetch_weather_hr_multiplier(venue_id: Optional[int], date_iso: str) -> float:
    if not venue_id or venue_id not in VENUE_COORDS:
        return 1.0
    lat, lon = VENUE_COORDS[venue_id]
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m",
        "start_date": date_iso, "end_date": date_iso, "timezone": "UTC",
    }
    headers = {"User-Agent":"mlb-picks/1.0"}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get(url, params=params, timeout=12)
            if r.status_code != 200:
                return 1.0
            j = r.json() or {}
            hh = (j.get("hourly") or {})
            temps = (hh.get("temperature_2m") or []) or []
            winds = (hh.get("windspeed_10m") or []) or []
            if not temps:
                return 1.0
            # rough pre-game/evening average = last 6 available hours
            t = sum(temps[-6:]) / max(1, len(temps[-6:]))
            w = sum(winds[-6:]) / max(1, len(winds[-6:])) if winds else 0.0
            return temp_wind_hr_multiplier(float(t), float(w))
    except Exception:
        return 1.0

# ---------- Odds ----------
async def fetch_market_odds(date_iso: str) -> Dict[Tuple[str,str], Dict[str, Any]]:
    if not ODDS_API_KEY:
        return {}
    sport_key="baseball_mlb"
    market_sets = [
        "player_home_runs,player_hits",
        "player_home_runs,player_hits_over_under",
        "player_home_run,player_total_hits,player_hits_over_under",
    ]
    async def try_once(csv:str):
        url=f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
        params={"regions":"us","markets":csv,"oddsFormat":"american","dateFormat":"iso","apiKey":ODDS_API_KEY}
        headers={"User-Agent":"mlb-picks/1.0"}
        try:
            async with httpx.AsyncClient(headers=headers) as client:
                r=await client.get(url, params=params, timeout=25)
                return r.json() if r.status_code==200 else []
        except Exception:
            return []
    out: Dict[Tuple[str,str], Dict[str,Any]] = {}
    for csv in market_sets:
        events = await try_once(csv)
        found=False
        for ev in events or []:
            for bk in ev.get("bookmakers",[]) or []:
                book=(bk.get("key") or "book").lower()
                for mk in bk.get("markets",[]) or []:
                    mkey=(mk.get("key") or "").lower()
                    for oc in mk.get("outcomes") or []:
                        raw=(oc.get("description") or oc.get("name") or "").strip()
                        if not raw: continue
                        odds=oc.get("price"); 
                        if odds is None: continue
                        label=raw.lower(); market=None; pname=None
                        if mkey in ("player_home_runs","player_home_run") or "home run" in label:
                            market="HR"; pname=raw.replace(" to hit a home run","").replace(" hr","").replace(" - hr","")
                        elif mkey in ("player_hits","player_total_hits","player_hits_over_under"):
                            point=oc.get("point"); is_over=looks_like_over(label)
                            pname=raw.replace(" over","").replace(" under","").replace(" hits","").replace(" total hits","")
                            if point is not None and is_over:
                                try: pt=float(point)
                                except Exception: pt=None
                                if pt is not None:
                                    if 0.5 <= pt < 1.5: market="H1"
                                    elif 1.5 <= pt < 2.5: market="H2"
                        elif "record a hit" in label:
                            market="H1"; pname=raw.replace(" to record a hit","")
                        if not (market and pname): continue
                        key=(normalize_name(pname), market)
                        cand={"odds": int(odds), "book": book}
                        out[key]=choose_best(out.get(key), cand)
                        found=True
        if found: break
    return out

# ---------- Routes ----------
@app.get("/health")
async def health():
    return {"ok": True, "service": BACKEND_NAME, "version":"1.5.0",
            "model_missing": not MODEL_OK, "model_error": MODEL_IMPORT_ERROR,
            "has_odds_api_key": bool(ODDS_API_KEY),
            "time": dt.datetime.utcnow().isoformat()+"Z"}

@app.get("/matchup_status")
async def matchup_status(date: str | None = None):
    date = date or dt.date.today().isoformat()
    m = await fetch_schedule_matchups(date)
    pitchers = [v["opp_pitcher_id"] for v in m.values() if v.get("opp_pitcher_id")]
    uniq = sorted(set(int(x) for x in pitchers if x))
    rates = await fetch_pitcher_rates(uniq, date) if uniq else {}
    return {"date": date, "teams": len(m), "with_pitcher": len(uniq), "pitcher_rates": len(rates)}

@app.get("/odds_status")
async def odds_status():
    has_key = bool(ODDS_API_KEY)
    sample = await fetch_market_odds(dt.date.today().isoformat()) if has_key else {}
    h1 = sum(1 for ((_, m), _) in sample.items() if m == "H1")
    h2 = sum(1 for ((_, m), _) in sample.items() if m == "H2")
    hr = sum(1 for ((_, m), _) in sample.items() if m == "HR")
    return {"has_api_key": has_key, "tot_props": len(sample), "by_market": {"H1": h1, "H2": h2, "HR": hr}}

@app.get("/markets")
async def markets(date: str = Query(default=None)):
    date = date or dt.date.today().isoformat()

    # players
    players = await fetch_lineups_for_today(date)
    if not players:
        players = await fetch_team_rosters_for_today(date)
    if not players:
        return JSONResponse(content=[], headers={"Cache-Control":"no-store"})

    # batter rates: season + recent15 (blend)
    season = await fetch_season_rates(players, date)
    recent = await fetch_recent15_rates(players, date, last_n=15)
    players = apply_batter_rates(players, season, recent)

    # matchups & park
    team_map = await fetch_schedule_matchups(date)
    pitcher_ids = sorted(set(v["opp_pitcher_id"] for v in team_map.values() if v.get("opp_pitcher_id")))
    pitch_rates = await fetch_pitcher_rates(pitcher_ids, date)

    # weather once per venue (cache per call)
    venues = sorted(set(v["venue_id"] for v in team_map.values() if v.get("venue_id")))
    weather_mult: Dict[int, float] = {}
    async def w_task(vid:int):
        weather_mult[vid] = await fetch_weather_hr_multiplier(vid, date)
    await asyncio.gather(*(w_task(int(v)) for v in venues))

    adj_players=[]
    for p in players:
        team=(p.get("team") or "").upper()
        m = team_map.get(team, {})
        venue_id=m.get("venue_id")
        opp_pid=m.get("opp_pitcher_id")

        hr_pa=float(p.get("hr_rate_rolling") or 0.02)
        hit_pa=float(p.get("hit_rate_rolling") or 0.24)

        pf_hr, pf_hit = park_multiplier(venue_id)
        pm_hr, pm_hit = 1.0, 1.0
        if opp_pid and opp_pid in pitch_rates:
            pr=pitch_rates[opp_pid]
            pm_hr, pm_hit = pitcher_multiplier(pr.get("hr9",1.1), pr.get("h9",8.3))
        wm = weather_mult.get(venue_id or -1, 1.0)

        adj = dict(p)
        adj["hr_rate_rolling"]  = clamp_prob(hr_pa  * pf_hr  * pm_hr * wm)
        adj["hit_rate_rolling"] = clamp_prob(hit_pa * pf_hit * pm_hit)  # weather effect small for hits; omit for now
        adj["venue_id"]=venue_id; adj["opp_pitcher_id"]=opp_pid; adj["hr_weather_mult"]=wm
        adj_players.append(adj)

    if not MODEL_OK:
        return JSONResponse(content=[{"date":date,"playerId":-1,"playerName":"Model not loaded",
                                      "team":"","lineupSpot":None,"hr_anytime_prob":0.0,
                                      "hits_1plus_prob":0.0,"hits_2plus_prob":0.0,
                                      "error": f"feature_builder import failed: {MODEL_IMPORT_ERROR}"}],
                            headers={"Cache-Control":"no-store"})

    modeled = build_today_features(date, adj_players)

    odds_map = await fetch_market_odds(date)

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
