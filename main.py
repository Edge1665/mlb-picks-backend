# main.py — per-game lineup+roster fallback, recent rates, robust odds mapping, clamped fair odds, smooth scores
from __future__ import annotations

import os
import re
import math
import asyncio
import datetime as dt
from typing import Optional, List, Dict, Any, Tuple, Set

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------- Model feature builder (robust) ----------------
MISSING_MODEL_ARTIFACTS = False
IMPORT_ERROR_DETAIL = ""
try:
    # robust builder signature: build_today_features(date_str: str, players: list[dict]) -> list[dict]
    from feature_builder import build_today_features
except Exception as e:
    MISSING_MODEL_ARTIFACTS = True
    IMPORT_ERROR_DETAIL = str(e)

# ---------------- App & CORS ----------------
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
app = FastAPI(title="MLB Picks API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Baseline probabilities (used if no sportsbook odds) ----------------
BASELINE_PROB = {
    "HR": 0.045,  # HR Anytime ~4.5%
    "H1": 0.63,   # 1+ Hit ~63%
    "H2": 0.24,   # 2+ Hits ~24%
}

# ================= Name normalization for odds =================
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

def normalize_name(n: str) -> str:
    """
    Lowercase, strip punctuation/parentheses/team tags, drop suffixes.
    Turns "Mookie Betts (LAD) Jr." → "mookie betts"
    """
    n = n.lower().strip()
    n = re.sub(r"[\(\)\-.,'’]", " ", n)  # strip (, ), -, ., commas, apostrophes
    n = re.sub(r"\s{2,}", " ", n)
    parts = [p for p in n.split() if p not in SUFFIXES and len(p) > 0]
    return " ".join(parts)

# ================= MLB API helpers =================
async def _get_schedule(client: httpx.AsyncClient, date_str: str) -> Dict[str, Any]:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str}
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

async def _get_boxscore(client: httpx.AsyncClient, game_pk: int) -> Optional[Dict[str, Any]]:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    r = await client.get(url, timeout=20)
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except Exception:
        return None

def _extract_lineup_from_boxscore_json(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Parse a boxscore JSON for batting orders (1–9)."""
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
                "lineupSpot": lineup_spot,
            })
    # dedupe
    seen: Set[int] = set()
    dedup: List[Dict[str, Any]] = []
    for r in out:
        if r["playerId"] in seen: continue
        seen.add(r["playerId"]); dedup.append(r)
    return dedup

async def _fetch_team_roster(client: httpx.AsyncClient, team_id: int, team_name: str) -> List[Dict[str, Any]]:
    """Roster fallback: active hitters (skip pitchers)."""
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster/active"
    try:
        r = await client.get(url, timeout=15)
        if r.status_code != 200:
            return []
        js = r.json()
        out: List[Dict[str, Any]] = []
        for item in js.get("roster", []) or []:
            person = item.get("person", {}) or {}
            pid = person.get("id")
            name = person.get("fullName") or "Unknown"
            pos = (item.get("position", {}) or {}).get("abbreviation", "")
            if pid and pos != "P":  # skip pitchers
                out.append({
                    "playerId": int(pid),
                    "playerName": str(name),
                    "team": team_name,
                    "lineupSpot": None,  # unknown until lineups post
                })
        return out
    except Exception:
        return []

async def fetch_players_for_today(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Per-game strategy:
      • For each game today: try posted lineups first.
      • If a game's lineups aren't posted yet, fall back to BOTH teams' active hitters.
    This guarantees you see ALL games/teams, not just the ones with a posted lineup.
    """
    if not date_str:
        date_str = dt.date.today().isoformat()

    async with httpx.AsyncClient() as client:
        sched = await _get_schedule(client, date_str)
        games = []
        for d in sched.get("dates", []) or []:
            for g in d.get("games", []) or []:
                games.append(g)

        players_all: List[Dict[str, Any]] = []

        async def process_game(g: Dict[str, Any]):
            # get team ids/names
            teams: List[Tuple[int, str]] = []
            for side in ("home", "away"):
                tinfo = g.get(f"{side}Team", {}) or {}
                tid = tinfo.get("id")
                tname = (tinfo.get("team", {}) or {}).get("name") or tinfo.get("name") or ""
                if tid:
                    teams.append((int(tid), str(tname)))

            # try lineups for this game
            pk = g.get("gamePk")
            game_players: List[Dict[str, Any]] = []
            if pk:
                js = await _get_boxscore(client, int(pk))
                if js:
                    game_players = _extract_lineup_from_boxscore_json(js)

            if not game_players:
                # fallback: both teams' active hitters
                roster_lists = await asyncio.gather(
                    *(_fetch_team_roster(client, tid, tname) for tid, tname in teams),
                    return_exceptions=True
                )
                tmp: List[Dict[str, Any]] = []
                for rs in roster_lists:
                    if isinstance(rs, Exception): continue
                    tmp.extend(rs)
                game_players = tmp

            return game_players

        # process all games concurrently
        per_game = await asyncio.gather(*(process_game(g) for g in games), return_exceptions=True)
        for chunk in per_game:
            if isinstance(chunk, Exception): continue
            players_all.extend(chunk)

        # final dedupe
        seen: Set[int] = set()
        dedup: List[Dict[str, Any]] = []
        for r in players_all:
            if r["playerId"] in seen: continue
            seen.add(r["playerId"]); dedup.append(r)
        return dedup

# ================= Per-player last-30-day hitting rates =================
async def fetch_recent_rates(players: List[dict], date_iso: str, window_days: int = 30) -> Dict[int, Dict[str, float | None]]:
    """
    For each playerId, fetch last-30-day PA/hits/HR and return:
      { playerId: { 'pa': float, 'hr_rate': float|None, 'hit_rate': float|None } }
    """
    end_dt = dt.date.fromisoformat(date_iso)
    start_dt = end_dt - dt.timedelta(days=window_days)
    start, end = start_dt.isoformat(), end_dt.isoformat()
    ids = sorted({int(p["playerId"]) for p in players})
    if not ids:
        return {}

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
    """
    Attach hr_rate_rolling / hit_rate_rolling (+ recent_pa) to each player dict.
    If a player has no recent PA, fallback to conservative baselines so features differ.
    """
    enriched: List[dict] = []
    for pl in players:
        pid = int(pl["playerId"])
        r = rates.get(pid, {})
        hr_rate  = r.get("hr_rate") if r.get("hr_rate") is not None else 0.02
        hit_rate = r.get("hit_rate") if r.get("hit_rate") is not None else 0.24
        pl2 = dict(pl)
        pl2["hr_rate_rolling"]  = float(hr_rate)
        pl2["hit_rate_rolling"] = float(hit_rate)
        pl2["recent_pa"] = int(r.get("pa") or 0)
        enriched.append(pl2)
    return enriched

# ================= Odds helpers (optional) =================
ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "").strip()

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
    # clamp to avoid absurd odds from near-0/1 probabilities
    p = max(0.01, min(0.99, float(p)))
    if p >= 0.5:
        return -int(round((p*100) / (1 - p)))
    else:
        return int(round(((1 - p) * 100) / p))

def _looks_like_over(text: str) -> bool:
    t = text.lower()
    return t.startswith("over") or " over " in t or t.strip() == "over"

async def fetch_market_odds(date_iso: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Returns {(normalized_player_name, market): {'odds': int, 'book': str}}
    Markets we normalize:
      'HR' -> HR Anytime
      'H1' -> 1+ Hits (Over 0.5)
      'H2' -> 2+ Hits (Over 1.5)
    If no API key or no data, returns {} (app still works).
    """
    if not ODDS_API_KEY:
        return {}

    sport_key = "baseball_mlb"
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "regions": "us",
        "markets": "player_home_run,player_hits_over_under,player_total_hits",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "apiKey": ODDS_API_KEY,
    }
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url, params=params, timeout=20)
            if r.status_code != 200:
                return {}
            events = r.json()
    except Exception:
        return {}

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def set_preferring_fanduel(key: Tuple[str, str], odds: int, book: str):
        """
        Store first seen; replace if new one is FanDuel.
        """
        cur = out.get(key)
        if (cur is None) or (cur["book"] != "fanduel" and book == "fanduel"):
            out[key] = {"odds": int(odds), "book": str(book)}

    for ev in events or []:
        for bk in ev.get("bookmakers", []) or []:
            book_key = (bk.get("key") or "book").lower()
            for mk in bk.get("markets", []) or []:
                mk_key = (mk.get("key") or "").lower()
                outcomes = mk.get("outcomes") or []
                for oc in outcomes:
                    raw = (oc.get("description") or oc.get("name") or "").strip()
                    name_l = raw.lower()
                    odds = oc.get("price")
                    if odds is None:
                        continue

                    # Determine market & extract player name
                    market_norm: Optional[str] = None
                    if "home run" in name_l or mk_key == "player_home_run":
                        market_norm = "HR"
                        pl_name = raw
                        for frag in [" to hit a home run", " - hr", " hr"]:
                            pl_name = pl_name.replace(frag, "")
                    elif mk_key in ("player_hits_over_under", "player_total_hits"):
                        point = oc.get("point")
                        is_over = _looks_like_over(name_l)
                        if point is None or not is_over:
                            continue
                        try:
                            pt = float(point)
                        except Exception:
                            continue
                        if 0.5 <= pt < 1.5:
                            market_norm = "H1"
                        elif 1.5 <= pt < 2.5:
                            market_norm = "H2"
                        else:
                            continue
                        pl_name = raw
                        for frag in [" over", " under", " hits", " total hits"]:
                            pl_name = pl_name.replace(frag, "")
                    elif "record a hit" in name_l:
                        market_norm = "H1"
                        pl_name = raw.replace(" to record a hit", "")
                    else:
                        continue

                    pl_key = normalize_name(pl_name)
                    if not pl_key:
                        continue

                    try:
                        set_preferring_fanduel((pl_key, market_norm), int(odds), book_key)
                    except Exception:
                        continue

    return out

# ================= Edge + smooth baseline-aware score =================
def score_pick(model_prob: float, market_prob: Optional[float], recent_pa: int, market: str) -> Tuple[Optional[float], float]:
    """
    Returns (edge, score_1_to_10).
    - With odds: edge = model_prob - market_prob
    - Without odds: edge=None, but score compares model_prob to a baseline for that market.
    Uses a smooth tanh to avoid everything pegging at 10.
    """
    # Confidence from last-30d PA: saturate ~30 PA, floor 0.2
    conf = min(1.0, max(0.2, recent_pa / 30.0))

    # Market weights (rarer outcome → higher impact)
    weight = {"HR": 1.1, "H2": 0.8, "H1": 0.6}.get(market, 0.7)

    if market_prob is None:
        baseline = BASELINE_PROB.get(market, 0.5)
        edge = None
        eff_edge = model_prob - baseline
    else:
        edge = model_prob - market_prob
        eff_edge = edge

    # scale (percentage points) to keep scores in a nice spread
    S = {"HR": 6.0, "H2": 10.0, "H1": 14.0}.get(market, 10.0)
    edge_pct = eff_edge * 100.0

    # Smooth mapping around 5.0 using tanh; multiplied by confidence and market weight
    delta = math.tanh((edge_pct / S) * weight * conf)  # in [-1, 1]
    score = 5.0 + 4.5 * delta  # yields [0.5, 9.5] before clamp
    score = max(1.0, min(10.0, round(score * 10.0) / 10.0))
    return edge, score

# ================= Build payload =================
async def compute_markets_payload(date: Optional[str] = None) -> List[Dict[str, Any]]:
    if date is None:
        date = dt.date.today().isoformat()

    players = await fetch_players_for_today(date)
    if not players:
        return []  # nothing scheduled or API issue

    # enrich with last-30d rates
    rates = await fetch_recent_rates(players, date, window_days=30)
    players = apply_recent_rates(players, rates)

    if MISSING_MODEL_ARTIFACTS:
        payload = []
        for p in players:
            payload.append({
                "date": date,
                "playerId": p["playerId"],
                "playerName": p["playerName"],
                "team": p["team"],
                "lineupSpot": p.get("lineupSpot"),
                "hr_anytime_prob": 0.0,
                "hits_1plus_prob": 0.0,
                "hits_2plus_prob": 0.0,
                "fair_hr_american": None,
                "fair_h1_american": None,
                "fair_h2_american": None,
                "hr_market_odds": None,
                "h1_market_odds": None,
                "h2_market_odds": None,
                "hr_market_prob": None,
                "h1_market_prob": None,
                "h2_market_prob": None,
                "hr_edge": None,
                "h1_edge": None,
                "h2_edge": None,
                "hr_score": 5.0,
                "h1_score": 5.0,
                "h2_score": 5.0,
                "recent_pa": p.get("recent_pa", 0),
                "error": f"feature_builder import failed: {IMPORT_ERROR_DETAIL}",
            })
        return payload

    # model predictions (robust builder ensures variation even if schema mismatch)
    preds = build_today_features(date, players)  # list[dict]

    # optional sportsbook odds
    odds_map = await fetch_market_odds(date)  # {(normalized_name, 'HR'): {'odds': -110, 'book': '...'}, ...}

    out: List[Dict[str, Any]] = []
    for rec in preds:
        # normalize name for odds lookup
        name_key = normalize_name(str(rec["playerName"]))
        pid = int(rec["playerId"])
        recent_pa = int(next((p.get("recent_pa", 0) for p in players if int(p["playerId"]) == pid), 0))

        # ensure model probabilities are sane before using them
        hr_p = max(0.0005, min(0.9995, float(rec["hr_anytime_prob"])))
        h1_p = max(0.0005, min(0.9995, float(rec["hits_1plus_prob"])))
        h2_p = max(0.0005, min(0.9995, float(rec["hits_2plus_prob"])))

        hr_odds = odds_map.get((name_key, "HR"), {}).get("odds")
        h1_odds = odds_map.get((name_key, "H1"), {}).get("odds")
        h2_odds = odds_map.get((name_key, "H2"), {}).get("odds")

        hr_mp = american_to_prob(hr_odds)
        h1_mp = american_to_prob(h1_odds)
        h2_mp = american_to_prob(h2_odds)

        hr_edge, hr_score = score_pick(hr_p, hr_mp, recent_pa, "HR")
        h1_edge, h1_score = score_pick(h1_p, h1_mp, recent_pa, "H1")
        h2_edge, h2_score = score_pick(h2_p, h2_mp, recent_pa, "H2")

        out.append({
            "date": rec.get("date", date),
            "playerId": pid,
            "playerName": rec["playerName"],
            "team": rec.get("team", ""),
            "lineupSpot": rec.get("lineupSpot"),

            # model probabilities (per game)
            "hr_anytime_prob": round(hr_p, 4),
            "hits_1plus_prob": round(h1_p, 4),
            "hits_2plus_prob": round(h2_p, 4),

            # model-implied fair odds (clamped)
            "fair_hr_american": prob_to_american(hr_p),
            "fair_h1_american": prob_to_american(h1_p),
            "fair_h2_american": prob_to_american(h2_p),

            # market odds (optional)
            "hr_market_odds": hr_odds,
            "h1_market_odds": h1_odds,
            "h2_market_odds": h2_odds,
            "hr_market_prob": None if hr_mp is None else round(hr_mp, 4),
            "h1_market_prob": None if h1_mp is None else round(h1_mp, 4),
            "h2_market_prob": None if h2_mp is None else round(h2_mp, 4),

            # edge & score
            "hr_edge": None if hr_edge is None else round(hr_edge, 4),
            "h1_edge": None if h1_edge is None else round(h1_edge, 4),
            "h2_edge": None if h2_edge is None else round(h2_edge, 4),
            "hr_score": hr_score,
            "h1_score": h1_score,
            "h2_score": h2_score,

            # extras for UI / QA (pass-through if present)
            "recent_pa": recent_pa,
            "hr_prob_pa_model": rec.get("hr_prob_pa_model"),
            "hit_prob_pa_model": rec.get("hit_prob_pa_model"),
            "hr_rate_rolling": rec.get("hr_rate_rolling"),
            "hit_rate_rolling": rec.get("hit_rate_rolling"),
            "n_pa_est": rec.get("n_pa_est"),
        })

    return out

# ================= Routes =================
@app.get("/health")
async def health():
    return {
        "ok": True,
        "time": dt.datetime.utcnow().isoformat(),
        "model_missing": MISSING_MODEL_ARTIFACTS,
        "import_error": IMPORT_ERROR_DETAIL
    }

@app.get("/markets")
async def markets(date: str | None = None):
    if not date:
        date = dt.date.today().isoformat()
    return await compute_markets_payload(date)
