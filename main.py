import os
import asyncio
import datetime as dt
from typing import Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---- CORS / ENV ----
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
SUPABASE_JWKS_URL = os.getenv("SUPABASE_JWKS_URL", "").strip()
SUPABASE_JWT_AUDIENCE = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")

app = FastAPI(title="MLB Picks API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Auth (MVP: accept Bearer token but do not verify signature) ----
async def parse_user_from_auth(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    # NOTE: In MVP we don't verify. Add JWKS-based verification in production.
    return {"sub": "user", "note": "token accepted (not verified in MVP)"}

# ---- Models for responses ----
class Prediction(BaseModel):
    date: str
    playerId: int
    playerName: str
    team: str
    hr_prob_pa: float
    hit_prob_pa: float
    hr_edge_vs_market: Optional[float] = None
    hit_edge_vs_market: Optional[float] = None

# Minimal in-memory hub for WebSocket clients
class BroadcastHub:
    def __init__(self):
        self.connections: List[WebSocket] = []
        self.latest_payload: List[Dict[str, Any]] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        if self.latest_payload:
            await websocket.send_json(self.latest_payload)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, payload: List[Dict[str, Any]]):
        self.latest_payload = payload
        living = []
        for ws in self.connections:
            try:
                await ws.send_json(payload)
                living.append(ws)
            except Exception:
                # drop broken connections
                pass
        self.connections = living

hub = BroadcastHub()

# ---- Health & Me ----
@app.get("/health")
async def health():
    return {"ok": True, "time": dt.datetime.utcnow().isoformat()}

@app.get("/me")
async def me(user = Depends(parse_user_from_auth)):
    return {"user": user}

# ------------------------------------------------------------------------------------
# REAL MODEL INFERENCE: /markets
# Requires files in repo root:
#   - hr_model.pkl
#   - hit_model.pkl
#   - model_schema.json
#   - feature_builder.py
# ------------------------------------------------------------------------------------

# Import the feature builder (loads your joblib models)
# If imports fail (e.g., missing pandas/numpy/joblib), raise a clear error.
MISSING_MODEL_ARTIFACTS = False
try:
    from feature_builder import build_today_features  # type: ignore
except Exception as e:
    MISSING_MODEL_ARTIFACTS = True
    IMPORT_ERROR_DETAIL = str(e)

# --- MLB StatsAPI lineup fetch (real) ---
# We use: 
#   Schedule: https://statsapi.mlb.com/api/v1/schedule?sportId=1&date=YYYY-MM-DD
#   Boxscore: https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore
# We extract batting orders (1–9) only; if no lineup yet, we skip that game.

async def _get_game_pks_for_date(client: httpx.AsyncClient, date_str: str) -> List[int]:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": date_str}
    r = await client.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    pks: List[int] = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            # filter only regular/official games
            if g.get("status", {}).get("codedGameState") in {"P", "S", "I", "F", "O", "D", "U", "C"}:
                pks.append(g.get("gamePk"))
            else:
                pks.append(g.get("gamePk"))
    return [pk for pk in pks if pk]

def _extract_lineup_from_boxscore_json(js: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for side in ("home", "away"):
        team_name = js.get("teams", {}).get(side, {}).get("team", {}).get("name", "")
        players = js.get("teams", {}).get(side, {}).get("players", {}) or {}
        for key, pl in players.items():
            # battingOrder is like "101","102",..., "901"; convert to 1..9
            bo = pl.get("battingOrder")
            if not bo:
                continue
            try:
                lineup_spot = int(bo) // 100
                if lineup_spot < 1 or lineup_spot > 9:
                    continue
            except Exception:
                continue
            person = pl.get("person", {})
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
    # dedupe by playerId (some double-entries can appear)
    seen = set()
    dedup: List[Dict[str, Any]] = []
    for r in out:
        if r["playerId"] in seen:
            continue
        seen.add(r["playerId"])
        dedup.append(r)
    return dedup

async def _fetch_boxscore(client: httpx.AsyncClient, game_pk: int) -> List[Dict[str, Any]]:
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    r = await client.get(url, timeout=20)
    if r.status_code != 200:
        return []
    try:
        return _extract_lineup_from_boxscore_json(r.json())
    except Exception:
        return []

async def fetch_lineups_for_today(date_str: Optional[str] = None) -> List[Dict[str, Any]]:
    if not date_str:
        date_str = dt.date.today().isoformat()
    async with httpx.AsyncClient() as client:
        game_pks = await _get_game_pks_for_date(client, date_str)
        # fetch boxscores concurrently
        tasks = [asyncio.create_task(_fetch_boxscore(client, pk)) for pk in game_pks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    players: List[Dict[str, Any]] = []
    for res in results:
        if isinstance(res, Exception):
            continue
        players.extend(res)
    # If no lineups are posted yet (early morning), you could fallback here to rosters.
    return players


async def compute_markets_payload(date: Optional[str] = None) -> List[Dict[str, Any]]:
    """Build the JSON payload for /markets and websocket broadcast."""
    if date is None:
        date = dt.date.today().isoformat()

    if MISSING_MODEL_ARTIFACTS:
        # Helpful message if imports failed
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
            "error": f"feature_builder import failed: {IMPORT_ERROR_DETAIL}"
        }]

    players = await fetch_lineups_for_today(date)
    # build_today_features returns a pandas DataFrame; convert to JSON-ready dicts
    df = build_today_features(date, players)
    # Ensure floats are native Python floats
    records: List[Dict[str, Any]] = []
    for r in df.to_dict(orient="records"):
        records.append({
            "date": str(r.get("date", date)),
            "playerId": int(r.get("playerId", -1)),
            "playerName": str(r.get("playerName", "")),
            "team": str(r.get("team", "")),
            "lineupSpot": r.get("lineupSpot"),
            "hr_prob_pa_model": float(r.get("hr_prob_pa_model", 0.0)),
            "hit_prob_pa_model": float(r.get("hit_prob_pa_model", 0.0)),
            "hr_anytime_prob": float(r.get("hr_anytime_prob", 0.0)),
            "hits_1plus_prob": float(r.get("hits_1plus_prob", 0.0)),
            "hits_2plus_prob": float(r.get("hits_2plus_prob", 0.0)),
        })
    return records

@app.get("/markets")
async def markets(date: Optional[str] = None, user = Depends(parse_user_from_auth)):
    return await compute_markets_payload(date)

# ------------------------------------------------------------------------------------
# Optional: keep the old /predictions stub for compatibility/tests (not used by UI)
# ------------------------------------------------------------------------------------
async def stub_fetch_predictions(date: Optional[str] = None) -> List[Prediction]:
    if not date:
        date = dt.date.today().isoformat()
    sample = [
        Prediction(date=date, playerId=660271, playerName="Aaron Judge", team="Yankees",
                   hr_prob_pa=0.12, hit_prob_pa=0.34, hr_edge_vs_market=0.02, hit_edge_vs_market=-0.01),
        Prediction(date=date, playerId=592450, playerName="Mike Trout", team="Angels",
                   hr_prob_pa=0.10, hit_prob_pa=0.32, hr_edge_vs_market=0.01, hit_edge_vs_market=0.03),
        Prediction(date=date, playerId=592626, playerName="Bryce Harper", team="Phillies",
                   hr_prob_pa=0.09, hit_prob_pa=0.31, hr_edge_vs_market=0.00, hit_edge_vs_market=0.02),
    ]
    return sample

@app.get("/predictions")
async def predictions(date: Optional[str] = None, user = Depends(parse_user_from_auth)):
    rows = await stub_fetch_predictions(date)
    return [r.model_dump() for r in rows]

# ------------------------------------------------------------------------------------
# WebSocket for live updates (now broadcasts REAL markets every 5 minutes)
# ------------------------------------------------------------------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await hub.connect(websocket)
    try:
        while True:
            # keep alive; client doesn't need to send anything
            await asyncio.sleep(30)
    except WebSocketDisconnect:
        hub.disconnect(websocket)

async def background_refresh():
    while True:
        try:
            payload = await compute_markets_payload()
            await hub.broadcast(payload)
        except Exception as e:
            # Broadcast an error payload rather than crashing the task
            await hub.broadcast([{"error": f"refresh failed: {str(e)}", "time": dt.datetime.utcnow().isoformat()}])
        await asyncio.sleep(300)  # 5 minutes

@app.on_event("startup")
async def on_start():
    asyncio.create_task(background_refresh())
