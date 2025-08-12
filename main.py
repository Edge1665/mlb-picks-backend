import os
import asyncio
import datetime as dt
from typing import List, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# ---- Simple JWT passthrough (MVP): we won't verify the signature unless JWKS is provided ----
# For MVP: frontend uses Supabase for auth; backend accepts Bearer token and extracts basic info if present.
# You can harden this by verifying with JWKS and checking 'aud' claim.

async def parse_user_from_auth(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1]
    # NOTE: We are NOT verifying signature in MVP. For production, add JWT verification here.
    # Return a minimal stub:
    return {"sub": "user", "note": "token accepted (not verified in MVP)"}

class Prediction(BaseModel):
    date: str
    playerId: int
    playerName: str
    team: str
    hr_prob_pa: float
    hit_prob_pa: float
    hr_edge_vs_market: float | None = None
    hit_edge_vs_market: float | None = None

class BroadcastHub:
    def __init__(self):
        self.connections: list[WebSocket] = []
        self.latest_payload: list[Prediction] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.connections.append(websocket)
        # push last known immediately
        if self.latest_payload:
            await websocket.send_json([p.model_dump() for p in self.latest_payload])

    def disconnect(self, websocket: WebSocket):
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, payload: list[Prediction]):
        self.latest_payload = payload
        living = []
        for ws in self.connections:
            try:
                await ws.send_json([p.model_dump() for p in payload])
                living.append(ws)
            except Exception:
                pass
        self.connections = living

hub = BroadcastHub()

@app.get("/health")
async def health():
    return {"ok": True, "time": dt.datetime.utcnow().isoformat()}

@app.get("/me")
async def me(user = Depends(parse_user_from_auth)):
    return {"user": user}

async def stub_fetch_predictions(date: Optional[str] = None) -> list[Prediction]:
    # Placeholder: create a few fake rows to validate plumbing/UI
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
    # TODO: replace stub with real model inference + odds
    rows = await stub_fetch_predictions(date)
    return [r.model_dump() for r in rows]

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
    # Broadcast fresh predictions every 5 minutes
    while True:
        rows = await stub_fetch_predictions()
        await hub.broadcast(rows)
        await asyncio.sleep(300)

@app.on_event("startup")
async def on_start():
    asyncio.create_task(background_refresh())
