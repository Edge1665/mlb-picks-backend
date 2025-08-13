# feature_builder.py — robust: use model if it expects our rate features; otherwise calibrated fallback per player
from __future__ import annotations

import json
from typing import List, Dict, Any

import joblib
import numpy as np

# ---- Load artifacts ----
HR_MODEL = joblib.load("hr_model.pkl")
HIT_MODEL = joblib.load("hit_model.pkl")
with open("model_schema.json", "r") as f:
    SCHEMA = json.load(f)

FEATURE_COLS: List[str] = SCHEMA.get("feature_cols", [])
FEATURE_SET = set(FEATURE_COLS or [])

# Do the models expect our recent-rate features?
EXPECTS_RATES = ("hr_rate_rolling" in FEATURE_SET) and ("hit_rate_rolling" in FEATURE_SET)

def _estimate_pa(lineup_spot: int | None) -> int:
    """Crude PA estimate by lineup slot."""
    if lineup_spot is None:
        return 4
    try:
        n = int(lineup_spot)
    except Exception:
        return 4
    if n <= 2:
        return 5
    if n <= 5:
        return 4
    return 4

def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def _safe_float(x, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

# --- Calibrated fallback from recent rates (used when model schema doesn’t include our rate features) ---
# These multipliers are conservative so outputs remain realistic.
HR_ALPHA  = 1.10   # slight bump on raw HR/PA
HIT_ALPHA = 1.00   # use raw H/PA

def _fallback_pa_probs(hr_rate_rolling: float, hit_rate_rolling: float) -> tuple[float, float]:
    hr_pa = _clip01(hr_rate_rolling * HR_ALPHA)
    hit_pa = _clip01(hit_rate_rolling * HIT_ALPHA)
    return hr_pa, hit_pa

def build_today_features(date_str: str, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    INPUT per player dict (from main.py):
      - playerId, playerName, team, lineupSpot
      - hr_rate_rolling (float), hit_rate_rolling (float)   # last 30d, set in main.py
    OUTPUT: list[dict] with per-PA + per-game probabilities.
            Also includes hr_rate_rolling, hit_rate_rolling, n_pa_est for debugging/QA.
    """
    out: List[Dict[str, Any]] = []

    # Neutral defaults (used only if rates missing)
    BASE = {
        "launch_speed": 90.0,
        "launch_angle": 12.0,
        "balls": 1.0,
        "strikes": 1.0,
        "ev_rolling": 90.0,
        "hr_rate_rolling": 0.02,   # ~2% HR per PA
        "hit_rate_rolling": 0.24,  # ~24% Hit per PA
    }

    use_model_with_rates = EXPECTS_RATES and (len(FEATURE_COLS) > 0)

    for pl in players:
        # pull player-specific recent rates (main.py sets these)
        hr_rr  = _safe_float(pl.get("hr_rate_rolling", BASE["hr_rate_rolling"]), BASE["hr_rate_rolling"])
        hit_rr = _safe_float(pl.get("hit_rate_rolling", BASE["hit_rate_rolling"]), BASE["hit_rate_rolling"])

        lineup_spot = pl.get("lineupSpot", None)
        n_pa = _estimate_pa(lineup_spot)

        if use_model_with_rates:
            # Build feature row in the exact order expected by the trained model
            feat_vals = dict(BASE)
            feat_vals["hr_rate_rolling"]  = hr_rr
            feat_vals["hit_rate_rolling"] = hit_rr
            row = [_safe_float(feat_vals.get(col, BASE.get(col, 0.0)), 0.0) for col in FEATURE_COLS]
            X = np.array([row], dtype=float)

            hr_pa  = float(HR_MODEL.predict_proba(X)[0, 1])
            hit_pa = float(HIT_MODEL.predict_proba(X)[0, 1])
        else:
            # Fallback: compute per-PA directly from recent rates (ensures variation across players)
            hr_pa, hit_pa = _fallback_pa_probs(hr_rr, hit_rr)

        # Convert per-PA to per-game
        hr_any = 1.0 - (1.0 - hr_pa) ** n_pa
        p0 = (1.0 - hit_pa) ** n_pa
        hits_1plus = 1.0 - p0
        p1 = n_pa * hit_pa * ((1.0 - hit_pa) ** (n_pa - 1))
        hits_2plus = 1.0 - (p0 + p1)

        out.append({
            "date": date_str,
            "playerId": int(pl.get("playerId", -1)),
            "playerName": str(pl.get("playerName", "")),
            "team": str(pl.get("team", "")),
            "lineupSpot": lineup_spot,

            # per-PA
            "hr_prob_pa_model": _clip01(hr_pa),
            "hit_prob_pa_model": _clip01(hit_pa),

            # per-game markets
            "hr_anytime_prob": _clip01(hr_any),
            "hits_1plus_prob": _clip01(hits_1plus),
            "hits_2plus_prob": _clip01(hits_2plus),

            # expose inputs so we can verify variation across players
            "hr_rate_rolling": round(hr_rr, 4),
            "hit_rate_rolling": round(hit_rr, 4),
            "n_pa_est": int(n_pa),
        })

    return out
