# feature_builder.py  — pandas-free, uses per-player recent rates if provided
from __future__ import annotations

import json
import math
from typing import List, Dict, Any

import joblib
import numpy as np

# Load models & schema once at import
HR_MODEL = joblib.load("hr_model.pkl")
HIT_MODEL = joblib.load("hit_model.pkl")
with open("model_schema.json", "r") as f:
    SCHEMA = json.load(f)

FEATURE_COLS: List[str] = SCHEMA.get("feature_cols", [])

# ---- Helpers ----

def _estimate_pa(lineup_spot: int | None) -> int:
    """
    Crude estimate of plate appearances by lineup slot.
    """
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
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

# ---- Main entrypoint used by your API ----

def build_today_features(date_str: str, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    players: list of dicts with at least:
        - playerId (int)
        - playerName (str)
        - team (str)
        - lineupSpot (int | None)
      Optional (recommended):
        - hr_rate_rolling (float)  # HR per PA over last ~30 days
        - hit_rate_rolling (float) # Hits per PA over last ~30 days

    Returns list of dicts with model probabilities + derived market probs.
    """
    out: List[Dict[str, Any]] = []

    # Defaults if recent rates are missing
    DEFAULTS = {
        "launch_speed": 90.0,
        "launch_angle": 12.0,
        "balls": 1.0,
        "strikes": 1.0,
        "ev_rolling": 90.0,
        "hr_rate_rolling": 0.02,   # ~2% HR/PA baseline
        "hit_rate_rolling": 0.24,  # ~24% hit/PA baseline
    }

    # Precompute index mapping for speed
    feat_names = FEATURE_COLS or list(DEFAULTS.keys())

    for pl in players:
        # Pull recent rates from the player dict if present; fall back to sane baselines
        hr_rr = pl.get("hr_rate_rolling", DEFAULTS["hr_rate_rolling"])
        hit_rr = pl.get("hit_rate_rolling", DEFAULTS["hit_rate_rolling"])

        try:
            hr_rr = float(hr_rr) if hr_rr is not None else DEFAULTS["hr_rate_rolling"]
        except Exception:
            hr_rr = DEFAULTS["hr_rate_rolling"]

        try:
            hit_rr = float(hit_rr) if hit_rr is not None else DEFAULTS["hit_rate_rolling"]
        except Exception:
            hit_rr = DEFAULTS["hit_rate_rolling"]

        # Start with neutral defaults and override the two rolling rates
        feature_values: Dict[str, float] = dict(DEFAULTS)
        feature_values["hr_rate_rolling"] = hr_rr
        feature_values["hit_rate_rolling"] = hit_rr

        # Build X row in the exact order the model expects
        row = [float(feature_values.get(col, DEFAULTS.get(col, 0.0))) for col in feat_names]
        X = np.array([row], dtype=float)

        # Predict per-PA probabilities
        hr_pa = float(HR_MODEL.predict_proba(X)[0, 1])
        hit_pa = float(HIT_MODEL.predict_proba(X)[0, 1])

        # Estimate per-game probabilities using a binomial approximation over n PAs
        n_pa = _estimate_pa(pl.get("lineupSpot"))

        # HR anytime = 1 - (1 - p_hr) ^ n
        hr_any = 1.0 - (1.0 - hr_pa) ** n_pa

        # Hits:
        # P(>=1) = 1 - (1 - p_hit)^n
        p0 = (1.0 - hit_pa) ** n_pa
        hits_1plus = 1.0 - p0
        # P(>=2) = 1 - [P(0) + P(1)]
        p1 = n_pa * hit_pa * ((1.0 - hit_pa) ** (n_pa - 1))
        hits_2plus = 1.0 - (p0 + p1)

        out.append(
            {
                "date": date_str,
                "playerId": int(pl.get("playerId", -1)),
                "playerName": str(pl.get("playerName", "")),
                "team": str(pl.get("team", "")),
                "lineupSpot": pl.get("lineupSpot", None),
                "hr_prob_pa_model": _clip01(hr_pa),
                "hit_prob_pa_model": _clip01(hit_pa),
                "hr_anytime_prob": _clip01(hr_any),
                "hits_1plus_prob": _clip01(hits_1plus),
                "hits_2plus_prob": _clip01(hits_2plus),
            }
        )

    return out
