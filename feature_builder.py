# feature_builder.py
from __future__ import annotations

import math
from typing import Any, Dict, List

# ---- small helpers ---------------------------------------------------------

def clamp_prob(p: float) -> float:
    return max(0.0005, min(0.9995, float(p)))

def est_pa_from_spot(spot: int | None) -> float:
    """
    Expected plate appearances by lineup spot (rough MLB averages).
    If unknown, use a neutral starter-ish value.
    """
    table = {
        1: 4.8, 2: 4.6, 3: 4.5, 4: 4.4, 5: 4.3,
        6: 4.2, 7: 4.1, 8: 4.0, 9: 3.9
    }
    if spot is None:
        return 4.2
    try:
        return float(table.get(int(spot), 4.2))
    except Exception:
        return 4.2

def game_prob_at_least_one(p_pa: float, n_pa: float) -> float:
    """P(X>=1) for Binomial(n, p) with fractional n treated continuously."""
    p_pa = clamp_prob(p_pa)
    # Allow fractional n via exp(n*log(1-p))
    return clamp_prob(1.0 - math.exp(n_pa * math.log(1.0 - p_pa)))

def game_prob_at_least_two(p_pa: float, n_pa: float) -> float:
    """P(X>=2) = 1 - [P(0)+P(1)] for Binomial(n, p) with fractional n."""
    p_pa = clamp_prob(p_pa)
    # P(0) = (1-p)^n  (fractional n OK)
    p0 = math.exp(n_pa * math.log(1.0 - p_pa))
    # P(1) ≈ n * p * (1-p)^(n-1) — extend to fractional n
    p1 = n_pa * p_pa * math.exp((n_pa - 1.0) * math.log(1.0 - p_pa))
    return clamp_prob(1.0 - p0 - p1)

# ---- main entry ------------------------------------------------------------

def build_today_features(date_iso: str, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Inputs per player (expected in each dict; set by main.py):
      - 'playerId', 'playerName', 'team', 'lineupSpot'
      - 'hr_rate_rolling'  (per-PA HR rate, e.g., 0.05)
      - 'hit_rate_rolling' (per-PA Hit rate, e.g., 0.24)
      - 'recent_pa'        (season-to-date plate appearances; for confidence)

    Outputs per player (adds):
      - 'hr_prob_pa_model', 'hit_prob_pa_model' (echo of per-PA inputs)
      - 'n_pa_est' (estimated PAs for today)
      - 'hr_anytime_prob' (game P(HR >= 1))
      - 'hits_1plus_prob' (game P(Hits >= 1))
      - 'hits_2plus_prob' (game P(Hits >= 2))
    """
    out: List[Dict[str, Any]] = []

    for p in players:
        hr_pa = clamp_prob(float(p.get("hr_rate_rolling", 0.02)))
        hit_pa = clamp_prob(float(p.get("hit_rate_rolling", 0.24)))
        lineup_spot = p.get("lineupSpot")
        recent_pa = int(p.get("recent_pa") or 0)

        # Base PA by lineup slot; nudge by "confidence" from recent PA (0..30 -> 0..+10%)
        base_n = est_pa_from_spot(lineup_spot)
        conf = max(0.0, min(1.0, recent_pa / 30.0))
        n_pa_est = base_n * (0.9 + 0.2 * conf)  # 0.9x .. 1.1x

        # Game-level probabilities from per-PA rates
        hr_any = game_prob_at_least_one(hr_pa, n_pa_est)
        h1_any = game_prob_at_least_one(hit_pa, n_pa_est)
        h2_any = game_prob_at_least_two(hit_pa, n_pa_est)

        row = dict(p)
        row["hr_prob_pa_model"] = hr_pa
        row["hit_prob_pa_model"] = hit_pa
        row["n_pa_est"] = round(n_pa_est, 2)

        row["hr_anytime_prob"] = hr_any
        row["hits_1plus_prob"] = h1_any
        row["hits_2plus_prob"] = h2_any

        out.append(row)

    return out
