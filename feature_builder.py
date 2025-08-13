# feature_builder.py
from __future__ import annotations
from typing import List, Dict, Any
import math

def _clamp(p: float) -> float:
    # keep probabilities numerically sane
    return max(0.0005, min(0.9995, float(p)))

def _estimate_pa(lineup_spot, recent_pa) -> float:
    """
    Estimate plate appearances for a game using lineup spot + a bit of confidence from recent PA.
    Spots earlier in the order get more PA on average.
    """
    base = 4.2
    adj_by_spot = {1: 0.40, 2: 0.30, 3: 0.25, 4: 0.20, 5: 0.10, 6: 0.00, 7: -0.10, 8: -0.25, 9: -0.40}
    adj = adj_by_spot.get(int(lineup_spot) if lineup_spot else 6, 0.0)

    # if we have more recent PA, trust the spot-adjusted estimate more
    pa30 = float(recent_pa or 0.0)
    conf = min(1.0, max(0.0, pa30 / 30.0))  # saturate around 30 PA
    est = (1.0 - conf) * 3.8 + conf * (base + adj)

    return max(2.5, min(5.5, est))

def build_today_features(date_str: str, players: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input players must already have:
      - hr_rate_rolling: per-PA HR rate (last 30d)  [0..1] or fallback
      - hit_rate_rolling: per-PA Hit rate (last 30d) [0..1] or fallback
      - lineupSpot: 1..9 or None
      - recent_pa: last-30d PA (for confidence in PA estimate)
    Output rows include per-game probabilities for:
      - hr_anytime_prob
      - hits_1plus_prob
      - hits_2plus_prob
    """
    out: List[Dict[str, Any]] = []

    for p in players:
        hr_r = float(p.get("hr_rate_rolling") or 0.02)       # conservative fallback if missing
        hit_r = float(p.get("hit_rate_rolling") or 0.24)     # conservative fallback if missing
        lineup_spot = p.get("lineupSpot")
        recent_pa = int(p.get("recent_pa") or 0)

        n_pa = _estimate_pa(lineup_spot, recent_pa)

        # Per-game HR anytime: 1 - (1 - p_hr_perPA)^PA
        hr_any = 1.0 - (1.0 - hr_r) ** n_pa

        # Per-game 1+ hit: 1 - (1 - p_hit_perPA)^PA
        h1 = 1.0 - (1.0 - hit_r) ** n_pa

        # Per-game 2+ hits ~ 1 - [P(0 hits) + P(1 hit)] using binomial approximation
        p0 = (1.0 - hit_r) ** n_pa
        p1 = n_pa * hit_r * ((1.0 - hit_r) ** max(0.0, n_pa - 1.0))
        h2 = max(0.0, 1.0 - p0 - p1)

        row = {
            "date": date_str,
            "playerId": int(p["playerId"]),
            "playerName": p["playerName"],
            "team": p.get("team", ""),
            "lineupSpot": lineup_spot,

            # expose per-PA model inputs for debugging/UI
            "hr_prob_pa_model": _clamp(hr_r),
            "hit_prob_pa_model": _clamp(hit_r),
            "hr_rate_rolling": _clamp(hr_r),
            "hit_rate_rolling": _clamp(hit_r),

            # estimated PA used
            "n_pa_est": round(float(n_pa), 2),

            # final per-game probabilities used by the app
            "hr_anytime_prob": _clamp(hr_any),
            "hits_1plus_prob": _clamp(h1),
            "hits_2plus_prob": _clamp(h2),
        }
        out.append(row)

    return out
