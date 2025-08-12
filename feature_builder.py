
import pandas as pd, numpy as np, datetime as dt
import joblib, json

HR_MODEL = joblib.load('hr_model.pkl')
HIT_MODEL = joblib.load('hit_model.pkl')
with open('model_schema.json') as f:
    SCHEMA = json.load(f)
FEATURE_COLS = SCHEMA['feature_cols']

def _estimate_pa(lineup_spot: int) -> int:
    # crude estimate for per-game PAs by lineup position
    if lineup_spot <= 2: return 5
    if lineup_spot <= 5: return 4
    return 4

def _at_least_k_successes(n, p, k):
    from math import comb
    return sum(comb(n,i)*(p**i)*((1-p)**(n-i)) for i in range(k, n+1))

def build_today_features(date_str: str, players: list[dict]) -> pd.DataFrame:
    rows = []
    for pl in players:
        # Placeholder: neutral recent features.
        feats = {
            'launch_speed': 90.0,
            'launch_angle': 12.0,
            'balls': 1.0,
            'strikes': 1.0,
            'ev_rolling': 90.0,
            'hr_rate_rolling': 0.02,
            'hit_rate_rolling': 0.24
        }
        X = np.array([[feats[c] for c in FEATURE_COLS]], dtype=float)
        hr_pa = float(HR_MODEL.predict_proba(X)[0,1])
        hit_pa = float(HIT_MODEL.predict_proba(X)[0,1])

        n_pa = _estimate_pa(int(pl.get('lineupSpot', 3)))
        hr_any = 1.0 - (1.0 - hr_pa)**n_pa
        hits_1 = 1.0 - (1.0 - hit_pa)**n_pa
        # P(X>=2) = 1 - [P(0) + P(1)] under Binomial(n,p)
        from math import comb
        p0 = (1-hit_pa)**n_pa
        p1 = n_pa*hit_pa*((1-hit_pa)**(n_pa-1))
        hits_2 = 1.0 - (p0 + p1)

        rows.append({
            'date': date_str,
            'playerId': pl['playerId'],
            'playerName': pl['playerName'],
            'team': pl['team'],
            'lineupSpot': pl.get('lineupSpot', None),
            'hr_prob_pa_model': hr_pa,
            'hit_prob_pa_model': hit_pa,
            'hr_anytime_prob': max(min(hr_any,1.0),0.0),
            'hits_1plus_prob': max(min(hits_1,1.0),0.0),
            'hits_2plus_prob': max(min(hits_2,1.0),0.0),
        })
    return pd.DataFrame(rows)
