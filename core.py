# core.py
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =========================
# 1) Shared parameters
# =========================
# This is the single source of truth for h.
# Scenario Lab + Planning Beliefs will use these same h values.
BASE_PRIOR = {
    "easy": dict(alpha=120.0, k=240.0, h=1.20),
    "moderate_run_comfort_pace": dict(alpha=170.0, k=170.0, h=1.15),
    "strength": dict(alpha=60.0, k=90.0, h=1.10),
}
DEFAULT_H = {k: BASE_PRIOR[k]["h"] for k in BASE_PRIOR}

# carryover expressed as "next 2–3 weeks multiplier" for narrative:
CARRYOVER_MULTIPLIER = 1.25

# =========================
# 3) Shared “belief math”
# =========================
def sat_gain(m, alpha, k, h):
    m = max(0.0, float(m))
    return alpha * (m ** h) / ((k ** h) + (m ** h) + 1e-9)


def weighted_load(easy, tempo, strength):
    return easy + 1.6 * tempo + 0.9 * strength


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_gain_from_minutes(row: pd.Series, params: Dict, carryover: float = 0.0) -> float:
    g_now = (
        sat_gain(row["easy_min"], **params["easy"])
        + sat_gain(row["moderate_run_comfort_pace_min"], **params["moderate_run_comfort_pace"])
        + sat_gain(row["strength_min"], **params["strength"])
    )
    if carryover != 0.0:
        g_prev = (
            sat_gain(row["easy_min_prev"], **params["easy"])
            + sat_gain(row["moderate_run_comfort_pace_min_prev"], **params["moderate_run_comfort_pace"])
            + sat_gain(row["strength_min_prev"], **params["strength"])
        )
        g_now = g_now + carryover * g_prev
    return float(g_now)


# =========================
# 4) Learn beliefs from 6 weeks (regularized)
# =========================
def estimate_carryover(df_wk: pd.DataFrame) -> float:
    """
    Estimate carryover coefficient c in:
      improvement ~ a*cur + c*prev
    where cur/prev are weighted loads this week / previous week.
    Busi
    """
    wk = df_wk.copy()
    cur = (0.45*wk["easy_min"] + 0.95*wk["moderate_run_comfort_pace_min"] + 0.30*wk["strength_min"]).values
    prev = np.roll(cur, 1)
    prev[0] = 0.0
    y = wk["speed_improvement"].values

    X = np.vstack([cur, prev]).T
    ridge = 1e-2
    beta = np.linalg.inv(X.T @ X + ridge*np.eye(2)) @ (X.T @ y)
    c = float(beta[1])
    return float(np.clip(c, 0.0, 0.9))


def carryover_half_life_from_c(c: float) -> float:
    """
    Half-life in weeks: c = exp(-ln(2)/half_life)
    """
    if c <= 0:
        return 0.0
    return float(np.log(2) / max(1e-6, -np.log(c)))


def fit_alpha_k(df_wk: pd.DataFrame, carryover: float = 0.0, seed: int = 202) -> Dict:
    """
    Fit alpha,k per lever with fixed h, using regularized random search.
    """
    rng = np.random.default_rng(seed)
    wk = df_wk.copy()

    wk["easy_min_prev"] = wk["easy_min"].shift(1).fillna(0.0)
    wk["moderate_run_comfort_pace_min_prev"] = wk["moderate_run_comfort_pace_min"].shift(1).fillna(0.0)
    wk["strength_min_prev"] = wk["strength_min"].shift(1).fillna(0.0)

    y = wk["speed_improvement"].values.astype(float)
    y_center = y - y.mean()
    y_scale = np.std(y_center) + 1e-6
    y_norm = y_center / y_scale

    bounds = {
        "easy":     {"alpha": (50, 220), "k": (120, 420)},
        "moderate_run_comfort_pace":    {"alpha": (80, 320), "k": (80, 280)},
        "strength": {"alpha": (20, 140), "k": (40, 180)},
    }

    prior = {
        k: dict(alpha=BASE_PRIOR[k]["alpha"], k=BASE_PRIOR[k]["k"], h=DEFAULT_H[k])
        for k in BASE_PRIOR
    }
    print(prior)
    def sample_params():
        p = {}
        for key in ["easy", "moderate_run_comfort_pace", "strength"]:
            a_lo, a_hi = bounds[key]["alpha"]
            k_lo, k_hi = bounds[key]["k"]
            p[key] = dict(
                alpha=float(rng.uniform(a_lo, a_hi)),
                k=float(rng.uniform(k_lo, k_hi)),
                h=DEFAULT_H[key],  # fixed
            )
        return p

    def loss(params):
        preds = []
        for _, row in wk.iterrows():
            preds.append(predict_gain_from_minutes(row, params, carryover=carryover))
        preds = np.array(preds, dtype=float)

        preds_center = preds - preds.mean()
        preds_norm = preds_center / (np.std(preds_center) + 1e-6)

        mse = float(np.mean((preds_norm - y_norm) ** 2))

        reg = 0.0
        for key in ["easy", "moderate_run_comfort_pace", "strength"]:
            reg += ((params[key]["alpha"] - prior[key]["alpha"]) / 80.0) ** 2
            reg += ((params[key]["k"] - prior[key]["k"]) / 120.0) ** 2

        return mse + 0.15 * reg

    best = None
    for _ in range(2500):
        p = sample_params()
        L = loss(p)
        if best is None or L < best["loss"]:
            best = {"loss": L, "params": p}

    return best["params"]


def estimate_risk_threshold(df_wk: pd.DataFrame) -> Dict:
    """
    Learn logistic mapping from load -> risky_week proxy.
    risky_week = soreness high OR missed days.
    """
    wk = df_wk.copy()
    wk["load"] = weighted_load(wk["easy_min"], wk["moderate_run_comfort_pace_min"], wk["strength_min"])
    risky = ((wk["avg_soreness"] >= 5.2) | (wk["missed_days"] >= 1)).astype(int).values
    load = wk["load"].values.astype(float)

    if risky.sum() == 0 or risky.sum() == len(risky):
        return {"threshold": float(np.median(load)), "slope": 0.02}

    thr_grid = np.linspace(load.min(), load.max(), 40)
    slope_grid = np.linspace(0.005, 0.06, 40)

    best = None
    for thr in thr_grid:
        for slope in slope_grid:
            p = sigmoid(slope * (load - thr))
            eps = 1e-6
            nll = -np.mean(risky*np.log(p+eps) + (1-risky)*np.log(1-p+eps))
            if best is None or nll < best["nll"]:
                best = {"nll": float(nll), "threshold": float(thr), "slope": float(slope)}
    return best


# =========================
# 5) Beliefs bundle
# =========================
@dataclass(frozen=True)
class Beliefs:
    params: Dict
    carryover_c: float
    half_life_wks: float
    risk_threshold: float
    risk_slope: float
    baseline_soreness_14d: float


@st.cache_data
def get_history_and_beliefs(seed: int = 11, weeks: int = 6) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Beliefs]:
    """
    Returns:
      df_daily, df_weekly_raw, df_weekly_proxy, beliefs
    """
    df_daily = pd.read_csv('./fitness_data/fitness_daily.csv')
    df_proxy = pd.read_csv('./fitness_data/fitness_weekly.csv')

    carry_c = estimate_carryover(df_proxy)
    half_life_wks = carryover_half_life_from_c(carry_c)
    learned_params = fit_alpha_k(df_proxy, carryover=carry_c, seed=202)
    
    risk_fit = estimate_risk_threshold(df_proxy)

    baseline_soreness_14d = float(df_daily.tail(14)["soreness"].mean())

    beliefs = Beliefs(
        params=learned_params,
        carryover_c=carry_c,
        half_life_wks=half_life_wks,
        risk_threshold=float(risk_fit["threshold"]),
        risk_slope=float(risk_fit["slope"]),
        baseline_soreness_14d=baseline_soreness_14d,
    )

    return df_daily, df_proxy, beliefs


# =========================
# 6) Scenario simulation helpers
# =========================
def compute_expected_improvement(easy, moderate_run_comfort_pace, strength, params):
    g = (
        sat_gain(easy, **params["easy"])
        + sat_gain(moderate_run_comfort_pace, **params["moderate_run_comfort_pace"])
        + sat_gain(strength, **params["strength"])
    )
    return CARRYOVER_MULTIPLIER * g


def risk_probability_from_load(load, threshold, slope):
    return float(sigmoid(slope * (load - threshold)))


def simulate_uncertainty(
    easy: int,
    tempo: int,
    strength: int,
    wk_context: Dict,
    beliefs: Beliefs,
    n: int = 2000,
    seed: int = 999
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo around learned params:
      - lognormal noise on alpha/k around learned values (instability increases with soreness/sleep/missed)
      - additive outcome noise
      - risk based on learned threshold model + noise
    """
    rng = np.random.default_rng(seed)

    soreness = float(wk_context["avg_soreness"])
    sleep = float(wk_context["avg_sleep"])
    missed = int(wk_context["missed_days"])

    instability = 0.08 + 0.03 * max(0, soreness - 4.0) + 0.02 * max(0, 7.0 - sleep) + 0.02 * missed
    instability = float(np.clip(instability, 0.08, 0.25))

    base = beliefs.params

    imps = []
    risks = []

    for _ in range(n):
        # jitter alpha/k; keep h fixed
        params = {}
        for key in ["easy", "moderate_run_comfort_pace", "strength"]:
            b = base[key]
            params[key] = dict(
                alpha=b["alpha"] * rng.lognormal(mean=0.0, sigma=instability),
                k=b["k"] * rng.lognormal(mean=0.0, sigma=instability),
                h=b["h"],
            )

        imp = compute_expected_improvement(easy, tempo, strength, params)
        imp = max(0.0, imp + rng.normal(0, 10.0 + 35.0 * instability))

        load = weighted_load(easy, tempo, strength)
        r = risk_probability_from_load(load, beliefs.risk_threshold, beliefs.risk_slope)
        r = float(np.clip(r + rng.normal(0, 0.06 * (1 + 3 * instability)), 0.0, 1.0))

        imps.append(imp)
        risks.append(r)

    return np.array(imps), np.array(risks)


#def recommend_plan(total_min: int, wk_context: Dict, beliefs: Beliefs, risk_posture: float) -> Dict:

# def recommend_plan(
#     total_min: int,
#     wk_context: Dict,
#     beliefs: "Beliefs",
#     risk_posture: float,
#     risk_limit_pct: float,
#     fixed_strength: int = 40,          # NEW
#     step: int = 10,                    # optional
#     n_sims: int = 700,                 # optional (speed)
#     seed: int = 123                    # optional
# ) -> Optional[Dict]:
#     """
#     Grid search with:
#       - fixed strength minutes
#       - hard risk constraint: E[risk]% <= risk_limit_pct
#       - utility = E[improvement] - lambda * E[risk]
#     """
#     if fixed_strength < 0 or fixed_strength > total_min:
#         return None

#     remaining = total_min - fixed_strength
#     lam = float(np.interp(risk_posture, [0.0, 1.0], [140.0, 40.0]))

#     best = None

#     # Search only over easy/tempo; strength is fixed
#     for easy in range(0, remaining + 1, step):
#         for tempo in range(0, remaining - easy + 1, step):
#             strength = fixed_strength

#             # Keep your tempo cap; choose whether it should be relative to total or remaining
#             if tempo > 0.45 * total_min:
#                 continue

#             imps, risks = simulate_uncertainty(
#                 easy, tempo, strength, wk_context, beliefs,
#                 n=n_sims, seed=seed
#             )
#             e_imp = float(np.mean(imps))
#             e_risk = float(np.mean(risks))          # 0..1

#             # Hard constraint (same unit as KPI "chance of breaking down")
#             if e_risk * 100.0 > risk_limit_pct:
#                 continue

#             utility = e_imp - lam * e_risk

#             if best is None or utility > best["utility"]:
#                 best = dict(
#                     easy=easy, tempo=tempo, strength=strength,
#                     e_imp=e_imp, e_risk=e_risk, utility=utility
#                 )

#     return best
from typing import Dict, Optional, List
import numpy as np


def enumerate_candidate_plans(
    total_min: int,
    fixed_strength: int = 40,
    step: int = 10,
    min_easy_min: int = 60,
) -> List[Dict]:
    """
    Build all valid plans where:
        easy + tempo + fixed_strength = total_min
        easy >= min_easy_min
        tempo >= 0
    """
    if fixed_strength < 0 or fixed_strength > total_min:
        return []

    remaining = total_min - fixed_strength
    if remaining <= 0:
        return []

    min_easy = min(min_easy_min, remaining)
    tempo_max = max(0, remaining - min_easy)

    candidates = []
    for tempo in range(0, tempo_max + 1, step):
        easy = remaining - tempo
        if easy < min_easy:
            continue

        candidates.append(
            {
                "easy": int(easy),
                "tempo": int(tempo),
                "strength": int(fixed_strength),
            }
        )

    return candidates


def recommend_plan(
    total_min: int,
    wk_context: Dict,
    beliefs,
    risk_posture: float,
    risk_limit_pct: float,
    fixed_strength: int = 40,
    step: int = 10,
    n_sims: int = 700,
    seed: int = 123,
    min_easy_min: int = 60,
) -> Optional[Dict]:
    """
    Returns the best recommendation under the requested risk limit when possible.
    If that limit is impossible, returns the lowest-risk plan instead.

    Output includes:
      - constraint_met: whether requested risk limit was satisfied
      - risk_limit_requested_pct
      - risk_limit_used_pct
      - fallback_used
    """
    candidates = enumerate_candidate_plans(
        total_min=total_min,
        fixed_strength=fixed_strength,
        step=step,
        min_easy_min=min_easy_min,
    )

    if not candidates:
        return None

    lam = float(np.interp(risk_posture, [0.0, 1.0], [140.0, 40.0]))
    rng = np.random.default_rng(seed)

    evaluated = []
    for cand in candidates:
        imps, risks = simulate_uncertainty(
            cand["easy"],
            cand["tempo"],
            cand["strength"],
            wk_context,
            beliefs,
            n=n_sims,
            seed=int(rng.integers(1, 1_000_000)),
        )

        e_imp = float(np.mean(imps))
        e_risk = float(np.mean(risks))          # 0..1
        e_risk_pct = e_risk * 100.0
        utility = e_imp - lam * e_risk

        evaluated.append(
            {
                **cand,
                "e_imp": e_imp,
                "e_risk": e_risk,
                "e_risk_pct": e_risk_pct,
                "utility": utility,
            }
        )

    feasible = [x for x in evaluated if x["e_risk_pct"] <= risk_limit_pct]

    if feasible:
        best = max(feasible, key=lambda x: x["utility"])
        best["constraint_met"] = True
        best["fallback_used"] = False
        best["risk_limit_requested_pct"] = float(risk_limit_pct)
        best["risk_limit_used_pct"] = float(risk_limit_pct)
        return best

    # fallback: always return the minimum-risk plan
    # tie-breaker: among same-risk plans, prefer higher expected improvement
    fallback = min(evaluated, key=lambda x: (x["e_risk_pct"], -x["e_imp"]))
    fallback["constraint_met"] = False
    fallback["fallback_used"] = True
    fallback["risk_limit_requested_pct"] = float(risk_limit_pct)
    fallback["risk_limit_used_pct"] = float(fallback["e_risk_pct"]) + 0.1
    return fallback