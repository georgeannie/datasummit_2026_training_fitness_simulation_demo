from datetime import date, timedelta
import numpy as np
import pandas as pd

def minutes_to_miles(minutes, pace_min_per_mile):
    return float(minutes) / float(pace_min_per_mile) if pace_min_per_mile > 0 else 0.0

def pace_estimate(df: pd.DataFrame) -> pd.DataFrame:
    PACE_EASY = 11.0   # min/mile
    PACE_TEMPO = 8.7   # min/mile

    df["miles_easy"] = df["minutes_easy"] / PACE_EASY
    df["miles_moderate_run_comfort_pace"] = df["minutes_moderate_run_comfort_pace"] / PACE_TEMPO
    df["miles_total"] = df["miles_easy"] + df["miles_moderate_run_comfort_pace"]
    return df

def e10K_estimate(wk: pd.DataFrame) -> pd.DataFrame:
    '''
    This is a simple, interpretable model to estimate 10K time from recent training.
    It captures the key intuition that more miles generally improve pace, 
    but with diminishing returns and plenty of noise. It also includes a simple recovery penalty 
    based on sleep and soreness, which are important modifiers
    '''

    TENK_MILES = 6.2137
    BETA_TEMPO = 1.35   # tempo miles carry more signal than easy miles

    wk["run_min"] = wk["easy_min"] + wk["moderate_run_comfort_pace_min"]
    wk["eff_miles"] = wk["easy_mi"] + BETA_TEMPO * wk["moderate_run_comfort_pace_mi"]

    # avoid divide-by-zero
    wk["eff_speed"] = np.where(wk["run_min"] > 0, wk["eff_miles"] / wk["run_min"], np.nan)  # miles per minute

    wk["tenk_time_raw"] = TENK_MILES / wk["eff_speed"]  # minutes

    # recovery penalty (simple + interpretable)
    wk["tenk_penalty"] = (
        0.8 * np.maximum(0, wk["avg_soreness"] - 4.0) +
        0.6 * np.maximum(0, 7.0 - wk["avg_sleep"])
    )

    wk["10k_pred_time"] = wk["tenk_time_raw"] + wk["tenk_penalty"]
    return wk

def generate_daily_data(seed: int = 7, weeks: int = 6) -> pd.DataFrame:
    """
    Generate daily training data for 6 weeks that feels plausible:
    - A mostly consistent routine
    - Some missed days
    - A couple recovery dips (sleep down, soreness up)
    """
    rng = np.random.default_rng(seed)
    start = date.today() - timedelta(days=7 * weeks)

    rows = []
    base_sleep = 7.1
    soreness = 3.0

    for d in range(7 * weeks):
        dt = start + timedelta(days=d)
        dow = dt.weekday()  # 0=Mon ... 6=Sun

        # weekly structure: Tue tempo, Thu easy, Sat long-ish, plus strength 2x
        easy = 0
        tempo = 0
        strength = 0

        # Missed days: small probability + slightly higher when soreness high
        miss_prob = 0.05 + 0.02 * max(0, soreness - 4)
        missed = rng.random() < miss_prob

        if not missed:
            # Base plan
            if dow in [0, 2, 4]:  # Mon/Wed/Fri: easy
                easy = int(rng.normal(40, 10))
            if dow == 1:          # Tue: tempo
                tempo = int(rng.normal(35, 8))
            if dow == 5:          # Sat: longer easy
                easy = int(rng.normal(70, 12))
            if dow in [3, 6]:     # Thu/Sun: strength or recovery
                strength = int(rng.normal(25, 6))

            easy = max(0, easy)
            tempo = max(0, tempo)
            strength = max(0, strength)

        # Create a couple "life happens" periods that drive uncertainty
        # Weeks 3 and 5: sleep down + soreness up
        wk = d // 7 + 1
        shock = 0.0
        if wk in [3, 5]:
            shock = 0.6

        sleep = float(np.clip(rng.normal(base_sleep - shock, 0.5), 5.0, 9.0))

        # soreness increases with load, decreases with sleep and rest
        load = easy + 1.4 * tempo + 0.8 * strength
        soreness = (
            0.70 * soreness
            + 0.012 * load
            - 0.25 * (sleep - 7.0)
            + rng.normal(0, 0.25)
        )
        soreness = float(np.clip(soreness, 1.0, 10.0))

        notes = ""
        if missed:
            notes = "missed"
        elif wk in [3, 5] and sleep < 6.5:
            notes = "low sleep week"

        pace_easy = float(np.clip(rng.normal(11.0, 0.6), 9.5, 13.0))
        pace_tempo = float(np.clip(rng.normal(8.6, 0.4), 7.6, 9.6))
        pace_strength = 999.0  # not meaningful miles

        miles_easy = minutes_to_miles(easy, pace_easy)
        miles_tempo = minutes_to_miles(tempo, pace_tempo)
        miles_total = miles_easy + miles_tempo

        rows.append(
            dict(
                date=pd.to_datetime(dt),
                minutes_easy=int(easy),
                minutes_moderate_run_comfort_pace=int(tempo),
                minutes_strength=int(strength),
                miles_easy=round(miles_easy, 3),
                miles_moderate_run_comfort_pace=round(miles_tempo, 3),
                miles_total=round(miles_total, 3),
                sleep_hours=round(sleep, 2),
                soreness=round(soreness, 2),
                notes=notes
            )
        )

    df = pd.DataFrame(rows)
    df["week"] = df["date"].dt.to_period("W").astype(str)
    return df


def weekly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = pace_estimate(df)
    wk = (
        df.groupby("week", as_index=False)
        .agg(
            easy_min=("minutes_easy", "sum"),
            moderate_run_comfort_pace_min=("minutes_moderate_run_comfort_pace", "sum"),
            strength_min=("minutes_strength", "sum"),
            easy_mi=("miles_easy", "sum"),
            moderate_run_comfort_pace_mi=("miles_moderate_run_comfort_pace", "sum"),
            total_mi=("miles_total", "sum"),
            avg_soreness=("soreness", "mean"),
            missed_days=("notes", lambda s: int((s == "missed").sum())),
            avg_sleep=("sleep_hours", "mean")
        )
    )

    wk["total_min"] = wk["easy_min"] + wk["moderate_run_comfort_pace_min"] + wk["strength_min"]
    
    #“Each week I estimate my current 10K time from my recent training volume.
    # More miles generally improve pace, but with diminishing returns and plenty of noise.”
    wk = e10K_estimate(wk)

     # Define weekly "improvement" as change vs previous week (positive is better)
    wk["speed_improvement"] = wk["10k_pred_time"].shift(1) - wk["10k_pred_time"]
    wk["speed_improvement"] = wk["speed_improvement"].fillna(0.0)
    return wk

if __name__ == "__main__":
    daily_df = generate_daily_data()
    print(daily_df.head())
    weekly_df = weekly_aggregate(daily_df)
    print(weekly_df.head())
    daily_df.to_csv("fitness_daily.csv", index=False)
    weekly_df.to_csv("fitness_weekly.csv", index=False)