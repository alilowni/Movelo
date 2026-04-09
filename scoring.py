"""
Bike scoring, trial log I/O, and data management.

The master CSV gets two extra runtime columns:
  - status:          "available" | "sold"  (persisted, toggled from dashboard)
  - days_on_market:  int (incremented by the scheduler each simulated day)
"""

import os
import random

import pandas as pd
import numpy as np

import config as cfg

TRIALS_LOG_COLUMNS = [
    "trial_num",
    "bike_id",
    "date",
    "selling_angle",
    "target_audience",
    "tone",
    "actions",
    "instagram_caption",
    "email_subject",
    "email_body",
    "image_prompt_a",
    "image_prompt_b",
    "urban_image_path",
    "nature_image_path",
]


# ---------------------------------------------------------------------------
# CSV helpers -- ensure status & days_on_market columns exist
# ---------------------------------------------------------------------------

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        df["status"] = "available"
    if "days_on_market" not in df.columns:
        df["days_on_market"] = 0
    df["status"] = df["status"].fillna("available")
    df["days_on_market"] = pd.to_numeric(df["days_on_market"], errors="coerce").fillna(0).astype(int)
    return df


def _persist_columns(csv_path: str = None) -> None:
    """Write status + days_on_market back to the master CSV (one-time bootstrap)."""
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    changed = False
    if "status" not in df.columns:
        df["status"] = "available"
        changed = True
    if "days_on_market" not in df.columns:
        df["days_on_market"] = 0
        changed = True
    if changed:
        df.to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Bike scoring
# ---------------------------------------------------------------------------

def load_and_score(csv_path: str = None) -> pd.DataFrame:
    """Load bike CSV, ensure runtime columns, compute sell_difficulty_score."""
    csv_path = csv_path or cfg.CSV_PATH
    _persist_columns(csv_path)
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)

    p_min, p_max = df["price"].min(), df["price"].max()
    df["_price_score"] = ((df["price"] - p_min) / (p_max - p_min)) * 5

    km = df["km_ridden"].copy()
    km_known = km.dropna()
    km_min, km_max = km_known.min(), km_known.max()
    df["_mileage_score"] = np.where(
        km.isna(),
        3.0,
        ((km - km_min) / (km_max - km_min)) * 5,
    )

    condition_map = {"Excellent": 1, "Very good": 2, "Good": 4}
    df["_condition_score"] = df["condition"].map(condition_map).fillna(3)

    year_map = {2024: 1, 2023: 2, 2022: 3, 2021: 4}
    df["_age_score"] = df["year"].map(year_map).fillna(4)

    day_penalty = df["days_on_market"] * cfg.SCORE_DECAY_PER_DAY

    df["sell_difficulty_score"] = (
        0.30 * df["_price_score"]
        + 0.30 * df["_mileage_score"]
        + 0.20 * df["_condition_score"]
        + 0.20 * df["_age_score"]
        + day_penalty
    ).round().astype(int).clip(0, 5)

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
    return df


def filter_hard_to_sell(df: pd.DataFrame,
                        threshold: int = None) -> pd.DataFrame:
    """Return available bikes at or above the threshold."""
    threshold = threshold if threshold is not None else cfg.HARD_SELL_THRESHOLD
    available = df[df["status"] != "sold"]
    hard = available[available["sell_difficulty_score"] >= threshold]
    if hard.empty and threshold > 0:
        return filter_hard_to_sell(df, threshold - 1)
    return hard


# ---------------------------------------------------------------------------
# Sold / days-on-market management
# ---------------------------------------------------------------------------

def mark_bike_sold(bike_id: int, csv_path: str = None) -> None:
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    df.loc[df["id"] == bike_id, "status"] = "sold"
    df.to_csv(csv_path, index=False)


def mark_bike_available(bike_id: int, csv_path: str = None) -> None:
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    df.loc[df["id"] == bike_id, "status"] = "available"
    df.to_csv(csv_path, index=False)


def increment_days_on_market(csv_path: str = None) -> None:
    """Add 1 simulated day to every available bike. Called each scheduler tick."""
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    mask = df["status"] != "sold"
    df.loc[mask, "days_on_market"] = df.loc[mask, "days_on_market"] + 1
    df.to_csv(csv_path, index=False)


def simulate_sales(bike_ids: list[int], scores: dict[int, int],
                   csv_path: str = None) -> list[int]:
    """Roll dice for each marketed bike. Returns list of bike IDs that 'sold'.

    Probability = BASE * (6 - score) / 5
    So score-1 bikes sell ~100% of BASE, score-5 bikes sell ~20% of BASE.
    """
    csv_path = csv_path or cfg.CSV_PATH
    base = cfg.AUTO_SELL_PROBABILITY
    sold = []
    for bid in bike_ids:
        score = scores.get(bid, 3)
        prob = base * (6 - score) / 5
        if random.random() < prob:
            sold.append(bid)
            mark_bike_sold(bid, csv_path)
    return sold


# ---------------------------------------------------------------------------
# Trials log (separate CSV)
# ---------------------------------------------------------------------------

def _ensure_trials_log(path: str = None) -> None:
    path = path or cfg.TRIALS_LOG_PATH
    if not os.path.exists(path):
        pd.DataFrame(columns=TRIALS_LOG_COLUMNS).to_csv(path, index=False)


def load_trials(path: str = None) -> pd.DataFrame:
    path = path or cfg.TRIALS_LOG_PATH
    _ensure_trials_log(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=TRIALS_LOG_COLUMNS)
    return df


def next_trial_number(path: str = None) -> int:
    path = path or cfg.TRIALS_LOG_PATH
    trials = load_trials(path)
    if trials.empty:
        return 1
    return int(trials["trial_num"].max()) + 1


def log_trial_to_csv(records: list[dict], path: str = None) -> None:
    path = path or cfg.TRIALS_LOG_PATH
    _ensure_trials_log(path)
    new_df = pd.DataFrame(records, columns=TRIALS_LOG_COLUMNS)
    new_df.to_csv(path, mode="a", header=False, index=False)


def load_trial_history_for_bike(bike_id: int, path: str = None) -> list[dict]:
    path = path or cfg.TRIALS_LOG_PATH
    trials = load_trials(path)
    if trials.empty:
        return []
    bike_trials = trials[trials["bike_id"] == bike_id].sort_values("trial_num")
    return bike_trials.to_dict(orient="records")


def join_bikes_and_trials(csv_path: str = None,
                          trials_path: str = None) -> pd.DataFrame:
    csv_path = csv_path or cfg.CSV_PATH
    trials_path = trials_path or cfg.TRIALS_LOG_PATH
    bikes = load_and_score(csv_path)
    trials = load_trials(trials_path)
    if trials.empty:
        return bikes
    return bikes.merge(trials, left_on="id", right_on="bike_id", how="left")


if __name__ == "__main__":
    df = load_and_score()
    print(df[["id", "title", "price", "condition", "status",
              "days_on_market", "sell_difficulty_score"]]
          .sort_values("sell_difficulty_score", ascending=False)
          .to_string(index=False))
