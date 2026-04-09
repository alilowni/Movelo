"""
Bike scoring, trial log I/O, and data management.

Runtime columns on the master CSV:
  - status:          "available" | "sold"
  - days_on_market:  int (incremented each campaign run)
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
    "sold_in_campaign",
]

DAY_PENALTY = 0.30


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        df["status"] = "available"
    if "days_on_market" not in df.columns:
        df["days_on_market"] = 0
    df["status"] = df["status"].fillna("available")
    df["days_on_market"] = pd.to_numeric(df["days_on_market"], errors="coerce").fillna(0).astype(int)
    return df


def _persist_columns(csv_path: str = None) -> None:
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
# Scoring
# ---------------------------------------------------------------------------

def load_and_score(csv_path: str = None) -> pd.DataFrame:
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
        km.isna(), 3.0,
        ((km - km_min) / (km_max - km_min)) * 5,
    )

    condition_map = {"Excellent": 1, "Very good": 2, "Good": 4}
    df["_condition_score"] = df["condition"].map(condition_map).fillna(3)

    year_map = {2024: 1, 2023: 2, 2022: 3, 2021: 4}
    df["_age_score"] = df["year"].map(year_map).fillna(4)

    day_penalty = df["days_on_market"] * DAY_PENALTY

    df["sell_difficulty_score"] = (
        0.30 * df["_price_score"]
        + 0.30 * df["_mileage_score"]
        + 0.20 * df["_condition_score"]
        + 0.20 * df["_age_score"]
        + day_penalty
    ).round().astype(int).clip(0, 5)

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
    return df


def filter_hard_to_sell(df: pd.DataFrame, threshold: int = None) -> pd.DataFrame:
    threshold = threshold if threshold is not None else cfg.HARD_SELL_THRESHOLD
    available = df[df["status"] != "sold"]
    hard = available[available["sell_difficulty_score"] >= threshold]
    if hard.empty and threshold > 0:
        return filter_hard_to_sell(df, threshold - 1)
    return hard


# ---------------------------------------------------------------------------
# Status management
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


def advance_day(csv_path: str = None) -> None:
    """Add 1 day to every available bike. Called once per campaign."""
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    mask = df["status"] != "sold"
    df.loc[mask, "days_on_market"] = df.loc[mask, "days_on_market"] + 1
    df.to_csv(csv_path, index=False)


def simulate_sales(bike_ids: list[int], scores: dict[int, int],
                   trial_num: int, csv_path: str = None,
                   trials_path: str = None) -> list[int]:
    """Roll dice for each marketed bike. Returns list of IDs that sold.
    P(sold) = BASE * (6 - score) / 5
    Also stamps sold_in_campaign in the trials log for easy retrieval.
    """
    csv_path = csv_path or cfg.CSV_PATH
    trials_path = trials_path or cfg.TRIALS_LOG_PATH
    base = cfg.AUTO_SELL_PROBABILITY
    sold = []
    for bid in bike_ids:
        score = scores.get(bid, 3)
        prob = base * (6 - score) / 5
        if random.random() < prob:
            sold.append(bid)
            mark_bike_sold(bid, csv_path)

    if sold:
        _stamp_sold_in_log(sold, trial_num, trials_path)
    return sold


def _stamp_sold_in_log(sold_ids: list[int], trial_num: int,
                       path: str = None) -> None:
    """Mark sold_in_campaign=yes for bikes sold in this campaign."""
    path = path or cfg.TRIALS_LOG_PATH
    df = pd.read_csv(path)
    if "sold_in_campaign" not in df.columns:
        df["sold_in_campaign"] = ""
    df["sold_in_campaign"] = df["sold_in_campaign"].fillna("").astype(str)
    mask = (df["trial_num"] == trial_num) & (df["bike_id"].isin(sold_ids))
    df.loc[mask, "sold_in_campaign"] = "yes"
    df.to_csv(path, index=False)


def unsold_bike_ids(csv_path: str = None) -> list[int]:
    """Simple retrieval: all bike IDs that are still available."""
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    return df[df["status"] != "sold"]["id"].tolist()


# ---------------------------------------------------------------------------
# Trials log
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
