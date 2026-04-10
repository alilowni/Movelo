# Bike scoring, campaign log I/O, and data helpers.
#
# inventory.csv gets two runtime columns:
#   status         — "available" | "sold"
#   days_on_market — int, incremented each campaign

import os
import random
import shutil

import pandas as pd
import numpy as np

import config as cfg


# Init — copy seed to working files if they don't exist yet

def init_working_files() -> None:
    if not os.path.exists(cfg.CSV_PATH):
        if not os.path.exists(cfg.SEED_CSV_PATH):
            raise FileNotFoundError(
                f"Seed file {cfg.SEED_CSV_PATH} not found. Cannot initialise."
            )
        shutil.copy2(cfg.SEED_CSV_PATH, cfg.CSV_PATH)

    if not os.path.exists(cfg.CAMPAIGNS_PATH):
        pd.DataFrame(columns=CAMPAIGN_COLUMNS).to_csv(cfg.CAMPAIGNS_PATH, index=False)

    if not os.path.exists(cfg.KNOWLEDGE_BASE_PATH):
        pd.DataFrame(columns=KB_COLUMNS).to_csv(cfg.KNOWLEDGE_BASE_PATH, index=False)


# Column definitions

KB_COLUMNS = [
    "bike_id", "trial_num", "date", "title", "brand", "category",
    "price", "sell_difficulty_score", "days_on_market", "campaigns_run",
    "selling_angle", "target_audience", "tone", "reason_note",
]

CAMPAIGN_COLUMNS = [
    "trial_num", "bike_id", "date", "selling_angle", "target_audience",
    "tone", "actions", "instagram_caption", "email_subject", "email_body",
    "image_prompt_a", "image_prompt_b", "urban_image_path", "nature_image_path",
    "sold_in_campaign",
]


# Score formula weights — must sum to 1.0
# Adjust with your sales team. "popularity" can be added later.
WEIGHTS = {
    "price":     0.30,   # higher price -> harder to sell
    "mileage":   0.30,   # more km -> harder to sell
    "condition": 0.20,   # worse condition -> harder to sell
    "age":       0.20,   # older -> harder to sell
    # "popularity": 0.0  # placeholder for human scoring
}

DAY_PENALTY_PER_DAY = 0.30


# Internal helpers

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


# Scoring — each sub-score is 0-5, higher = harder to sell

def _score_price(df: pd.DataFrame) -> pd.Series:
    p_min, p_max = df["price"].min(), df["price"].max()
    if p_max == p_min:
        return pd.Series(2.5, index=df.index)
    return ((df["price"] - p_min) / (p_max - p_min)) * 5


def _score_mileage(df: pd.DataFrame) -> pd.Series:
    # Unknown km defaults to mid (3)
    km = df["km_ridden"].copy()
    km_known = km.dropna()
    if km_known.empty or km_known.max() == km_known.min():
        return pd.Series(3.0, index=df.index)
    km_min, km_max = km_known.min(), km_known.max()
    return pd.Series(
        np.where(km.isna(), 3.0, ((km - km_min) / (km_max - km_min)) * 5),
        index=df.index,
    )


def _score_condition(df: pd.DataFrame) -> pd.Series:
    mapping = {"Excellent": 1, "Very good": 2, "Good": 4}
    return df["condition"].map(mapping).fillna(3)


def _score_age(df: pd.DataFrame) -> pd.Series:
    mapping = {2024: 1, 2023: 2, 2022: 3, 2021: 4}
    return df["year"].map(mapping).fillna(4)


def load_and_score(csv_path: str = None) -> pd.DataFrame:
    # base = weighted sum of price + mileage + condition + age
    # penalty = days_on_market * DAY_PENALTY_PER_DAY
    # final = clamp(round(base + penalty), 0, 5)
    csv_path = csv_path or cfg.CSV_PATH
    _persist_columns(csv_path)
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)

    w = WEIGHTS
    base = (
        w["price"]     * _score_price(df)
        + w["mileage"] * _score_mileage(df)
        + w["condition"] * _score_condition(df)
        + w["age"]     * _score_age(df)
    )

    # Future: subtract popularity (popular bikes are easier to sell)
    # if "popularity" in df.columns:
    #     base -= w["popularity"] * df["popularity"].fillna(0)

    penalty = df["days_on_market"] * DAY_PENALTY_PER_DAY
    df["sell_difficulty_score"] = (base + penalty).round().astype(int).clip(0, 5)
    return df


def filter_hard_to_sell(df: pd.DataFrame, threshold: int = None) -> pd.DataFrame:
    # Recursively lowers threshold if nothing matches
    threshold = threshold if threshold is not None else cfg.HARD_SELL_THRESHOLD
    available = df[df["status"] != "sold"]
    hard = available[available["sell_difficulty_score"] >= threshold]
    if hard.empty and threshold > 0:
        return filter_hard_to_sell(df, threshold - 1)
    return hard


# Status management

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
    # +1 day for every available bike. Called once per campaign.
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    mask = df["status"] != "sold"
    df.loc[mask, "days_on_market"] = df.loc[mask, "days_on_market"] + 1
    df.to_csv(csv_path, index=False)


# Sales simulation
# P(sold) = AUTO_SELL_PROBABILITY * (6 - score) / 5

def simulate_sales(bike_ids: list[int], scores: dict[int, int],
                   trial_num: int, csv_path: str = None,
                   trials_path: str = None) -> list[int]:
    csv_path = csv_path or cfg.CSV_PATH
    trials_path = trials_path or cfg.CAMPAIGNS_PATH
    base = cfg.AUTO_SELL_PROBABILITY
    sold = []
    for bid in bike_ids:
        score = scores.get(bid, 3)
        prob = base * (6 - score) / 5
        if random.random() < prob:
            sold.append(bid)
            try:
                mark_bike_sold(bid, csv_path)
            except Exception as e:
                print(f"  WARN: failed to mark bike {bid} as sold: {e}")

    if sold:
        try:
            _stamp_sold_in_log(sold, trial_num, trials_path)
        except Exception as e:
            print(f"  WARN: failed to stamp sold in log: {e}")
        try:
            _record_sale_insights(sold, trial_num, csv_path, trials_path)
        except Exception as e:
            print(f"  WARN: knowledge base update failed: {e}")
    return sold


def _record_sale_insights(sold_ids: list[int], trial_num: int,
                          csv_path: str, trials_path: str) -> None:
    # Ask LLM why each sold bike likely sold, save to knowledge base
    try:
        from agents import summarize_sale_reason
    except Exception as e:
        print(f"  WARN: could not import summarize_sale_reason: {e}")
        return

    try:
        bikes = load_and_score(csv_path)
        campaigns = load_campaigns(trials_path)
    except Exception as e:
        print(f"  WARN: could not load data for sale insights: {e}")
        return

    rows = []
    for bid in sold_ids:
        try:
            bike_match = bikes[bikes["id"] == bid]
            if bike_match.empty:
                continue
            b = bike_match.iloc[0]

            bike_camps = campaigns[campaigns["bike_id"] == bid].sort_values("trial_num")
            last = bike_camps.iloc[-1].to_dict() if not bike_camps.empty else {}
            n_campaigns = len(bike_camps)

            try:
                note = summarize_sale_reason(b, last, n_campaigns)
            except Exception as e:
                note = f"(auto-note unavailable: {e})"

            rows.append({
                "bike_id": int(bid),
                "trial_num": int(trial_num),
                "date": str(last.get("date", "")),
                "title": b.get("title", ""),
                "brand": b.get("brand", ""),
                "category": b.get("category", ""),
                "price": float(b.get("price", 0) or 0),
                "sell_difficulty_score": int(b.get("sell_difficulty_score", 0) or 0)
                    if "sell_difficulty_score" in b else 0,
                "days_on_market": int(b.get("days_on_market", 0) or 0),
                "campaigns_run": int(n_campaigns),
                "selling_angle": last.get("selling_angle", ""),
                "target_audience": last.get("target_audience", ""),
                "tone": last.get("tone", ""),
                "reason_note": note,
            })
        except Exception as e:
            print(f"  WARN: skipping insight for bike {bid}: {e}")

    if rows:
        try:
            append_knowledge_base(rows)
        except Exception as e:
            print(f"  WARN: failed to write knowledge base: {e}")


# Knowledge base

def _ensure_knowledge_base(path: str = None) -> None:
    path = path or cfg.KNOWLEDGE_BASE_PATH
    if not os.path.exists(path):
        pd.DataFrame(columns=KB_COLUMNS).to_csv(path, index=False)


def load_knowledge_base(path: str = None) -> pd.DataFrame:
    path = path or cfg.KNOWLEDGE_BASE_PATH
    _ensure_knowledge_base(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=KB_COLUMNS)
    return df


def append_knowledge_base(records: list[dict], path: str = None) -> None:
    try:
        path = path or cfg.KNOWLEDGE_BASE_PATH
        _ensure_knowledge_base(path)
        new_df = pd.DataFrame(records, columns=KB_COLUMNS)
        new_df.to_csv(path, mode="a", header=False, index=False)
    except Exception as e:
        print(f"  WARN: could not append to knowledge base: {e}")


def _stamp_sold_in_log(sold_ids: list[int], trial_num: int,
                       path: str = None) -> None:
    path = path or cfg.CAMPAIGNS_PATH
    df = pd.read_csv(path)
    if "sold_in_campaign" not in df.columns:
        df["sold_in_campaign"] = ""
    df["sold_in_campaign"] = df["sold_in_campaign"].fillna("").astype(str)
    mask = (df["trial_num"] == trial_num) & (df["bike_id"].isin(sold_ids))
    df.loc[mask, "sold_in_campaign"] = "yes"
    df.to_csv(path, index=False)


def unsold_bike_ids(csv_path: str = None) -> list[int]:
    csv_path = csv_path or cfg.CSV_PATH
    df = pd.read_csv(csv_path)
    df = _ensure_columns(df)
    return df[df["status"] != "sold"]["id"].tolist()


# Campaigns log

def _ensure_campaigns(path: str = None) -> None:
    path = path or cfg.CAMPAIGNS_PATH
    if not os.path.exists(path):
        pd.DataFrame(columns=CAMPAIGN_COLUMNS).to_csv(path, index=False)


def load_campaigns(path: str = None) -> pd.DataFrame:
    path = path or cfg.CAMPAIGNS_PATH
    _ensure_campaigns(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=CAMPAIGN_COLUMNS)
    return df


def next_campaign_number(path: str = None) -> int:
    path = path or cfg.CAMPAIGNS_PATH
    df = load_campaigns(path)
    if df.empty:
        return 1
    return int(df["trial_num"].max()) + 1


def log_campaign(records: list[dict], path: str = None) -> None:
    try:
        path = path or cfg.CAMPAIGNS_PATH
        _ensure_campaigns(path)
        new_df = pd.DataFrame(records, columns=CAMPAIGN_COLUMNS)
        new_df.to_csv(path, mode="a", header=False, index=False)
    except Exception as e:
        print(f"  WARN: could not write campaign log: {e}")


def load_bike_campaign_history(bike_id: int, path: str = None) -> list[dict]:
    path = path or cfg.CAMPAIGNS_PATH
    df = load_campaigns(path)
    if df.empty:
        return []
    bike_rows = df[df["bike_id"] == bike_id].sort_values("trial_num")
    return bike_rows.to_dict(orient="records")


def join_bikes_and_campaigns(csv_path: str = None,
                             campaigns_path: str = None) -> pd.DataFrame:
    csv_path = csv_path or cfg.CSV_PATH
    campaigns_path = campaigns_path or cfg.CAMPAIGNS_PATH
    bikes = load_and_score(csv_path)
    campaigns = load_campaigns(campaigns_path)
    if campaigns.empty:
        return bikes
    return bikes.merge(campaigns, left_on="id", right_on="bike_id", how="left")
