import pandas as pd
import numpy as np

CSV_PATH = "sample_data_movelo_links.csv"


def load_and_score(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load bike CSV and add a 0-5 sell_difficulty_score column."""
    df = pd.read_csv(csv_path)

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

    df["sell_difficulty_score"] = (
        0.30 * df["_price_score"]
        + 0.30 * df["_mileage_score"]
        + 0.20 * df["_condition_score"]
        + 0.20 * df["_age_score"]
    ).round().astype(int).clip(0, 5)

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True)
    return df


def get_trial_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted list of existing campaign_trial_N columns."""
    return sorted([c for c in df.columns if c.startswith("campaign_trial_")])


def next_trial_number(df: pd.DataFrame) -> int:
    """Return the next trial number (1-based)."""
    existing = get_trial_columns(df)
    if not existing:
        return 1
    nums = [int(c.split("_")[-1]) for c in existing]
    return max(nums) + 1


def filter_hard_to_sell(df: pd.DataFrame, threshold: int = 5) -> pd.DataFrame:
    """Return bikes at or above the threshold. Falls back to lower if none hit it."""
    hard = df[df["sell_difficulty_score"] >= threshold]
    if hard.empty and threshold > 0:
        return filter_hard_to_sell(df, threshold - 1)
    return hard


def save_trial(df: pd.DataFrame, trial_num: int, trial_data: dict,
               csv_path: str = CSV_PATH):
    """Write a new campaign_trial_N column back to the CSV.

    trial_data: {bike_id: "date | thought | action"}
    """
    col_name = f"campaign_trial_{trial_num}"
    full_df = pd.read_csv(csv_path)

    full_df[col_name] = ""
    for bike_id, entry in trial_data.items():
        mask = full_df["id"] == int(bike_id)
        full_df.loc[mask, col_name] = entry

    full_df.to_csv(csv_path, index=False)
    return col_name


if __name__ == "__main__":
    df = load_and_score()
    print(df[["id", "title", "price", "km_ridden", "condition", "year", "sell_difficulty_score"]]
          .sort_values("sell_difficulty_score", ascending=False)
          .to_string(index=False))
