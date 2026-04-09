"""
One-time migration: read existing output/trial_N/bike_*/content.json files
and backfill them into trials_log.csv.

Also strips any campaign_trial_* columns from the master bike CSV.

Usage:
    python migrate_trials.py
"""

import json
import os
import re
from pathlib import Path

import pandas as pd

import config as cfg
from scoring import (
    TRIALS_LOG_COLUMNS,
    log_trial_to_csv,
    load_trials,
)

OUTPUT_DIR = Path("output")


def _extract_bike_id(folder_name: str) -> int | None:
    m = re.match(r"bike_(\d+)_", folder_name)
    return int(m.group(1)) if m else None


def backfill_from_output() -> int:
    """Scan output/trial_N/ folders and create trials_log.csv rows."""
    if not OUTPUT_DIR.exists():
        print("No output/ directory found -- nothing to migrate.")
        return 0

    existing = load_trials()
    existing_keys = set()
    if not existing.empty:
        existing_keys = set(
            zip(existing["trial_num"].astype(int), existing["bike_id"].astype(int))
        )

    records: list[dict] = []
    for trial_dir in sorted(OUTPUT_DIR.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue
        trial_num = int(trial_dir.name.split("_")[1])

        for bike_dir in sorted(trial_dir.iterdir()):
            if not bike_dir.is_dir():
                continue
            bike_id = _extract_bike_id(bike_dir.name)
            if bike_id is None:
                continue
            if (trial_num, bike_id) in existing_keys:
                continue

            content_path = bike_dir / "content.json"
            if not content_path.exists():
                continue

            with open(content_path) as f:
                c = json.load(f)

            urban_path = bike_dir / "urban.png"
            nature_path = bike_dir / "nature.png"

            record = {
                "trial_num": trial_num,
                "bike_id": bike_id,
                "date": "",
                "selling_angle": "",
                "target_audience": "",
                "tone": "",
                "actions": "",
                "instagram_caption": c.get("instagram_caption", "") or "",
                "email_subject": c.get("email_subject", "") or "",
                "email_body": c.get("email_body", "") or "",
                "image_prompt_a": c.get("image_prompt_a", "") or "",
                "image_prompt_b": c.get("image_prompt_b", "") or "",
                "urban_image_path": str(urban_path) if urban_path.exists() else "",
                "nature_image_path": str(nature_path) if nature_path.exists() else "",
            }
            records.append(record)

    if records:
        log_trial_to_csv(records)
        print(f"  Backfilled {len(records)} records into {cfg.TRIALS_LOG_PATH}")
    else:
        print("  No new records to backfill.")

    return len(records)


def strip_trial_columns_from_master() -> None:
    """Remove any campaign_trial_* columns from the master bike CSV."""
    df = pd.read_csv(cfg.CSV_PATH)
    trial_cols = [c for c in df.columns if c.startswith("campaign_trial_")]
    if not trial_cols:
        print("  No campaign_trial_* columns to strip from master CSV.")
        return
    df.drop(columns=trial_cols, inplace=True)
    df.to_csv(cfg.CSV_PATH, index=False)
    print(f"  Stripped columns from master CSV: {trial_cols}")


if __name__ == "__main__":
    print("=== Backfilling trials from output/ folders ===")
    backfill_from_output()
    print()
    print("=== Cleaning master CSV ===")
    strip_trial_columns_from_master()
    print()
    print("Done. Check trials_log.csv for the migrated data.")
