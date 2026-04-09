"""
movelo Refurbished Bike Marketing Pipeline (PoC)

Usage:
    1. Put your Gemini API key in .env  (GOOGLE_API_KEY=...)
    2. source venv/bin/activate
    3. python main.py

Each run = one "trial". The pipeline:
  - Scores bikes, picks the hard-to-sell ones
  - Marketing Manager reads bike data + past trial history, writes new strategy
  - Marketer generates captions, emails, image prompts
  - Images generated and saved to output/trial_N/
  - Trial logged back into the CSV as a new campaign_trial_N column

Output:
    output/trial_N/
      bike_<id>_<slug>/
        content.json
        urban.png
        nature.png
"""

import json
import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

from scoring import load_and_score, filter_hard_to_sell, next_trial_number, save_trial
from agents import run_manager_agent, run_marketer_agent, generate_images

SEP = "=" * 60


def _check_api_key():
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key or key == "your-gemini-api-key-here":
        print("ERROR: GOOGLE_API_KEY is not set.")
        print("  Edit .env and paste your Gemini API key, then re-run.")
        sys.exit(1)


def main():
    _check_api_key()

    # ------ Step 1: Score ------
    print(SEP)
    print("STEP 1 -- Scoring all bikes")
    print(SEP)
    df = load_and_score()
    trial_num = next_trial_number(df)
    print(f"  This is TRIAL {trial_num}")

    cols = ["id", "title", "price", "condition", "sell_difficulty_score"]
    top = df[cols].sort_values("sell_difficulty_score", ascending=False).head(8)
    print(top.to_string(index=False))
    print(f"  ({len(df)} bikes total)\n")

    # ------ Step 2: Filter ------
    print(SEP)
    print("STEP 2 -- Filtering hard-to-sell bikes")
    print(SEP)
    hard = filter_hard_to_sell(df)
    min_score = hard["sell_difficulty_score"].min()
    print(f"  {len(hard)} bikes with score >= {min_score}:")
    for _, r in hard.iterrows():
        print(f"    [{r['id']}] {r['title']}  (EUR {r['price']:.0f}, score {r['sell_difficulty_score']})")
    print()

    # ------ Step 3: Marketing Manager ------
    print(SEP)
    print("STEP 3 -- Marketing Manager agent")
    print(SEP)
    briefs = run_manager_agent(hard)
    for b in briefs:
        print(f"  Bike {b.get('bike_id')}: {b.get('selling_angle', '?')[:80]}")
    print()

    # ------ Step 4: Marketer ------
    print(SEP)
    print("STEP 4 -- Marketer agent")
    print(SEP)
    content = run_marketer_agent(briefs, hard)
    for c in content:
        print(f"  Bike {c.get('bike_id')}: \"{c.get('email_subject', '')}\"")
    print()

    # ------ Step 5: Generate images ------
    print(SEP)
    print(f"STEP 5 -- Generating images (saved to output/trial_{trial_num}/)")
    print(SEP)
    img_results = generate_images(content, hard, trial_num)
    print()

    # ------ Step 6: Log trial to CSV ------
    print(SEP)
    print("STEP 6 -- Logging trial to CSV")
    print(SEP)

    today = date.today().isoformat()

    brief_lookup = {b.get("bike_id"): b for b in briefs}
    content_lookup = {c.get("bike_id"): c for c in content}

    trial_data = {}
    for _, row in hard.iterrows():
        bid = int(row["id"])
        b = brief_lookup.get(bid, {})
        c = content_lookup.get(bid, {})
        imgs = img_results.get(bid, {})

        thought = b.get("selling_angle", "N/A")
        actions = []
        if c.get("instagram_caption"):
            actions.append("instagram_post")
        if c.get("email_body"):
            actions.append("email")
        n_imgs = len(imgs.get("images", []))
        if n_imgs:
            actions.append(f"{n_imgs}_images")

        entry = f"{today} | {thought} | {', '.join(actions)}"
        trial_data[bid] = entry

    col_name = save_trial(df, trial_num, trial_data)
    print(f"  Saved column '{col_name}' to CSV")
    for bid, entry in trial_data.items():
        print(f"    Bike {bid}: {entry}")
    print()

    # ------ Summary ------
    print(SEP)
    print(f"DONE -- Trial {trial_num} complete")
    print(SEP)
    print(f"  CSV updated: {col_name}")
    for bid, info in img_results.items():
        n = len(info["images"])
        print(f"  {info['folder']}/  ({n} images + content.json)")
    print()


if __name__ == "__main__":
    main()
