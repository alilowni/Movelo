"""
movelo Refurbished Bike Marketing Pipeline

Usage:
    python main.py                 Run one campaign (advances 1 day)
    python main.py --test-api      Test API connections only
    python main.py --dashboard     Launch the Streamlit dashboard

All settings are controlled via .env (see .env.example).
"""

import argparse
import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

import config as cfg
from pipeline import Pipeline
from scoring import (
    load_and_score,
    filter_hard_to_sell,
    next_trial_number,
    log_trial_to_csv,
    advance_day,
    simulate_sales,
)
from agents import (
    run_manager_agent,
    run_marketer_agent,
    generate_images,
    test_llm_api,
    test_image_api,
)


def preflight() -> None:
    if not cfg.GOOGLE_API_KEY or cfg.GOOGLE_API_KEY == "your-gemini-api-key-here":
        print("ERROR: GOOGLE_API_KEY not set. Edit .env and re-run.")
        sys.exit(1)


def run_api_tests() -> bool:
    print("Testing LLM API ...", end=" ", flush=True)
    llm_ok = test_llm_api()
    print("OK" if llm_ok else "FAILED")

    print("Testing Image API ...", end=" ", flush=True)
    img_ok = test_image_api()
    print("OK" if img_ok else "FAILED")

    if llm_ok and img_ok:
        print("All API tests passed.")
    else:
        print("Some API tests failed.")
    return llm_ok and img_ok


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_advance_day(ctx: dict) -> dict:
    advance_day()
    print("  +1 day for all available bikes")
    return ctx


def step_score(ctx: dict) -> dict:
    df = load_and_score()
    trial_num = next_trial_number()
    avail = df[df["status"] != "sold"]
    n_sold = len(df) - len(avail)
    danger = len(avail[avail["sell_difficulty_score"] >= cfg.HARD_SELL_THRESHOLD])
    max_day = int(avail["days_on_market"].max()) if not avail.empty else 0
    print(f"  Campaign {trial_num} | {len(avail)} available, {n_sold} sold | "
          f"{danger} in danger (>={cfg.HARD_SELL_THRESHOLD}) | day {max_day}")
    ctx["df"] = df
    ctx["trial_num"] = trial_num
    return ctx


def step_filter(ctx: dict) -> dict:
    hard = filter_hard_to_sell(ctx["df"])
    ids = [int(r["id"]) for _, r in hard.iterrows()]
    print(f"  {len(hard)} bikes targeted: {ids}")
    ctx["hard"] = hard
    return ctx


def step_manager(ctx: dict) -> dict:
    briefs = run_manager_agent(ctx["hard"])
    for b in briefs:
        print(f"  bike {b.get('bike_id')}: {b.get('selling_angle', '?')[:60]}")
    ctx["briefs"] = briefs
    return ctx


def step_marketer(ctx: dict) -> dict:
    content = run_marketer_agent(ctx["briefs"], ctx["hard"])
    for c in content:
        subj = c.get("email_subject", "") or "(no email)"
        print(f"  bike {c.get('bike_id')}: {subj}")
    ctx["content"] = content
    return ctx


def step_images(ctx: dict) -> dict:
    img_results = generate_images(ctx["content"], ctx["hard"], ctx["trial_num"])
    for bid, info in img_results.items():
        print(f"  bike {bid}: {len(info['images'])} images")
    ctx["img_results"] = img_results
    return ctx


def step_log(ctx: dict) -> dict:
    today = date.today().isoformat()
    trial_num = ctx["trial_num"]
    briefs = ctx["briefs"]
    content = ctx["content"]
    img_results = ctx["img_results"]
    hard = ctx["hard"]

    brief_lookup = {b.get("bike_id"): b for b in briefs}
    content_lookup = {c.get("bike_id"): c for c in content}

    records = []
    for _, row in hard.iterrows():
        bid = int(row["id"])
        b = brief_lookup.get(bid, {})
        c = content_lookup.get(bid, {})
        imgs = img_results.get(bid, {})

        actions = []
        if c.get("instagram_caption"):
            actions.append("instagram_post")
        if c.get("email_body"):
            actions.append("email")
        n_imgs = len(imgs.get("images", []))
        if n_imgs:
            actions.append(f"{n_imgs}_images")

        image_paths = imgs.get("images", [])
        urban_path = next((p for p in image_paths if "urban" in p), "")
        nature_path = next((p for p in image_paths if "nature" in p), "")

        records.append({
            "trial_num": trial_num,
            "bike_id": bid,
            "date": today,
            "selling_angle": b.get("selling_angle", ""),
            "target_audience": b.get("target_audience", ""),
            "tone": b.get("tone", ""),
            "actions": ", ".join(actions),
            "instagram_caption": c.get("instagram_caption", "") or "",
            "email_subject": c.get("email_subject", "") or "",
            "email_body": c.get("email_body", "") or "",
            "image_prompt_a": c.get("image_prompt_a", "") or "",
            "image_prompt_b": c.get("image_prompt_b", "") or "",
            "urban_image_path": urban_path,
            "nature_image_path": nature_path,
            "sold_in_campaign": "",
        })

    log_trial_to_csv(records)
    print(f"  {len(records)} records logged")
    ctx["records"] = records
    return ctx


def step_auto_sell(ctx: dict) -> dict:
    hard = ctx["hard"]
    bike_ids = [int(r["id"]) for _, r in hard.iterrows()]
    scores = {int(r["id"]): int(r["sell_difficulty_score"]) for _, r in hard.iterrows()}
    sold = simulate_sales(bike_ids, scores, trial_num=ctx["trial_num"])
    if sold:
        print(f"  Auto-sold: {sold}")
    else:
        print(f"  No auto-sales this round")
    ctx["auto_sold"] = sold
    return ctx


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    pipe = Pipeline()
    pipe.register_step("Advance day", step_advance_day)
    pipe.register_step("Score bikes", step_score)
    pipe.register_step("Filter", step_filter)
    pipe.register_step("Manager", step_manager)
    pipe.register_step("Marketer", step_marketer)
    pipe.register_step("Images", step_images)
    pipe.register_step("Log", step_log)
    pipe.register_step("Sales", step_auto_sell)
    return pipe


def run_once() -> None:
    preflight()
    print("API tests...")
    if not run_api_tests():
        sys.exit(1)
    print()
    build_pipeline().run()


def run_dashboard() -> None:
    os.execvp("streamlit", ["streamlit", "run", "dashboard.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="movelo marketing pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-api", action="store_true",
                       help="Test API connections and exit")
    group.add_argument("--dashboard", action="store_true",
                       help="Launch the Streamlit dashboard")
    args = parser.parse_args()

    if args.test_api:
        preflight()
        ok = run_api_tests()
        sys.exit(0 if ok else 1)
    elif args.dashboard:
        run_dashboard()
    else:
        run_once()


if __name__ == "__main__":
    main()
