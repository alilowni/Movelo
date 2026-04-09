"""
movelo Refurbished Bike Marketing Pipeline

Usage:
    python main.py                 Run one campaign
    python main.py --tick          Advance 1 simulated day + run campaign
    python main.py --tick 5        Advance 5 simulated days + run campaign
    python main.py --test-api      Test API connections only
    python main.py --loop          Run continuously (every LOOP_INTERVAL_SEC)
    python main.py --dashboard     Launch the Streamlit dashboard

All settings are controlled via .env (see .env.example).
"""

import argparse
import os
import sys
import time
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
    increment_days_on_market,
    simulate_sales,
)
from agents import (
    run_manager_agent,
    run_marketer_agent,
    generate_images,
    test_llm_api,
    test_image_api,
)


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def preflight() -> None:
    if not cfg.GOOGLE_API_KEY or cfg.GOOGLE_API_KEY == "your-gemini-api-key-here":
        print("ERROR: GOOGLE_API_KEY not set. Edit .env and re-run.")
        sys.exit(1)


def run_api_tests() -> bool:
    """Test both APIs. Returns True if all pass."""
    print("Testing LLM API ...", end=" ", flush=True)
    llm_ok = test_llm_api()
    print("OK" if llm_ok else "FAILED")

    print("Testing Image API ...", end=" ", flush=True)
    img_ok = test_image_api()
    print("OK" if img_ok else "FAILED")

    if llm_ok and img_ok:
        print("All API tests passed.")
    else:
        print("Some API tests failed -- check your key and network.")

    return llm_ok and img_ok


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_score(ctx: dict) -> dict:
    df = load_and_score()
    trial_num = next_trial_number()
    n_avail = len(df[df["status"] != "sold"])
    n_sold = len(df[df["status"] == "sold"])
    print(f"  Campaign {trial_num} | {len(df)} bikes ({n_avail} available, {n_sold} sold)")

    ctx["df"] = df
    ctx["trial_num"] = trial_num
    return ctx


def step_filter(ctx: dict) -> dict:
    hard = filter_hard_to_sell(ctx["df"])
    ids = [int(r["id"]) for _, r in hard.iterrows()]
    print(f"  {len(hard)} hard-to-sell bikes: {ids}")

    ctx["hard"] = hard
    return ctx


def step_manager(ctx: dict) -> dict:
    briefs = run_manager_agent(ctx["hard"])
    for b in briefs:
        why = b.get("why_different", "")
        angle = b.get("selling_angle", "?")[:60]
        print(f"  bike {b.get('bike_id')}: {angle}")
        if why:
            print(f"    (vs. past: {why[:80]})")

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
        n = len(info["images"])
        print(f"  bike {bid}: {n} images -> {info['folder']}")

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
        })

    log_trial_to_csv(records)
    print(f"  {len(records)} records -> trials_log.csv")

    ctx["records"] = records
    return ctx


def step_auto_sell(ctx: dict) -> dict:
    """Roll dice for each marketed bike -- some may sell after the campaign."""
    hard = ctx["hard"]
    bike_ids = [int(r["id"]) for _, r in hard.iterrows()]
    scores = {int(r["id"]): int(r["sell_difficulty_score"]) for _, r in hard.iterrows()}
    sold = simulate_sales(bike_ids, scores)
    if sold:
        print(f"  Sold after campaign: {sold}")
    else:
        print(f"  No bikes sold this round (probability-based)")
    ctx["auto_sold"] = sold
    return ctx


# ---------------------------------------------------------------------------
# Build the pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    pipe = Pipeline()
    pipe.register_step("Score bikes", step_score)
    pipe.register_step("Filter hard-to-sell", step_filter)
    pipe.register_step("Marketing Manager", step_manager)
    pipe.register_step("Marketer", step_marketer)
    pipe.register_step("Generate images", step_images)
    pipe.register_step("Log campaign", step_log)
    pipe.register_step("Simulate sales", step_auto_sell)
    return pipe


# ---------------------------------------------------------------------------
# Run modes
# ---------------------------------------------------------------------------

def run_once() -> None:
    preflight()
    print("Running API tests before pipeline...")
    if not run_api_tests():
        sys.exit(1)
    print()
    build_pipeline().run()


def run_tick(days: int = 1) -> None:
    """Advance N simulated days (increment scores), then run one campaign."""
    preflight()
    print("Running API tests...")
    if not run_api_tests():
        sys.exit(1)

    print(f"\nAdvancing {days} simulated day(s)...")
    for _ in range(days):
        increment_days_on_market()

    df = load_and_score()
    n_avail = len(df[df["status"] != "sold"])
    if n_avail == 0:
        print("All bikes are sold! Nothing to do.")
        return

    print(f"  {n_avail} available bikes (scores updated with +{days} days)\n")
    build_pipeline().run()


def run_loop() -> None:
    preflight()
    print("Running API tests before starting loop...")
    if not run_api_tests():
        sys.exit(1)

    interval = cfg.LOOP_INTERVAL_SEC
    iteration = 0
    print(f"\nScheduler loop (interval={interval}s, each tick = 1 simulated day)")
    print("Press Ctrl+C to stop.\n")

    while True:
        iteration += 1
        print(f"{'='*50}")
        print(f"Day {iteration}")
        print(f"{'='*50}")

        increment_days_on_market()

        df = load_and_score()
        n_avail = len(df[df["status"] != "sold"])
        if n_avail == 0:
            print("All bikes are sold! Stopping.")
            break

        try:
            build_pipeline().run()
        except SystemExit:
            print("  Pipeline step failed, will retry next tick.")
        except Exception as e:
            print(f"  Pipeline error: {e}, will retry next tick.")

        print(f"\nSleeping {interval}s...\n")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


def run_dashboard() -> None:
    os.execvp("streamlit", ["streamlit", "run", "dashboard.py"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="movelo marketing pipeline")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--test-api", action="store_true",
                       help="Test API connections and exit")
    group.add_argument("--tick", nargs="?", const=1, type=int, metavar="DAYS",
                       help="Advance N simulated days (default 1) + run campaign")
    group.add_argument("--loop", action="store_true",
                       help=f"Run continuously every {cfg.LOOP_INTERVAL_SEC}s")
    group.add_argument("--dashboard", action="store_true",
                       help="Launch the Streamlit dashboard")
    args = parser.parse_args()

    if args.test_api:
        preflight()
        ok = run_api_tests()
        sys.exit(0 if ok else 1)
    elif args.tick is not None:
        run_tick(args.tick)
    elif args.loop:
        run_loop()
    elif args.dashboard:
        run_dashboard()
    else:
        run_once()


if __name__ == "__main__":
    main()
