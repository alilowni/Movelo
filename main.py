# Movelo — Refurbished Bike Marketing Pipeline
#
# python main.py              Run one campaign (advances 1 day)
# python main.py --test-api   Test API connections only
# python main.py --dashboard  Launch the Streamlit dashboard

import argparse
import os
import sys
from datetime import date

from dotenv import load_dotenv

load_dotenv()

import config as cfg
from pipeline import Pipeline
from scoring import (
    init_working_files,
    load_and_score,
    filter_hard_to_sell,
    next_campaign_number,
    log_campaign,
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
    print("API check: ", end="", flush=True)
    llm_ok = test_llm_api()
    print(f"LLM {'ok' if llm_ok else 'FAIL'}", end=" | ", flush=True)
    img_ok = test_image_api()
    print(f"Image {'ok' if img_ok else 'FAIL'}")
    if not (llm_ok and img_ok):
        print("  Some API tests failed.")
    return llm_ok and img_ok


# Pipeline steps

def step_advance_day(ctx: dict) -> dict:
    advance_day()
    return ctx


def step_score(ctx: dict) -> dict:
    df = load_and_score()
    trial_num = next_campaign_number()
    avail = df[df["status"] != "sold"]
    n_sold = len(df) - len(avail)
    danger = len(avail[avail["sell_difficulty_score"] >= cfg.HARD_SELL_THRESHOLD])
    max_day = int(avail["days_on_market"].max()) if not avail.empty else 0
    print(f"#{trial_num} | {len(avail)} avail, {n_sold} sold, "
          f"{danger} in danger | day {max_day}", end="", flush=True)
    ctx["df"] = df
    ctx["trial_num"] = trial_num
    return ctx


def step_filter(ctx: dict) -> dict:
    hard = filter_hard_to_sell(ctx["df"])
    cap = cfg.MAX_BIKES_PER_CAMPAIGN
    total_danger = len(hard)
    if len(hard) > cap:
        hard = hard.sort_values("sell_difficulty_score", ascending=False).head(cap)
    ids = [int(r["id"]) for _, r in hard.iterrows()]
    msg = f"{len(hard)} bikes"
    if total_danger > cap:
        msg += f" (of {total_danger} in danger)"
    print(f"{msg}: {ids}", end="", flush=True)
    ctx["hard"] = hard
    return ctx


def step_manager(ctx: dict) -> dict:
    briefs = run_manager_agent(ctx["hard"])
    if not briefs:
        print("no briefs (manager failed)", end="", flush=True)
    else:
        print(f"{len(briefs)} briefs", end="", flush=True)
    ctx["briefs"] = briefs
    return ctx


def step_marketer(ctx: dict) -> dict:
    if not ctx.get("briefs"):
        print("skipped (no briefs)", end="", flush=True)
        ctx["content"] = []
        return ctx
    content = run_marketer_agent(ctx["briefs"], ctx["hard"])
    if not content:
        print("no content (marketer failed)", end="", flush=True)
    else:
        print(f"{len(content)} content pieces", end="", flush=True)
    ctx["content"] = content
    return ctx


def step_images(ctx: dict) -> dict:
    if not ctx.get("content"):
        print("skipped (no content)", end="", flush=True)
        ctx["img_results"] = {}
        return ctx
    img_results = generate_images(ctx["content"], ctx["hard"], ctx["trial_num"])
    total_imgs = sum(len(info["images"]) for info in img_results.values())
    print(f"{total_imgs} images for {len(img_results)} bikes", end="", flush=True)
    ctx["img_results"] = img_results
    return ctx


def step_log(ctx: dict) -> dict:
    today = date.today().isoformat()
    trial_num = ctx["trial_num"]
    briefs = ctx.get("briefs", [])
    content = ctx.get("content", [])
    img_results = ctx.get("img_results", {})
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

    log_campaign(records)
    print(f"{len(records)} records", end="", flush=True)
    ctx["records"] = records
    return ctx


def step_auto_sell(ctx: dict) -> dict:
    hard = ctx["hard"]
    bike_ids = [int(r["id"]) for _, r in hard.iterrows()]
    scores = {int(r["id"]): int(r["sell_difficulty_score"]) for _, r in hard.iterrows()}
    try:
        sold = simulate_sales(bike_ids, scores, trial_num=ctx["trial_num"])
    except Exception as e:
        print(f"error ({e})", end="", flush=True)
        ctx["auto_sold"] = []
        return ctx
    if sold:
        print(f"sold {len(sold)}: {sold}", end="", flush=True)
    else:
        print(f"no sales", end="", flush=True)
    ctx["auto_sold"] = sold
    return ctx


# Pipeline assembly

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
    init_working_files()
    if not run_api_tests():
        sys.exit(1)
    build_pipeline().run()


def run_dashboard() -> None:
    os.execvp("streamlit", ["streamlit", "run", "dashboard.py"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Movelo marketing pipeline")
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
