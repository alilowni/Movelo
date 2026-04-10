# AI agents — manager, marketer, image gen, sale reason analysis.

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests
from google import genai
from google.genai import types as genai_types
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

import config as cfg
import prompts
from scoring import load_bike_campaign_history

OUTPUT_DIR = Path(cfg.OUTPUT_DIR)
IMAGE_CACHE_DIR = OUTPUT_DIR / ".image_cache"


def _make_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=cfg.LLM_MODEL, temperature=temperature)


# API health checks

def test_llm_api() -> bool:
    try:
        llm = _make_llm(temperature=0.0)
        resp = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        return "OK" in resp.content.upper()
    except Exception as e:
        print(f"  LLM test failed: {e}")
        return False


def test_image_api() -> bool:
    try:
        client = genai.Client()
        resp = client.models.generate_content(
            model=cfg.IMAGE_MODEL,
            contents="Generate a tiny 64x64 solid blue square image.",
            config=genai.types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )
        for part in resp.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                return True
        return False
    except Exception as e:
        print(f"  Image API test failed: {e}")
        return False


# Bike context builders

def _bike_context(row: pd.Series) -> str:
    # Compact one-liner with key fields + short campaign history
    color = row.get("color", "")
    color_str = f", Color: {color}" if pd.notna(color) and color else ""
    km = row["km_ridden"]
    km_str = f"{int(km)} km" if pd.notna(km) else "unknown km"
    year = int(row["year"]) if pd.notna(row["year"]) else "unknown"
    desc = row.get("description", "")
    desc_str = f" | {desc[:150]}" if pd.notna(desc) and desc else ""

    line = (
        f"[{row['id']}] {row['title']} | {row['brand']} {row['category']} | "
        f"€{row['price']:.0f} | {row['condition']} | {year} | {km_str} | "
        f"Score {row['sell_difficulty_score']}/5 | Day {row.get('days_on_market', 0)}"
        f"{color_str}{desc_str}"
    )

    history = load_bike_campaign_history(int(row["id"]))
    if history:
        past = []
        for h in history:
            angle = h.get("selling_angle", "?")[:50]
            tone = h.get("tone", "?")
            audience = h.get("target_audience", "?")[:40]
            past.append(f"#{h.get('trial_num','?')}: {angle} | tone={tone} | audience={audience}")
        line += f"\n  PAST ({len(history)}x): " + " // ".join(past)

    return line


def _bike_image_context(row: pd.Series) -> str:
    # Visual details used in image generation prompts
    parts = [f"{row['brand']} {row['title']}"]
    cat = row.get("category", "")
    if pd.notna(cat) and cat:
        parts.append(f"category: {cat}")
        is_city = any(k in str(cat).lower() for k in ["city", "urban", "e-city"])
        parts.append(f"setting hint: {'urban/city' if is_city else 'nature/trail'}")
    color = row.get("color", "")
    if pd.notna(color) and color:
        parts.append(f"color: {color}")
    condition = row.get("condition", "")
    if pd.notna(condition):
        parts.append(f"condition: {condition}")
    frame = row.get("frame_size", "")
    if pd.notna(frame) and frame:
        parts.append(f"frame: {frame}")
    parts.append(f"brand on frame: {row['brand']}")
    return ", ".join(parts)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]


def _parse_json_response(raw: str, step_name: str) -> list[dict] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  WARN: {step_name} returned invalid JSON: {e}")
        return None
    if not isinstance(data, list):
        print(f"  WARN: {step_name} returned JSON but not an array.")
        return None
    return data


def trial_output_dir(trial_num: int) -> Path:
    d = OUTPUT_DIR / f"trial_{trial_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# Image download (with disk cache)

def _download_image(url: str) -> bytes | None:
    if not url:
        return None
    IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.md5(url.encode()).hexdigest()
    ext = Path(url.split("?")[0]).suffix or ".webp"
    cache_path = IMAGE_CACHE_DIR / f"{url_hash}{ext}"
    if cache_path.exists():
        return cache_path.read_bytes()
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        return resp.content
    except Exception as e:
        print(f"  WARN: image download failed ({e})")
        return None


def _mime_from_url(url: str) -> str:
    lower = url.lower().split("?")[0]
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    return "image/webp"


# Sale reason — short note on why a bike sold

def summarize_sale_reason(bike_row: pd.Series, last_campaign: dict,
                          campaigns_run: int) -> str:
    llm = _make_llm(temperature=0.3)
    km = bike_row.get("km_ridden")
    km_str = f"{int(km)} km" if pd.notna(km) else "unknown km"
    year = int(bike_row["year"]) if pd.notna(bike_row.get("year")) else "unknown"

    bike_line = (
        f"{bike_row.get('title','')} | {bike_row.get('brand','')} "
        f"{bike_row.get('category','')} | €{float(bike_row.get('price',0) or 0):.0f} | "
        f"{bike_row.get('condition','')} | {year} | {km_str} | "
        f"score {bike_row.get('sell_difficulty_score','?')}/5 | "
        f"days listed {bike_row.get('days_on_market',0)} | "
        f"campaigns run {campaigns_run}"
    )
    if last_campaign:
        camp_line = (
            f"Last campaign — angle: {last_campaign.get('selling_angle','?')} | "
            f"audience: {last_campaign.get('target_audience','?')} | "
            f"tone: {last_campaign.get('tone','?')} | "
            f"actions: {last_campaign.get('actions','?')}"
        )
    else:
        camp_line = "Last campaign — none (sold without targeted marketing)."

    payload = f"{bike_line}\n{camp_line}"
    try:
        response = llm.invoke([
            SystemMessage(content=prompts.SALE_REASON),
            HumanMessage(content=payload),
        ])
        text = (response.content or "").strip().strip('"').strip()
        return text or "(no reason produced)"
    except Exception as e:
        return f"(LLM error: {e})"


# Marketing Manager agent

def run_manager_agent(bikes_df: pd.DataFrame, max_retries: int = 2) -> list[dict]:
    llm = _make_llm(temperature=cfg.MANAGER_TEMPERATURE)
    bike_texts = "\n".join(_bike_context(row) for _, row in bikes_df.iterrows())
    for attempt in range(max_retries):
        try:
            response = llm.invoke([
                SystemMessage(content=prompts.MANAGER),
                HumanMessage(content=bike_texts),
            ])
        except Exception as e:
            print(f"  ERROR: Manager call failed: {e}")
            sys.exit(1)
        result = _parse_json_response(response.content, "Manager")
        if result is not None:
            return result
        print(f" retry {attempt + 1}", end="", flush=True)
    print("\nERROR: Manager failed after retries.")
    sys.exit(1)


# Marketer agent — processes briefs in batches of 5

MARKETER_BATCH_SIZE = 5
MARKETER_MAX_RETRIES = 2


def run_marketer_agent(briefs: list[dict], bikes_df: pd.DataFrame) -> list[dict]:
    llm = _make_llm(temperature=cfg.MARKETER_TEMPERATURE)

    bike_visual = {
        int(row["id"]): _bike_image_context(row) for _, row in bikes_df.iterrows()
    }

    all_results: list[dict] = []
    for i in range(0, len(briefs), MARKETER_BATCH_SIZE):
        batch = briefs[i : i + MARKETER_BATCH_SIZE]
        batch_ids = [b.get("bike_id") for b in batch]

        context_parts = []
        for brief in batch:
            bid = brief.get("bike_id")
            visual = bike_visual.get(bid, "no details")
            context_parts.append(
                f"Brief: {json.dumps(brief)}\nBike visual: {visual}"
            )
        payload = "\n---\n".join(context_parts)

        result = None
        for attempt in range(MARKETER_MAX_RETRIES):
            try:
                response = llm.invoke([
                    SystemMessage(content=prompts.MARKETER),
                    HumanMessage(content=payload),
                ])
            except Exception as e:
                print(f"  ERROR: Marketer call failed: {e}")
                sys.exit(1)
            result = _parse_json_response(response.content, "Marketer")
            if result is not None:
                break
            print(f" retry {attempt + 1}", end="", flush=True)

        if result is None:
            print(f"\nERROR: Marketer batch failed for bikes {batch_ids}")
            sys.exit(1)
        all_results.extend(result)

    return all_results


# Image generation

def generate_images(content_list: list[dict], bikes_df: pd.DataFrame,
                    trial_num: int) -> dict:
    try:
        client = genai.Client()
    except Exception as e:
        print(f"  ERROR: Could not create Gemini client: {e}")
        sys.exit(1)

    title_lookup = {int(row["id"]): row["title"] for _, row in bikes_df.iterrows()}
    image_url_lookup = {int(row["id"]): row.get("image_url", "") for _, row in bikes_df.iterrows()}
    visual_lookup = {int(row["id"]): _bike_image_context(row) for _, row in bikes_df.iterrows()}

    base_dir = trial_output_dir(trial_num)
    results = {}

    for item in content_list:
        bid = item.get("bike_id")
        title = title_lookup.get(bid, f"bike_{bid}")
        folder = base_dir / f"bike_{bid}_{_slug(title)}"
        folder.mkdir(parents=True, exist_ok=True)

        brief_path = folder / "content.json"
        brief_path.write_text(json.dumps(item, indent=2))

        bike_image_url = image_url_lookup.get(bid, "")
        bike_image_bytes = _download_image(bike_image_url)
        has_ref = bike_image_bytes is not None
        visual = visual_lookup.get(bid, "")

        paths = []
        for label, key in [("urban", "image_prompt_a"), ("nature", "image_prompt_b")]:
            prompt = item.get(key, "")
            if not prompt:
                continue

            fpath = folder / f"{label}.png"
            print(f"\n    {bid}/{label} ", end="", flush=True)
            try:
                contents = []
                brand = visual.split(",")[0].split()[0]
                brand_inst = prompts.BRAND_INSTRUCTION.format(brand=brand)
                if has_ref:
                    contents.append(genai_types.Part.from_bytes(
                        data=bike_image_bytes,
                        mime_type=_mime_from_url(bike_image_url),
                    ))
                    contents.append(genai_types.Part.from_text(
                        text=prompts.IMAGE_WITH_REF.format(
                            visual=visual, brand_inst=brand_inst, prompt=prompt,
                        )
                    ))
                else:
                    contents.append(genai_types.Part.from_text(
                        text=prompts.IMAGE_NO_REF.format(
                            visual=visual, brand_inst=brand_inst, prompt=prompt,
                        )
                    ))

                response = client.models.generate_content(
                    model=cfg.IMAGE_MODEL,
                    contents=contents,
                    config=genai.types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )
                image_bytes = None
                for part in response.candidates[0].content.parts:
                    if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                        image_bytes = part.inline_data.data
                        break

                if not image_bytes:
                    print("skip", end="", flush=True)
                    continue

                fpath.write_bytes(image_bytes)
                paths.append(str(fpath))
                print("ok", end="", flush=True)
            except Exception as e:
                print(f"fail", end="", flush=True)

        results[bid] = {"folder": str(folder), "images": paths}

    return results

