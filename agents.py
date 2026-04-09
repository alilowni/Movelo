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
from scoring import load_trial_history_for_bike

OUTPUT_DIR = Path(cfg.OUTPUT_DIR)
IMAGE_CACHE_DIR = OUTPUT_DIR / ".image_cache"


def _make_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=cfg.LLM_MODEL, temperature=temperature)


# ---------------------------------------------------------------------------
# API health checks
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Bike context builder (compact for token efficiency)
# ---------------------------------------------------------------------------

def _bike_context(row: pd.Series) -> str:
    """Compact bike context string. Includes only relevant fields + short history."""
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

    history = load_trial_history_for_bike(int(row["id"]))
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
    """Extracted visual details for image generation prompts."""
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


def _parse_json_response(raw: str, step_name: str) -> list[dict]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  ERROR: {step_name} returned invalid JSON: {e}")
        sys.exit(1)
    if not isinstance(data, list):
        print(f"  ERROR: {step_name} returned JSON but not an array.")
        sys.exit(1)
    return data


def trial_output_dir(trial_num: int) -> Path:
    d = OUTPUT_DIR / f"trial_{trial_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Image download
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Marketing Manager
# ---------------------------------------------------------------------------

MANAGER_SYSTEM_PROMPT = """\
You are a marketing manager for movelo, a refurbished bike shop.
Analyze hard-to-sell bikes and create a brief for each.

RULES:
- Be concise. No filler.
- If a bike has PAST campaigns listed, you MUST pick a different angle, \
audience, and tone. Never repeat what was tried.
- After 3+ failed campaigns, try a radically different approach.

For EACH bike return JSON:
- bike_id (int)
- target_audience (short, specific)
- selling_angle (1 sentence max)
- content_types (list: "instagram_post", "email", "image_ad")
- tone (1 word)
- key_message (1 sentence)
- why_different (1 sentence if past campaigns exist, else "first campaign")

Reply ONLY with a JSON array.
"""


def run_manager_agent(bikes_df: pd.DataFrame) -> list[dict]:
    llm = _make_llm(temperature=cfg.MANAGER_TEMPERATURE)
    bike_texts = "\n".join(_bike_context(row) for _, row in bikes_df.iterrows())
    try:
        response = llm.invoke([
            SystemMessage(content=MANAGER_SYSTEM_PROMPT),
            HumanMessage(content=bike_texts),
        ])
    except Exception as e:
        print(f"  ERROR: Manager call failed: {e}")
        sys.exit(1)
    return _parse_json_response(response.content, "Manager")


# ---------------------------------------------------------------------------
# Marketer
# ---------------------------------------------------------------------------

MARKETER_SYSTEM_PROMPT = """\
You are a marketer for movelo, a refurbished bike shop based in the Netherlands.
You receive briefs and produce content. Be concise.

For EACH brief produce:
- bike_id (int)
- instagram_caption: Casual, slightly playful tone. Short (max 120 words). \
Include 5-8 relevant hashtags at the end. Don't overdo emojis (max 2). \
Focus on the lifestyle benefit, not specs.
- email_subject: Short, direct, max 8 words. Dutch communication style.
- email_body: Clean, professional, direct. 2-3 sentences max. \
No hype, no exclamation marks. State the value clearly. \
Think Dutch directness: what it is, why it matters, what to do next.
- image_prompt_a: PRIMARY lifestyle photo. Use the exact bike details \
(brand, model, color, type) from the data. IMPORTANT: Match the bike category — \
if the bike is a City/E-City/Urban bike, use a city/urban setting (Dutch streets, \
canal bridge, bike lane). If it is a Trekking/E-Trekking/MTB/Gravel bike, use a \
nature/trail setting. Describe: rider (~30yo), setting, lighting, mood, outfit. \
The brand name must be visible on the bike frame. \
Include a small "movelo" watermark/logo in the bottom-right corner. \
Keep under 80 words. Must say "photo advertisement for movelo refurbished bike shop".
- image_prompt_b: SECONDARY lifestyle photo. Same bike details. \
Different setting from prompt A. If prompt A was urban, make this one a park/nature \
scene. If prompt A was nature, make this one urban. Same branding rules: \
brand name visible on bike, small "movelo" logo bottom-right corner. \
Under 80 words. Must say "photo advertisement for movelo refurbished bike shop".

Reply ONLY with a JSON array.
"""


def run_marketer_agent(briefs: list[dict], bikes_df: pd.DataFrame) -> list[dict]:
    llm = _make_llm(temperature=cfg.MARKETER_TEMPERATURE)

    bike_visual = {
        int(row["id"]): _bike_image_context(row) for _, row in bikes_df.iterrows()
    }

    context_parts = []
    for brief in briefs:
        bid = brief.get("bike_id")
        visual = bike_visual.get(bid, "no details")
        context_parts.append(
            f"Brief: {json.dumps(brief)}\nBike visual: {visual}"
        )

    payload = "\n---\n".join(context_parts)
    try:
        response = llm.invoke([
            SystemMessage(content=MARKETER_SYSTEM_PROMPT),
            HumanMessage(content=payload),
        ])
    except Exception as e:
        print(f"  ERROR: Marketer call failed: {e}")
        sys.exit(1)
    return _parse_json_response(response.content, "Marketer")


# ---------------------------------------------------------------------------
# Image Generation
# ---------------------------------------------------------------------------

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
            print(f"  bike {bid} / {label} ...", end=" ", flush=True)
            try:
                contents = []
                brand_inst = (
                    f"The brand name '{visual.split(',')[0].split()[0]}' "
                    f"must be visible on the bike frame. "
                    f"Include a small 'movelo' watermark in the bottom-right corner."
                )
                if has_ref:
                    contents.append(genai_types.Part.from_bytes(
                        data=bike_image_bytes,
                        mime_type=_mime_from_url(bike_image_url),
                    ))
                    contents.append(genai_types.Part.from_text(
                        text=(
                            f"Generate a realistic lifestyle photo ad featuring "
                            f"THIS exact bike from the reference image. "
                            f"Bike: {visual}. {brand_inst} {prompt}"
                        )
                    ))
                else:
                    contents.append(genai_types.Part.from_text(
                        text=(
                            f"Generate a realistic lifestyle photo. "
                            f"Bike: {visual}. {brand_inst} {prompt}"
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
                    print("no image in response")
                    continue

                fpath.write_bytes(image_bytes)
                paths.append(str(fpath))
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")

        results[bid] = {"folder": str(folder), "images": paths}

    return results
