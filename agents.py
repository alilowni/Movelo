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
    """Quick LLM ping -- returns True if the API responds."""
    try:
        llm = _make_llm(temperature=0.0)
        resp = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        return "OK" in resp.content.upper()
    except Exception as e:
        print(f"  LLM test failed: {e}")
        return False


def test_image_api() -> bool:
    """Quick image-gen ping -- returns True if the API responds."""
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
# Helpers
# ---------------------------------------------------------------------------

def _bike_summary(row: pd.Series) -> str:
    """Context string for a single bike, with full trial history from trials_log.csv."""
    parts = [
        f"ID: {row['id']}",
        f"Title: {row['title']}",
        f"Brand: {row['brand']}",
        f"Category: {row['category']}",
        f"Price: EUR {row['price']:.0f}",
        f"Condition: {row['condition']}",
        f"KM ridden: {row['km_ridden'] if pd.notna(row['km_ridden']) else 'N/A'}",
        f"Year: {int(row['year']) if pd.notna(row['year']) else 'N/A'}",
        f"Frame size: {row['frame_size']}",
        f"Sell difficulty score: {row['sell_difficulty_score']}",
        f"Days on market: {row.get('days_on_market', 0)}",
    ]
    desc = row.get("description", "")
    if pd.notna(desc) and desc:
        parts.append(f"Description: {desc}")

    history = load_trial_history_for_bike(int(row["id"]))
    if history:
        parts.append(f"\n=== PAST CAMPAIGNS ({len(history)} total) — DO NOT REPEAT THESE ===")
        for rec in history:
            trial_n = rec.get("trial_num", "?")
            parts.append(f"  Campaign #{trial_n} ({rec.get('date', '?')}):")
            parts.append(f"    Selling angle: {rec.get('selling_angle', 'N/A')}")
            parts.append(f"    Target audience: {rec.get('target_audience', 'N/A')}")
            parts.append(f"    Tone: {rec.get('tone', 'N/A')}")
            parts.append(f"    Actions: {rec.get('actions', 'N/A')}")
            subj = rec.get("email_subject", "")
            if subj:
                parts.append(f"    Email subject: {subj}")
            caption = rec.get("instagram_caption", "")
            if caption:
                short = caption[:120] + "..." if len(str(caption)) > 120 else caption
                parts.append(f"    Instagram: {short}")
        parts.append("=== END PAST CAMPAIGNS ===")
    else:
        parts.append("No previous campaigns for this bike.")

    return "\n".join(parts)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]


def _parse_json_response(raw: str, step_name: str) -> list[dict]:
    """Strip markdown fences and parse JSON."""
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
# Image download helper
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
# Marketing Manager Agent
# ---------------------------------------------------------------------------

MANAGER_SYSTEM_PROMPT = """\
You are a marketing manager for a refurbished bike online shop called movelo.
Your job: look at bikes that are hard to sell and write a short marketing brief \
for each one.

CRITICAL RULES:
1. Each bike's data includes a "PAST CAMPAIGNS" section showing every strategy \
we already tried. READ IT CAREFULLY.
2. You MUST choose a DIFFERENT selling angle, target audience, and tone each time.  \
If the past campaign used "value-focused" tone, try "adventurous" or "premium".  \
If it targeted "budget-conscious riders", target "weekend explorers" or "commuters".
3. Briefly explain in "why_different" how your new strategy differs from past ones.
4. If a bike has been marketed 3+ times and still not sold, consider a completely \
different approach: different audience, different angle, different content types.

For EACH bike, return:
- bike_id
- target_audience (who would buy this -- must differ from past campaigns)
- selling_angle (the main hook / value proposition -- must differ from past)
- content_types (list: pick from "instagram_post", "email", "image_ad")
- tone (e.g. "adventurous", "value-focused", "urban-chic" -- must differ from past)
- key_message (one sentence the marketer should build around)
- why_different (1 sentence: how this differs from the last campaign)

Reply ONLY with a JSON array. No extra text.
"""


def run_manager_agent(bikes_df: pd.DataFrame) -> list[dict]:
    llm = _make_llm(temperature=cfg.MANAGER_TEMPERATURE)

    bike_texts = "\n---\n".join(
        _bike_summary(row) for _, row in bikes_df.iterrows()
    )

    try:
        response = llm.invoke([
            SystemMessage(content=MANAGER_SYSTEM_PROMPT),
            HumanMessage(content=f"Here are the bikes that need marketing help:\n\n{bike_texts}"),
        ])
    except Exception as e:
        print(f"  ERROR: Marketing Manager call failed: {e}")
        sys.exit(1)

    return _parse_json_response(response.content, "Marketing Manager")


# ---------------------------------------------------------------------------
# Marketer Agent
# ---------------------------------------------------------------------------

MARKETER_SYSTEM_PROMPT = """\
You are a creative marketer for movelo, a refurbished bike shop.
You receive a marketing brief from your manager and you execute it.

For EACH brief, produce:
- bike_id
- instagram_caption (engaging, with hashtags, max 200 words)
- email_subject (short, punchy)
- email_body (friendly, 3-4 sentences)
- image_prompt_a (lifestyle photo prompt: a ~30 year old rider in an URBAN setting — \
city street, cafe, park path. Describe the scene, lighting, mood, what the rider wears. \
Be specific about the bike type and color.)
- image_prompt_b (lifestyle photo prompt: same ~30 year old demographic but in a \
NATURE / ADVENTURE setting — forest trail, countryside road, mountain overlook. \
Different vibe from prompt A. Describe scene, lighting, mood, rider outfit.)

IMPORTANT: Both image prompts must mention it is a photo advertisement for a \
refurbished bike shop. Do NOT include any text or logos in the image description. \
Keep prompts under 100 words each.

Reply ONLY with a JSON array. No extra text.
"""


def run_marketer_agent(briefs: list[dict], bikes_df: pd.DataFrame) -> list[dict]:
    llm = _make_llm(temperature=cfg.MARKETER_TEMPERATURE)

    bike_lookup = {
        int(row["id"]): _bike_summary(row) for _, row in bikes_df.iterrows()
    }

    context_parts = []
    for brief in briefs:
        bid = brief.get("bike_id")
        bike_info = bike_lookup.get(bid, "No extra info available.")
        context_parts.append(
            f"== Brief ==\n{json.dumps(brief, indent=2)}\n\n== Bike Data ==\n{bike_info}"
        )

    payload = "\n\n---\n\n".join(context_parts)

    try:
        response = llm.invoke([
            SystemMessage(content=MARKETER_SYSTEM_PROMPT),
            HumanMessage(content=f"Execute these briefs:\n\n{payload}"),
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

    title_lookup = {
        int(row["id"]): row["title"] for _, row in bikes_df.iterrows()
    }
    image_url_lookup = {
        int(row["id"]): row.get("image_url", "") for _, row in bikes_df.iterrows()
    }

    base_dir = trial_output_dir(trial_num)
    results = {}

    for item in content_list:
        bid = item.get("bike_id")
        title = title_lookup.get(bid, f"bike_{bid}")
        folder_name = f"bike_{bid}_{_slug(title)}"
        folder = base_dir / folder_name
        folder.mkdir(parents=True, exist_ok=True)

        brief_path = folder / "content.json"
        brief_path.write_text(json.dumps(item, indent=2))

        bike_image_url = image_url_lookup.get(bid, "")
        bike_image_bytes = _download_image(bike_image_url)
        has_ref = bike_image_bytes is not None

        paths = []
        for label, key in [("urban", "image_prompt_a"), ("nature", "image_prompt_b")]:
            prompt = item.get(key, "")
            if not prompt:
                continue

            fpath = folder / f"{label}.png"
            print(f"  bike {bid} / {label} ...", end=" ", flush=True)
            try:
                contents = []
                if has_ref:
                    contents.append(genai_types.Part.from_bytes(
                        data=bike_image_bytes,
                        mime_type=_mime_from_url(bike_image_url),
                    ))
                    contents.append(genai_types.Part.from_text(
                        text=f"Generate a single high-quality lifestyle photo ad "
                             f"featuring THIS exact bike shown in the reference image. {prompt}"
                    ))
                else:
                    contents.append(genai_types.Part.from_text(
                        text=f"Generate a single high-quality photo. {prompt}"
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
