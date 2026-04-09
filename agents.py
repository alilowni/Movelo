import json
import re
import sys
from pathlib import Path

import pandas as pd
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

from scoring import get_trial_columns

OUTPUT_DIR = Path("output")


def _make_llm(temperature: float = 0.7) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=temperature)


def _bike_summary(row: pd.Series) -> str:
    """Context string for a single bike."""
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
    ]
    desc = row.get("description", "")
    if pd.notna(desc) and desc:
        parts.append(f"Description: {desc}")

    trial_cols = get_trial_columns(row.to_frame().T)
    for col in trial_cols:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(f"Previous {col}: {val}")

    return "\n".join(parts)


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40]


def _parse_json_response(raw: str, step_name: str) -> list[dict]:
    """Strip markdown fences and parse JSON. Exit with clear message on failure."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n  ERROR: {step_name} returned invalid JSON.")
        print(f"  Parse error: {e}")
        print(f"  Raw response (first 500 chars):\n{text[:500]}")
        sys.exit(1)

    if not isinstance(data, list):
        print(f"\n  ERROR: {step_name} returned JSON but not an array.")
        sys.exit(1)

    return data


def trial_output_dir(trial_num: int) -> Path:
    """Return output/trial_N/ path, creating it if needed."""
    d = OUTPUT_DIR / f"trial_{trial_num}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Marketing Manager Agent
# ---------------------------------------------------------------------------

MANAGER_SYSTEM_PROMPT = """\
You are a marketing manager for a refurbished bike online shop called movelo.
Your job: look at bikes that are hard to sell and write a short marketing brief \
for each one.

IMPORTANT: Some bikes may have "Previous campaign_trial_N" fields. These are \
past marketing attempts we already tried. READ them carefully and come up with \
a DIFFERENT strategy this time. Do not repeat what was already done.

For EACH bike, return:
- bike_id
- target_audience (who would buy this)
- selling_angle (the main hook / value proposition)
- content_types (list: pick from "instagram_post", "email", "image_ad")
- tone (e.g. "adventurous", "value-focused", "urban-chic")
- key_message (one sentence the marketer should build around)

Reply ONLY with a JSON array. No extra text.
"""


def run_manager_agent(bikes_df: pd.DataFrame) -> list[dict]:
    """Send hard-to-sell bikes to the Marketing Manager and get strategy briefs."""
    llm = _make_llm(temperature=0.4)

    bike_texts = "\n---\n".join(
        _bike_summary(row) for _, row in bikes_df.iterrows()
    )

    try:
        response = llm.invoke([
            SystemMessage(content=MANAGER_SYSTEM_PROMPT),
            HumanMessage(content=f"Here are the bikes that need marketing help:\n\n{bike_texts}"),
        ])
    except Exception as e:
        print(f"\n  ERROR: Marketing Manager API call failed: {e}")
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
    """Take manager briefs + original bike data, produce marketing content."""
    llm = _make_llm(temperature=0.8)

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
        print(f"\n  ERROR: Marketer API call failed: {e}")
        sys.exit(1)

    return _parse_json_response(response.content, "Marketer")


# ---------------------------------------------------------------------------
# Image Generation (Gemini Flash native image generation)
# ---------------------------------------------------------------------------

def generate_images(content_list: list[dict], bikes_df: pd.DataFrame,
                    trial_num: int) -> dict:
    """Generate 2 images per bike and save to output/trial_N/<bike_folder>/."""
    try:
        client = genai.Client()
    except Exception as e:
        print(f"\n  ERROR: Could not create Gemini client: {e}")
        sys.exit(1)

    title_lookup = {
        int(row["id"]): row["title"] for _, row in bikes_df.iterrows()
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

        paths = []
        for label, key in [("urban", "image_prompt_a"), ("nature", "image_prompt_b")]:
            prompt = item.get(key, "")
            if not prompt:
                print(f"  WARNING: No {key} for bike {bid}, skipping {label} image.")
                continue

            fpath = folder / f"{label}.png"
            print(f"  Generating {label} image for bike {bid} ...", end=" ", flush=True)
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash-image",
                    contents=f"Generate a single high-quality photo. {prompt}",
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
                    print("FAILED (no image in response)")
                    continue

                fpath.write_bytes(image_bytes)
                paths.append(str(fpath))
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")

        results[bid] = {"folder": str(folder), "images": paths}

    return results
