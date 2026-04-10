# central config — all tunables loaded from .env

import os
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


# api
GOOGLE_API_KEY = _env("GOOGLE_API_KEY", "")

# models
LLM_MODEL = _env("LLM_MODEL", "gemini-2.0-flash")
IMAGE_MODEL = _env("IMAGE_MODEL", "gemini-2.5-flash-image")

# agent creativity — higher = more varied output
MANAGER_TEMPERATURE = _env_float("MANAGER_TEMPERATURE", 0.4)
MARKETER_TEMPERATURE = _env_float("MARKETER_TEMPERATURE", 0.7)

# scoring and sales simulation
HARD_SELL_THRESHOLD = _env_int("HARD_SELL_THRESHOLD", 3)          # bikes >= this get targeted
AUTO_SELL_PROBABILITY = _env_float("AUTO_SELL_PROBABILITY", 0.40) # base chance of sale per campaign
MAX_BIKES_PER_CAMPAIGN = _env_int("MAX_BIKES_PER_CAMPAIGN", 10)   # caps api calls per run

# paths — auto-created at runtime, no need to change
SEED_CSV_PATH = _env("SEED_CSV_PATH", "sample_data_movelo_links.csv")
CSV_PATH = _env("CSV_PATH", "inventory.csv")
CAMPAIGNS_PATH = _env("CAMPAIGNS_PATH", "campaigns.csv")
KNOWLEDGE_BASE_PATH = _env("KNOWLEDGE_BASE_PATH", "knowledge_base.csv")
OUTPUT_DIR = _env("OUTPUT_DIR", "output")
IMAGE_EVAL_PATH = _env("IMAGE_EVAL_PATH", "image_evaluations.xlsx")
