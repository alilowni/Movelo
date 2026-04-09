"""
Central configuration -- all tunables in one place, driven by .env.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    return int(_env(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(_env(key, str(default)))


# -- API --
GOOGLE_API_KEY = _env("GOOGLE_API_KEY", "")

# -- Models --
LLM_MODEL = _env("LLM_MODEL", "gemini-2.0-flash")
IMAGE_MODEL = _env("IMAGE_MODEL", "gemini-2.5-flash-image")

# -- Temperatures --
MANAGER_TEMPERATURE = _env_float("MANAGER_TEMPERATURE", 0.4)
MARKETER_TEMPERATURE = _env_float("MARKETER_TEMPERATURE", 0.7)

# -- Scoring --
HARD_SELL_THRESHOLD = _env_int("HARD_SELL_THRESHOLD", 3)
AUTO_SELL_PROBABILITY = _env_float("AUTO_SELL_PROBABILITY", 0.40)
MAX_BIKES_PER_CAMPAIGN = _env_int("MAX_BIKES_PER_CAMPAIGN", 10)

# -- Paths --
CSV_PATH = _env("CSV_PATH", "sample_data_movelo_links.csv")
TRIALS_LOG_PATH = _env("TRIALS_LOG_PATH", "trials_log.csv")
KNOWLEDGE_BASE_PATH = _env("KNOWLEDGE_BASE_PATH", "knowledge_base.csv")
OUTPUT_DIR = _env("OUTPUT_DIR", "output")
