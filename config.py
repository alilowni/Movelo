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
MARKETER_TEMPERATURE = _env_float("MARKETER_TEMPERATURE", 0.8)

# -- Scoring --
HARD_SELL_THRESHOLD = _env_int("HARD_SELL_THRESHOLD", 3)
SCORE_DECAY_PER_DAY = _env_float("SCORE_DECAY_PER_DAY", 0.15)
AUTO_SELL_PROBABILITY = _env_float("AUTO_SELL_PROBABILITY", 0.25)

# -- Scheduler --
LOOP_INTERVAL_SEC = _env_int("LOOP_INTERVAL_SEC", 300)

# -- Paths --
CSV_PATH = _env("CSV_PATH", "sample_data_movelo_links.csv")
TRIALS_LOG_PATH = _env("TRIALS_LOG_PATH", "trials_log.csv")
OUTPUT_DIR = _env("OUTPUT_DIR", "output")
