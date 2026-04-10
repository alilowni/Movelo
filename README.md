# Movelo

AI-powered marketing pipeline for a refurbished bike shop. Built during the **PON AI Hackathon** for Movelo.

The system scores bikes by how hard they are to sell, then automatically generates targeted marketing campaigns (strategy briefs, Instagram copy, emails, lifestyle photos) for the toughest ones. Every campaign is logged so the AI never repeats a failed approach.

---

## What You Need

- Python 3.11+
- A **Google Gemini API key** (get one at [aistudio.google.com](https://aistudio.google.com))
- The files in this repo

**Key files:**

| File | What it does |
|---|---|
| `main.py` | Entry point — run campaigns, test the API, or launch the dashboard |
| `pipeline.py` | Controls the order and steps of the pipeline |
| `scoring.py` | Scores bikes and handles sales simulation |
| `agents.py` | LLM agents (manager, marketer, image generation) |
| `prompts.py` | All AI prompts in one place — easy to tune |
| `dashboard.py` | Streamlit dashboard (5 pages) |
| `sample_data_movelo_links.csv` | Seed data — 49 refurbished bikes (never modified at runtime) |
| `.env` | Your API key and settings (you create this from `.env.example`) |

---

## How to Run

**1. Clone the repo and set up a virtual environment:**

```bash
git clone <repo-url>
cd Movelo
python3.11 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Add your API key:**

```bash
cp .env.example .env
```

Open `.env` and replace `your-gemini-api-key-here` with your actual Gemini API key. That's the only required change.

**3. Verify everything works:**

```bash
python main.py --test-api
```

**4. Run a campaign:**

```bash
python main.py
```

Each run simulates one day. The system scores all bikes, picks the hardest to sell, and generates marketing content for them.

**5. Open the dashboard:**

```bash
python main.py --dashboard
# or
streamlit run dashboard.py
```

---

## How the Pipeline Works

Each run goes through 8 steps:

1. **Advance Day** — every available bike gains +1 day on market
2. **Score** — each bike gets a difficulty score from 0 to 5 (higher = harder to sell)
3. **Filter** — picks the top N bikes with the highest scores
4. **Marketing Manager Agent** — reads bike data and past campaign history, writes a strategy brief
5. **Marketer Agent** — reads the brief and generates Instagram captions, email copy, and image prompts
6. **Image Generation** — creates 2 lifestyle photos per bike (urban + nature) using the product photo as a reference
7. **Log Campaign** — saves everything (strategy, copy, image paths) to `campaigns.csv`
8. **Auto-sell** — simulates sales probabilistically; easier bikes sell more often, and insights are saved to `knowledge_base.csv`

---

## Scoring

Each bike gets a score from **0 to 5**. Higher score = harder to sell = gets prioritized for campaigns.

The score is based on four factors:

- **Price** (30%) — more expensive relative to inventory = harder to sell
- **Mileage** (30%) — more kilometers = harder. Unknown mileage defaults to mid-range.
- **Condition** (20%) — "Good" scores higher than "Excellent" (counterintuitive but intentional — good condition bikes may be priced similarly to excellent ones)
- **Age** (20%) — older bikes (2021 and before) score higher

There is also a **day penalty**: +0.30 per day the bike has been sitting on the market. This means bikes that don't sell keep getting more attention over time.

The score is a placeholder formula for the PoC. The plan is to replace it with real business logic, including a human-provided "popularity" score from the sales team. That hook is already stubbed in `scoring.py` with weight `0.0`.

---

## Data Files

| File | What gets stored |
|---|---|
| `sample_data_movelo_links.csv` | Original 49 bikes — never modified, used as seed |
| `inventory.csv` | Working copy — tracks status (available/sold) and days on market |
| `campaigns.csv` | One row per bike per campaign — strategy, copy, image paths |
| `knowledge_base.csv` | AI-written notes explaining why a bike likely sold — informs future campaigns |

On first run, `inventory.csv` is automatically created from the seed file.

---

## Dashboard

Run with `python main.py --dashboard` or `streamlit run dashboard.py`.

Five pages:

- **Bike Inventory** — all bikes with scores, filters by brand/category/score, detail view with product photo
- **Campaigns** — browse by campaign number or follow a single bike's full marketing journey
- **Run Campaign** — one button that runs the full pipeline and streams live logs
- **Analytics** — trend charts, risky bikes list, brand frequency
- **Knowledge Base** — sale insights with filters, showing what worked and why

---

## Configuration

All settings live in `.env`. The only required one is `GOOGLE_API_KEY`.

| Setting | Default | What it does |
|---|---|---|
| `GOOGLE_API_KEY` | — | Your Gemini API key (required) |
| `LLM_MODEL` | `gemini-2.0-flash` | Model for manager and marketer agents |
| `IMAGE_MODEL` | `gemini-2.5-flash-image` | Model for image generation |
| `HARD_SELL_THRESHOLD` | `3` | Minimum score for a bike to be targeted |
| `AUTO_SELL_PROBABILITY` | `0.40` | Base chance a bike sells after a campaign |
| `MAX_BIKES_PER_CAMPAIGN` | `10` | Max bikes per run (caps API calls) |
| `MANAGER_TEMPERATURE` | `0.4` | Manager agent creativity (0 = strict, 1 = creative) |
| `MARKETER_TEMPERATURE` | `0.7` | Marketer agent creativity |

---

## Questions?

Reach out to **ali.lowni@gmail.com**
