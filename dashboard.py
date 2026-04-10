# movelo marketing dashboard
# run: streamlit run dashboard.py  or  python main.py --dashboard

import re
import subprocess
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

import config as cfg
import scoring
from scoring import (
    init_working_files,
    load_and_score,
    load_campaigns,
    join_bikes_and_campaigns,
    load_knowledge_base,
    load_image_evaluations,
)

init_working_files()

st.set_page_config(page_title="Movelo Dashboard", page_icon="🚲", layout="wide")

if "campaign_running" not in st.session_state:
    st.session_state.campaign_running = False


# cached data loaders

@st.cache_data(ttl=10)
def get_bikes() -> pd.DataFrame:
    return load_and_score()


@st.cache_data(ttl=10)
def get_campaigns() -> pd.DataFrame:
    return load_campaigns()


@st.cache_data(ttl=10)
def get_joined() -> pd.DataFrame:
    return join_bikes_and_campaigns()


@st.cache_data(ttl=10)
def get_knowledge_base() -> pd.DataFrame:
    return load_knowledge_base()


@st.cache_data(ttl=60)
def get_image_evals() -> dict:
    return load_image_evaluations()


# display helpers

def risk_color(score: int) -> str:
    if score >= 4:
        return "🔴"
    if score == 3:
        return "🟡"
    return "🟢"


def status_badge(status: str) -> str:
    return "🏷 SOLD" if status == "sold" else "✅ Available"


# pipeline log parser — turns raw main.py output into clean summary

def _parse_pipeline_log(raw: str) -> tuple[str, str]:
    lines = []
    campaign_num = "?"
    sold_text = "no sales"
    total_seconds = 0.0
    current_step = None

    for line in raw.splitlines():
        s = line.strip()
        if not s or s.startswith("Done"):
            continue

        is_step = bool(re.match(r"^\[\d+/\d+\]", s))
        tm = re.search(r"\((\d+\.?\d*)s\)\s*$", s)

        if tm:
            total_seconds += float(tm.group(1))

        cn = re.search(r"#(\d+)", s)
        if cn:
            campaign_num = cn.group(1)

        sm = re.search(r"sold (\d+):", s)
        if sm:
            sold_text = f"{sm.group(1)} sold"

        if s.startswith("API check:"):
            lines.append(f"  ✓  {s}")
        elif is_step and tm:
            clean = re.sub(r"^\[\d+/\d+\]\s*", "", s)
            lines.append(f"  ✓  {clean}")
            current_step = None
        elif is_step and not tm:
            m = re.match(r"^\[\d+/\d+\]\s*(.+)", s)
            current_step = m.group(1).strip() if m else None
        elif not is_step and tm:
            detail = re.sub(r"\s*\(\d+\.?\d*s\)\s*$", "", s).strip()
            t = tm.group(0)
            prefix = f"{current_step}  " if current_step else ""
            lines.append(f"  ✓  {prefix}{detail} {t}")
            current_step = None

    log = "\n".join(lines)
    label = f"Campaign {campaign_num} · {sold_text} · {total_seconds:.0f}s"
    return log, label


# campaign runner — executes main.py as subprocess, locks ui during run

def _run_campaign():
    if st.session_state.campaign_running:
        return

    bikes_df = get_bikes()
    n_avail = len(bikes_df[bikes_df["status"] != "sold"])
    if n_avail == 0:
        st.toast("All bikes are already sold.", icon="✅")
        return

    st.session_state.campaign_running = True

    with st.status("Running campaign — please wait...", expanded=True) as status:
        try:
            result = subprocess.run(
                [sys.executable, "main.py"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).parent), timeout=600,
            )
            raw = result.stdout or ""
            if result.stderr:
                raw += "\n" + result.stderr

            if result.returncode == 0:
                log, label = _parse_pipeline_log(raw)
                st.code(log, language=None)
                with st.expander("Raw output"):
                    st.code(raw, language=None)
                status.update(label=label, state="complete", expanded=False)
                st.toast("Campaign complete!", icon="✅")
            else:
                st.code(raw, language=None)
                status.update(label="Campaign failed", state="error")
        except Exception as e:
            st.error(f"Campaign error: {e}")
            status.update(label="Campaign failed", state="error")

    st.session_state.campaign_running = False
    st.cache_data.clear()
    st.rerun()


# campaign card — reusable component for both campaign and bike journey views

def _render_campaign_card(row: pd.Series, bikes_df: pd.DataFrame,
                          evals: dict | None = None) -> None:
    bid = int(row["bike_id"])
    bike_info = bikes_df[bikes_df["id"] == bid]
    title = bike_info["title"].values[0] if not bike_info.empty else f"Bike {bid}"
    img_url = bike_info["image_url"].values[0] if not bike_info.empty else ""
    status = bike_info["status"].values[0] if not bike_info.empty else "available"

    trial_n = int(row.get("trial_num", 0))
    sold_here = str(row.get("sold_in_campaign", "")).strip().lower() == "yes"
    header = f"Campaign {trial_n} — Bike {bid}: {title}"
    if sold_here:
        header += " 🎉 SOLD THIS CAMPAIGN"
    elif status == "sold":
        header += " 🏷 SOLD"

    bike_eval = (evals or {}).get(bid, {})
    eval_images = bike_eval.get("images", [])

    with st.expander(header, expanded=True):
        st.markdown(f"**Selling angle:** {row.get('selling_angle', 'N/A')}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Audience:** {row.get('target_audience', 'N/A')}")
        with c2:
            st.markdown(f"**Tone:** {row.get('tone', 'N/A')}")
        with c3:
            st.markdown(f"**Actions:** {row.get('actions', 'N/A')}")

        subj = row.get("email_subject", "")
        body = row.get("email_body", "")
        if pd.notna(subj) and subj:
            st.markdown("**Email**")
            email_text = f"Subject: {subj}"
            if pd.notna(body) and body:
                email_text += f"\n\n{body}"
            st.code(email_text, language=None)

        caption = row.get("instagram_caption", "")
        if pd.notna(caption) and caption:
            st.markdown("**Instagram caption**")
            st.code(str(caption), language=None)

        # ai-generated images (product photo + urban/nature ads)
        urban_path = row.get("urban_image_path", "")
        nature_path = row.get("nature_image_path", "")
        has_urban = pd.notna(urban_path) and urban_path and Path(urban_path).exists()
        has_nature = pd.notna(nature_path) and nature_path and Path(nature_path).exists()
        has_product = pd.notna(img_url) and img_url

        if has_product or has_urban or has_nature:
            img_cols = st.columns(3)
            with img_cols[0]:
                if has_product:
                    st.image(img_url, caption="Product photo", width="stretch")
            with img_cols[1]:
                if has_urban:
                    st.image(str(urban_path), caption="Urban ad", width="stretch")
            with img_cols[2]:
                if has_nature:
                    st.image(str(nature_path), caption="Nature ad", width="stretch")

        # banner marketing assets from image evaluations
        if eval_images:
            st.markdown("**Marketing assets**")
            banner_cols = st.columns(3)
            for i, img in enumerate(eval_images[:3]):
                with banner_cols[i]:
                    try:
                        with open(img["path"], "rb") as f:
                            st.image(f.read(), caption=f"Banner {i + 1}", width="stretch")
                    except FileNotFoundError:
                        st.caption(f"Banner {i + 1} — image not found")
                    if img.get("html_url"):
                        st.markdown(
                            f'<a href="{img["html_url"]}" target="_blank" '
                            f'rel="noopener noreferrer">🌐 Landing page</a>',
                            unsafe_allow_html=True,
                        )


# sidebar

running = st.session_state.campaign_running

st.sidebar.title("Movelo")
st.sidebar.caption("Refurbished Bike Marketing Dashboard")

if running:
    st.sidebar.warning("⏳ Campaign running — please wait...")

page = st.sidebar.radio(
    "Navigate",
    ["🗄️ Bike Inventory", "📣 Campaigns", "📊 Analytics", "🧠 Knowledge Base"],
    captions=[
        "Your database — all available bikes and their scores",
        "Actions your AI agents run to automate selling",
        "Dashboards and monitoring for your campaigns",
        "Learned insights — gets smarter with every sale",
    ],
    disabled=running,
)

st.sidebar.markdown("---")
sb1, sb2 = st.sidebar.columns(2)
with sb1:
    refresh_clicked = st.button("🔄 Refresh", use_container_width=True, disabled=running)
with sb2:
    run_clicked = st.button("▶ Campaign", type="primary", use_container_width=True, disabled=running)

if refresh_clicked:
    st.cache_data.clear()
    st.rerun()

if run_clicked:
    _run_campaign()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Made with ❤️ at PON Hackathon  \n"
    "Karen Oskam · karen.oskam@pon.com  \n"
    "Ali Lowni alowni@vwpfs.com  \n"
    "John Broekhof · john.broekhof@pon.com  \n"
    "Aron Tjhin · aron.tjhin@pon.com  \n"
    "Guus Kroon · guus.kroon@pon.com  \n"
    "Zeger Knops · zeger.knops@pon.com"
    "  \n"
    "[GitHub](https://github.com/alilowni/Movelo)"
)


# page: bike inventory

if page == "🗄️ Bike Inventory":
    st.title("Bike Inventory")
    with st.expander("About this page"):
        st.markdown(
            "Your full bike catalog with live scoring. "
            "Toggle **Risky bikes** to see which ones need marketing attention."
        )
    bikes = get_bikes()

    n_sold = len(bikes[bikes["status"] == "sold"])
    n_avail = len(bikes) - n_sold
    n_risky = len(bikes[(bikes["sell_difficulty_score"] >= cfg.HARD_SELL_THRESHOLD) & (bikes["status"] != "sold")])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Available", n_avail)
    col2.metric("Sold", n_sold)
    col3.metric("Risky", n_risky)
    col4.metric("Avg price", f"€{bikes['price'].mean():.0f}")

    # filters
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        brands = st.multiselect("Brand", sorted(bikes["brand"].unique()))
    with fcol2:
        categories = st.multiselect("Category", sorted(bikes["category"].unique()))
    with fcol3:
        risky_only = st.toggle("🔴 Risky only", value=False,
                                help=f"Score ≥ {cfg.HARD_SELL_THRESHOLD}")
    with fcol4:
        show_sold = st.toggle("🏷️ Sold only", value=False)

    filtered = bikes.copy()
    if brands:
        filtered = filtered[filtered["brand"].isin(brands)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if risky_only:
        filtered = filtered[filtered["sell_difficulty_score"] >= cfg.HARD_SELL_THRESHOLD]
    if show_sold:
        filtered = filtered[filtered["status"] == "sold"]
    else:
        filtered = filtered[filtered["status"] != "sold"]

    if risky_only and not filtered.empty:
        st.caption(f"Showing {len(filtered)} risky bikes (score ≥ {cfg.HARD_SELL_THRESHOLD}) — these get targeted by campaigns.")

    display = filtered[
        ["id", "title", "brand", "category", "price", "condition",
         "sell_difficulty_score", "days_on_market", "status"]
    ].copy()
    display["risk"] = display["sell_difficulty_score"].apply(risk_color)
    display["status_badge"] = display["status"].apply(status_badge)
    display = display.sort_values("sell_difficulty_score", ascending=False)

    st.dataframe(
        display[["risk", "id", "title", "brand", "category", "price",
                 "condition", "sell_difficulty_score", "days_on_market", "status_badge"]],
        width="stretch",
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn("Price (€)", format="%.0f"),
            "sell_difficulty_score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=5, format="%d"
            ),
            "days_on_market": "Days listed",
            "status_badge": "Status",
            "risk": " ",
        },
    )

    # score explanation
    with st.expander("How is the score calculated?"):
        st.markdown(
            f"Each bike gets a **sell difficulty score** from 0 (easy) to 5 (hard). "
            f"Bikes scoring **≥ {cfg.HARD_SELL_THRESHOLD}** are flagged as \"risky\" "
            f"and get targeted by marketing campaigns.\n\n"
            f"**Formula:** `score = round(base + day_penalty)`, clamped 0–5\n\n"
            f"| Factor | Weight | 0 (easy) | 5 (hard) |\n"
            f"|---|---|---|---|\n"
            f"| Price | 30% | Cheapest in inventory | Most expensive |\n"
            f"| Mileage | 30% | 0 km | Highest km |\n"
            f"| Condition | 20% | Excellent | Good |\n"
            f"| Age | 20% | 2024 | 2021 or older |\n"
            f"| Day penalty | +{scoring.DAY_PENALTY_PER_DAY}/day | Day 0 | Accumulates |\n\n"
            f"All factors are relative to the current inventory. "
            f"A *popularity* weight will be added later based on human input from the sales team."
        )

    # bike detail view
    st.markdown("### Bike Details")
    bike_ids = filtered["id"].tolist()
    if bike_ids:
        selected_id = st.selectbox(
            "Select a bike", bike_ids,
            format_func=lambda x: f"[{x}] {bikes[bikes['id']==x]['title'].values[0]}"
        )
        bike_row = bikes[bikes["id"] == selected_id].iloc[0]

        dcol1, dcol2 = st.columns([1, 2])
        with dcol1:
            img_url = bike_row.get("image_url", "")
            if pd.notna(img_url) and img_url:
                st.image(img_url, caption=bike_row["title"], width="stretch")
            else:
                st.info("No product image")
        with dcol2:
            st.markdown(f"**{bike_row['title']}**")
            km = bike_row['km_ridden']
            year = int(bike_row['year']) if pd.notna(bike_row.get('year')) else 'N/A'
            info_lines = (
                f"| | |\n|---|---|\n"
                f"| **Brand** | {bike_row['brand']} |\n"
                f"| **Category** | {bike_row['category']} |\n"
                f"| **Price** | €{bike_row['price']:.0f} |\n"
                f"| **Condition** | {bike_row['condition']} |\n"
                f"| **KM** | {km if pd.notna(km) else 'N/A'} |\n"
                f"| **Year** | {year} |\n"
                f"| **Score** | {risk_color(bike_row['sell_difficulty_score'])} {bike_row['sell_difficulty_score']}/5 |\n"
                f"| **Days listed** | {bike_row['days_on_market']} |"
            )
            st.markdown(info_lines)

            desc = bike_row.get("description", "")
            if pd.notna(desc) and desc:
                st.caption(desc)

            product_url = bike_row.get("product_url", "")
            if pd.notna(product_url) and product_url:
                st.link_button("View on website", product_url)

            current_status = bike_row.get("status", "available")
            if current_status == "sold":
                st.success("This bike is marked as SOLD")

    st.info(
        "**Production roadmap** — In a real deployment this inventory connects to your "
        "POS or warehouse system via API (e.g. Shopify, Lightspeed). The scoring formula "
        "currently uses price, mileage, condition, and age with static weights; these would "
        "be replaced by a trained model once enough sales data is collected. A *popularity* "
        "signal (page views, saves, test-ride requests) can be added as an extra weight to "
        "reflect real demand."
    )


# page: campaigns

elif page == "📣 Campaigns":
    st.title("Marketing Campaigns")
    with st.expander("About this page"):
        st.markdown(
            "Every campaign your AI agents have run. "
            "Switch between a per-campaign view and a per-bike journey to track what was tried."
        )
    trials = get_campaigns()
    bikes_df = get_bikes()
    evals = get_image_evals()

    if trials.empty:
        st.info("No campaigns yet. Click **▶ Campaign** in the sidebar to start one.")
    else:
        bikes_with_assets = {bid for bid, e in evals.items() if e.get("images")}

        vcol1, vcol2 = st.columns([2, 1])
        with vcol1:
            view_mode = st.radio("View by", ["Campaign", "Bike journey"],
                                 horizontal=True, label_visibility="collapsed")
        with vcol2:
            assets_only = st.toggle("📎 Has marketing assets", value=False,
                                     help="Show only bikes that have banners + landing page")

        # campaign view — browse by campaign number
        if view_mode == "Campaign":
            trial_nums = sorted(trials["trial_num"].unique())
            tcol1, tcol2 = st.columns([1, 3])
            with tcol1:
                st.metric("Total campaigns", len(trial_nums))
            with tcol2:
                selected_trial = st.selectbox("Select campaign", trial_nums,
                                              format_func=lambda x: f"Campaign {int(x)}")

            trial_data = trials[trials["trial_num"] == selected_trial]
            if assets_only:
                trial_data = trial_data[trial_data["bike_id"].astype(int).isin(bikes_with_assets)]
            trial_date = trial_data["date"].dropna().values[0] if not trial_data["date"].dropna().empty else "N/A"
            n_with_assets = len(trial_data[trial_data["bike_id"].astype(int).isin(bikes_with_assets)])
            st.markdown(f"### Campaign {int(selected_trial)} — {trial_date} — {len(trial_data)} bikes ({n_with_assets} with assets)")

            for _, row in trial_data.iterrows():
                _render_campaign_card(row, bikes_df, evals)

        # bike journey view — browse by bike across all campaigns
        else:
            targeted_ids = sorted(trials["bike_id"].unique())
            if assets_only:
                targeted_ids = [bid for bid in targeted_ids if int(bid) in bikes_with_assets]
            selected_bike = st.selectbox(
                "Select bike to see full journey",
                targeted_ids,
                format_func=lambda x: (
                    f"[{int(x)}] "
                    f"{bikes_df[bikes_df['id']==int(x)]['title'].values[0] if not bikes_df[bikes_df['id']==int(x)].empty else x}"
                    f"{' 🏷 SOLD' if not bikes_df[bikes_df['id']==int(x)].empty and bikes_df[bikes_df['id']==int(x)]['status'].values[0]=='sold' else ''}"
                    f"{' 📎' if int(x) in bikes_with_assets else ''}"
                ),
            )
            bike_info = bikes_df[bikes_df["id"] == int(selected_bike)]
            if not bike_info.empty:
                b = bike_info.iloc[0]
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Score", f"{b['sell_difficulty_score']}/5")
                mc2.metric("Days listed", int(b["days_on_market"]))
                mc3.metric("Status", "SOLD" if b["status"] == "sold" else "Available")
                bike_campaigns = trials[trials["bike_id"] == int(selected_bike)]
                mc4.metric("Campaigns", len(bike_campaigns))

            bike_trials = trials[trials["bike_id"] == int(selected_bike)].sort_values("trial_num")
            if bike_trials.empty:
                st.info("No campaigns for this bike yet.")
            else:
                for _, row in bike_trials.iterrows():
                    _render_campaign_card(row, bikes_df, evals)

    st.info(
        "**Production roadmap** — Campaigns are currently triggered manually from the sidebar. "
        "In production, a scheduler (cron, Airflow, or a simple Cloud Function) would run campaigns "
        "on a cadence (e.g. daily). Each campaign's generated content (captions, emails, images) "
        "would be pushed directly to publishing APIs — Meta Business Suite for Instagram, "
        "an ESP like Mailchimp or Brevo for emails. A/B test results and engagement metrics "
        "would feed back into the scoring model to close the loop."
    )


# page: analytics

elif page == "📊 Analytics":
    st.title("Analytics")
    with st.expander("About this page"):
        st.markdown(
            "High-level trends across all campaigns. "
            "Track how many bikes are being targeted, sold over time, and which brands get the most attention."
        )
    bikes_df = get_bikes()
    trials = get_campaigns()
    joined = get_joined()

    n_avail = len(bikes_df[bikes_df["status"] != "sold"])
    n_sold = len(bikes_df[bikes_df["status"] == "sold"])
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Available", n_avail)
    col2.metric("Sold", n_sold)
    n_trials = int(trials["trial_num"].max()) if not trials.empty else 0
    col3.metric("Campaigns run", n_trials)
    targeted = trials["bike_id"].nunique() if not trials.empty else 0
    col4.metric("Bikes marketed", targeted)

    # risky bikes table
    st.markdown("### Risky bikes (available, score >= 4)")
    risky = bikes_df[(bikes_df["sell_difficulty_score"] >= 4) & (bikes_df["status"] != "sold")].copy()

    if not trials.empty:
        trial_counts = trials.groupby("bike_id").size().reset_index(name="times_marketed")
        risky = risky.merge(trial_counts, left_on="id", right_on="bike_id", how="left")
        risky["times_marketed"] = risky["times_marketed"].fillna(0).astype(int)
    else:
        risky["times_marketed"] = 0

    risky = risky.sort_values("sell_difficulty_score", ascending=False)
    st.dataframe(
        risky[["id", "title", "brand", "price", "condition",
               "sell_difficulty_score", "days_on_market", "times_marketed"]],
        width="stretch",
        hide_index=True,
        column_config={
            "price": st.column_config.NumberColumn("Price (€)", format="%.0f"),
            "sell_difficulty_score": st.column_config.ProgressColumn(
                "Score", min_value=0, max_value=5, format="%d"
            ),
            "days_on_market": "Days listed",
        },
    )

    never_marketed = risky[risky["times_marketed"] == 0]
    if not never_marketed.empty:
        st.warning(f"{len(never_marketed)} high-risk bikes have NEVER been marketed:")
        for _, r in never_marketed.iterrows():
            st.markdown(f"- **[{r['id']}] {r['title']}** — €{r['price']:.0f}, score {r['sell_difficulty_score']}, {r['days_on_market']} days")

    # trend charts
    if not trials.empty and "date" in trials.columns:
        st.markdown("### Trends")
        sold_flags = trials[trials["sold_in_campaign"] == "yes"].copy() if "sold_in_campaign" in trials.columns else pd.DataFrame()

        tcol1, tcol2 = st.columns(2)
        with tcol1:
            st.markdown("**Bikes targeted per campaign**")
            bikes_per_campaign = (
                trials.groupby("trial_num")["bike_id"]
                .nunique()
                .reset_index()
                .rename(columns={"bike_id": "bikes_targeted", "trial_num": "campaign"})
                .sort_values("campaign")
            )
            bikes_per_campaign = bikes_per_campaign.set_index("campaign")
            st.line_chart(bikes_per_campaign["bikes_targeted"])

        with tcol2:
            st.markdown("**Cumulative bikes sold**")
            if not sold_flags.empty:
                sold_per_campaign = (
                    sold_flags.groupby("trial_num")["bike_id"]
                    .nunique()
                    .reset_index()
                    .rename(columns={"bike_id": "sold", "trial_num": "campaign"})
                    .sort_values("campaign")
                )
                sold_per_campaign["cumulative_sold"] = sold_per_campaign["sold"].cumsum()
                sold_per_campaign = sold_per_campaign.set_index("campaign")
                st.line_chart(sold_per_campaign["cumulative_sold"])
            else:
                st.info("No sales recorded yet.")

        st.markdown("**Available vs Sold (current snapshot)**")
        snap = pd.DataFrame({
            "Status": ["Available", "Sold"],
            "Count": [n_avail, n_sold],
        }).set_index("Status")
        st.bar_chart(snap)

    # brand frequency
    st.markdown("### Marketing frequency by brand")
    if not trials.empty:
        brand_merge = trials.merge(bikes_df[["id", "brand"]], left_on="bike_id", right_on="id", how="left")
        brand_counts = brand_merge.groupby("brand").size().reset_index(name="campaigns")
        brand_counts = brand_counts.sort_values("campaigns", ascending=False)
        st.bar_chart(brand_counts.set_index("brand")["campaigns"])

    # full joined table
    st.markdown("### Full joined table")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        filter_brands = st.multiselect("Filter by brand", sorted(bikes_df["brand"].unique()), key="a_brand")
    with fcol2:
        filter_trial = st.multiselect(
            "Filter by campaign",
            sorted(trials["trial_num"].unique()) if not trials.empty else [],
            key="a_trial",
        )

    display_joined = joined.copy()
    if filter_brands:
        display_joined = display_joined[display_joined["brand"].isin(filter_brands)]
    if filter_trial:
        display_joined = display_joined[display_joined["trial_num"].isin(filter_trial)]

    show_cols = ["id", "title", "brand", "price", "sell_difficulty_score", "status", "days_on_market"]
    if "trial_num" in display_joined.columns:
        show_cols += ["trial_num", "date", "selling_angle", "actions"]
    st.dataframe(display_joined[show_cols], width="stretch", hide_index=True)

    st.info(
        "**Production roadmap** — These charts currently rely on internal simulation data. "
        "In production, analytics would ingest real engagement metrics (impressions, clicks, CTR) "
        "from ad platforms and email providers. Conversion tracking (test-ride bookings, purchases) "
        "would connect to the CRM, enabling ROI calculations per campaign and per bike. "
        "Dashboards could be migrated to a BI tool like Metabase or Looker for team-wide access."
    )


# page: knowledge base

elif page == "🧠 Knowledge Base":
    st.title("Sales Knowledge Base")
    with st.expander("About this page"):
        st.markdown(
            "Automatic insights on **why each bike sold**. "
            "The AI captures a short note per sale — the manager agent reads these to improve future campaigns."
        )

    kb = get_knowledge_base()

    if kb.empty:
        st.info(
            "No sales recorded yet. Click **▶ Campaign** in the sidebar — every "
            "time a bike sells, an insight will be added here."
        )
    else:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bikes sold", len(kb))
        m2.metric("Avg score sold", f"{kb['sell_difficulty_score'].mean():.1f}/5")
        m3.metric("Avg days listed", f"{kb['days_on_market'].mean():.0f}")
        m4.metric("Avg campaigns run", f"{kb['campaigns_run'].mean():.1f}")

        st.markdown("### Latest insights")
        kb_sorted = kb.sort_values("trial_num", ascending=False)

        # filters
        fc1, fc2 = st.columns(2)
        with fc1:
            brand_filter = st.multiselect(
                "Filter by brand", sorted(kb["brand"].dropna().unique()), key="kb_brand"
            )
        with fc2:
            tone_filter = st.multiselect(
                "Filter by tone", sorted(kb["tone"].dropna().unique()), key="kb_tone"
            )

        view = kb_sorted.copy()
        if brand_filter:
            view = view[view["brand"].isin(brand_filter)]
        if tone_filter:
            view = view[view["tone"].isin(tone_filter)]

        # insight cards
        for _, row in view.iterrows():
            r = row.to_dict()
            header = (
                f"Bike {int(r['bike_id'])} — {r['title']} "
                f"(€{float(r['price']):.0f}) · campaign "
                f"{int(r['trial_num'])}"
            )
            with st.expander(header, expanded=True):
                reason = r.get("reason_note") or "—"
                st.markdown(f"**Why it sold:** {reason}")
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.caption("Bike")
                    st.markdown(f"{r.get('brand', '—')} · {r.get('category', '—')}")
                with c2:
                    st.caption("Performance")
                    st.markdown(f"Score {int(r.get('sell_difficulty_score', 0))}/5 · {int(r.get('days_on_market', 0))}d · {int(r.get('campaigns_run', 0))} runs")
                with c3:
                    st.caption("Strategy")
                    angle = r.get("selling_angle") or "—"
                    st.markdown(f"{angle[:80]}")
                with c4:
                    st.caption("Tone & Audience")
                    tone = r.get("tone") or "—"
                    audience = r.get("target_audience") or "—"
                    st.markdown(f"{tone} · {audience[:40]}")

        st.markdown("### Full table")
        kb_cols = [c for c in [
            "bike_id", "trial_num", "date", "title", "brand", "category",
            "price", "sell_difficulty_score", "days_on_market",
            "campaigns_run", "tone", "selling_angle", "target_audience",
            "reason_note",
        ] if c in view.columns]
        st.dataframe(view[kb_cols], width="stretch", hide_index=True)

    st.info(
        "**Production roadmap** — The knowledge base is currently a flat CSV with simple brand/category "
        "retrieval. In production, this would be stored in a vector database (Pinecone, Weaviate, or "
        "pgvector) with embeddings, enabling semantic retrieval — e.g. \"what worked for expensive "
        "trekking bikes in winter?\". The sale-reason agent would also ingest real buyer feedback "
        "(reviews, survey responses) instead of inferring reasons from campaign data alone. "
        "Over time this becomes the core learning loop: sell → learn → improve → sell better."
    )
