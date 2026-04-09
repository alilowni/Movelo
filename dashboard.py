"""
movelo Marketing Dashboard

Run with:
    streamlit run dashboard.py
    -- or --
    python main.py --dashboard
"""

import subprocess
import sys
import pandas as pd
import streamlit as st
from pathlib import Path

from scoring import (
    load_and_score,
    load_trials,
    join_bikes_and_trials,
    load_knowledge_base,
)

st.set_page_config(page_title="movelo Dashboard", page_icon="🚲", layout="wide")


@st.cache_data(ttl=10)
def get_bikes() -> pd.DataFrame:
    return load_and_score()


@st.cache_data(ttl=10)
def get_trials() -> pd.DataFrame:
    return load_trials()


@st.cache_data(ttl=10)
def get_joined() -> pd.DataFrame:
    return join_bikes_and_trials()


@st.cache_data(ttl=10)
def get_knowledge_base() -> pd.DataFrame:
    return load_knowledge_base()


def danger_color(score: int) -> str:
    if score >= 4:
        return "🔴"
    if score == 3:
        return "🟡"
    return "🟢"


def status_badge(status: str) -> str:
    return "🏷 SOLD" if status == "sold" else "✅ Available"


# ---------------------------------------------------------------------------
# Campaign card renderer
# ---------------------------------------------------------------------------

def _render_campaign_card(row: pd.Series, bikes_df: pd.DataFrame) -> None:
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


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("movelo")
st.sidebar.caption("Refurbished Bike Marketing Dashboard")

page = st.sidebar.radio(
    "Navigate",
    ["Bike Inventory", "Campaigns", "Run Campaign", "Analytics", "Knowledge Base"],
)

st.sidebar.markdown("---")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------------------------
# Page 1: Bike Inventory
# ---------------------------------------------------------------------------

if page == "Bike Inventory":
    st.title("Bike Inventory")
    bikes = get_bikes()

    n_sold = len(bikes[bikes["status"] == "sold"])
    n_avail = len(bikes) - n_sold
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Available", n_avail)
    col2.metric("Sold", n_sold)
    col3.metric("In danger (4-5)", len(bikes[(bikes["sell_difficulty_score"] >= 4) & (bikes["status"] != "sold")]))
    col4.metric("Avg price", f"€{bikes['price'].mean():.0f}")

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        brands = st.multiselect("Brand", sorted(bikes["brand"].unique()))
    with fcol2:
        categories = st.multiselect("Category", sorted(bikes["category"].unique()))
    with fcol3:
        min_score, max_score = st.slider("Difficulty score", 0, 5, (0, 5))
    with fcol4:
        show_sold = st.checkbox("Show sold bikes", value=False)

    filtered = bikes.copy()
    if brands:
        filtered = filtered[filtered["brand"].isin(brands)]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    filtered = filtered[
        (filtered["sell_difficulty_score"] >= min_score)
        & (filtered["sell_difficulty_score"] <= max_score)
    ]
    if not show_sold:
        filtered = filtered[filtered["status"] != "sold"]

    display = filtered[
        ["id", "title", "brand", "category", "price", "condition",
         "sell_difficulty_score", "days_on_market", "status"]
    ].copy()
    display["danger"] = display["sell_difficulty_score"].apply(danger_color)
    display["status_badge"] = display["status"].apply(status_badge)
    display = display.sort_values("sell_difficulty_score", ascending=False)

    st.dataframe(
        display[["danger", "id", "title", "brand", "category", "price",
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
            "danger": " ",
        },
    )

    st.markdown("### Bike Details & Actions")
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
            st.markdown(f"Brand: {bike_row['brand']} | Category: {bike_row['category']}")
            st.markdown(f"Price: **€{bike_row['price']:.0f}** | Condition: {bike_row['condition']}")
            km = bike_row['km_ridden']
            st.markdown(f"KM: {km if pd.notna(km) else 'N/A'} | Year: {int(bike_row['year']) if pd.notna(bike_row['year']) else 'N/A'}")
            st.markdown(f"Score: {danger_color(bike_row['sell_difficulty_score'])} **{bike_row['sell_difficulty_score']}**/5 | Days listed: {bike_row['days_on_market']}")

            desc = bike_row.get("description", "")
            if pd.notna(desc) and desc:
                st.caption(desc)

            product_url = bike_row.get("product_url", "")
            if pd.notna(product_url) and product_url:
                st.link_button("View on website", product_url)

            st.markdown("---")
            current_status = bike_row.get("status", "available")
            if current_status == "sold":
                st.success("This bike is marked as SOLD")
            else:
                st.info("Sales happen automatically at the end of each campaign.")


# ---------------------------------------------------------------------------
# Page 2: Campaigns
# ---------------------------------------------------------------------------

elif page == "Campaigns":
    st.title("Marketing Campaigns")
    trials = get_trials()
    bikes_df = get_bikes()

    if trials.empty:
        st.info("No campaigns yet. Go to **Run Campaign** to start one.")
    else:
        view_mode = st.radio("View by", ["Campaign", "Bike journey"],
                             horizontal=True, label_visibility="collapsed")

        if view_mode == "Campaign":
            trial_nums = sorted(trials["trial_num"].unique())
            tcol1, tcol2 = st.columns([1, 3])
            with tcol1:
                st.metric("Total campaigns", len(trial_nums))
            with tcol2:
                selected_trial = st.selectbox("Select campaign", trial_nums,
                                              format_func=lambda x: f"Campaign {int(x)}")

            trial_data = trials[trials["trial_num"] == selected_trial]
            trial_date = trial_data["date"].dropna().values[0] if not trial_data["date"].dropna().empty else "N/A"
            st.markdown(f"### Campaign {int(selected_trial)} — {trial_date} — {len(trial_data)} bikes")

            for _, row in trial_data.iterrows():
                _render_campaign_card(row, bikes_df)

        else:
            targeted_ids = sorted(trials["bike_id"].unique())
            selected_bike = st.selectbox(
                "Select bike to see full journey",
                targeted_ids,
                format_func=lambda x: (
                    f"[{int(x)}] "
                    f"{bikes_df[bikes_df['id']==int(x)]['title'].values[0] if not bikes_df[bikes_df['id']==int(x)].empty else x}"
                    f"{' 🏷 SOLD' if not bikes_df[bikes_df['id']==int(x)].empty and bikes_df[bikes_df['id']==int(x)]['status'].values[0]=='sold' else ''}"
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
                st.dataframe(
                    bike_trials[["trial_num", "date", "selling_angle", "target_audience",
                                 "tone", "actions", "email_subject"]],
                    width="stretch",
                    hide_index=True,
                )
                for _, row in bike_trials.iterrows():
                    _render_campaign_card(row, bikes_df)


# ---------------------------------------------------------------------------
# Page 3: Run Campaign (next day)
# ---------------------------------------------------------------------------

elif page == "Run Campaign":
    st.title("Run Next Campaign")
    bikes_df = get_bikes()
    trials = get_trials()

    n_avail = len(bikes_df[bikes_df["status"] != "sold"])
    n_sold = len(bikes_df[bikes_df["status"] == "sold"])
    current_day = int(bikes_df[bikes_df["status"] != "sold"]["days_on_market"].max()) if n_avail > 0 else 0
    n_campaigns = int(trials["trial_num"].max()) if not trials.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Day", current_day)
    col2.metric("Campaigns", n_campaigns)
    col3.metric("Available", n_avail)
    col4.metric("Sold", n_sold)

    if n_avail == 0:
        st.success("All bikes are sold!")
    else:
        st.markdown(
            f"Clicking the button below will **advance 1 day** (scores update), "
            f"then run a full marketing campaign for hard-to-sell bikes."
        )

        if st.button("Run next campaign", type="primary"):
            log_area = st.empty()
            with st.spinner("Running simulation..."):
                result = subprocess.run(
                    [sys.executable, "main.py"],
                    capture_output=True,
                    text=True,
                    cwd=str(Path(__file__).parent),
                    timeout=600,
                )
                output = result.stdout
                if result.stderr:
                    output += "\n" + result.stderr

            if result.returncode == 0:
                st.success("Campaign complete!")
            else:
                st.error(f"Pipeline exited with code {result.returncode}")

            st.markdown("### Pipeline log")
            st.code(output, language=None)

            st.cache_data.clear()
            st.markdown("---")
            st.markdown("Refresh the page or click **Campaigns** to see results.")


# ---------------------------------------------------------------------------
# Page 4: Analytics
# ---------------------------------------------------------------------------

elif page == "Analytics":
    st.title("Analytics")
    bikes_df = get_bikes()
    trials = get_trials()
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

    st.markdown("### Bikes in danger (available, score >= 4)")
    danger = bikes_df[(bikes_df["sell_difficulty_score"] >= 4) & (bikes_df["status"] != "sold")].copy()

    if not trials.empty:
        trial_counts = trials.groupby("bike_id").size().reset_index(name="times_marketed")
        danger = danger.merge(trial_counts, left_on="id", right_on="bike_id", how="left")
        danger["times_marketed"] = danger["times_marketed"].fillna(0).astype(int)
    else:
        danger["times_marketed"] = 0

    danger = danger.sort_values("sell_difficulty_score", ascending=False)
    st.dataframe(
        danger[["id", "title", "brand", "price", "condition",
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

    never_marketed = danger[danger["times_marketed"] == 0]
    if not never_marketed.empty:
        st.warning(f"{len(never_marketed)} high-danger bikes have NEVER been marketed:")
        for _, r in never_marketed.iterrows():
            st.markdown(f"- **[{r['id']}] {r['title']}** — €{r['price']:.0f}, score {r['sell_difficulty_score']}, {r['days_on_market']} days")

    # -- Trend charts --
    if not trials.empty and "date" in trials.columns:
        st.markdown("### Trends")
        campaign_dates = (
            trials.groupby("trial_num")["date"]
            .first()
            .reset_index()
            .rename(columns={"date": "Date", "trial_num": "Campaign"})
            .sort_values("Campaign")
        )

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

    st.markdown("### Marketing frequency by brand")
    if not trials.empty:
        brand_merge = trials.merge(bikes_df[["id", "brand"]], left_on="bike_id", right_on="id", how="left")
        brand_counts = brand_merge.groupby("brand").size().reset_index(name="campaigns")
        brand_counts = brand_counts.sort_values("campaigns", ascending=False)
        st.bar_chart(brand_counts.set_index("brand")["campaigns"])

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


# ---------------------------------------------------------------------------
# Page 5: Knowledge Base
# ---------------------------------------------------------------------------

elif page == "Knowledge Base":
    st.title("Sales Knowledge Base")
    st.caption(
        "Short notes on **why each sold bike likely sold** — captured automatically "
        "the moment a bike sells. Use this to spot what works so future campaigns "
        "can lean into it."
    )

    kb = get_knowledge_base()

    if kb.empty:
        st.info(
            "No sales recorded yet. Run campaigns from **Run Campaign** — every "
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

        for _, row in view.iterrows():
            header = (
                f"Bike {int(row['bike_id'])} — {row['title']} "
                f"(€{float(row['price']):.0f}) · sold in campaign "
                f"{int(row['trial_num'])}"
            )
            with st.expander(header, expanded=True):
                st.markdown(f"**Why it sold:** {row['reason_note']}")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown(f"**Brand:** {row['brand']}")
                    st.markdown(f"**Category:** {row['category']}")
                with c2:
                    st.markdown(f"**Score:** {int(row['sell_difficulty_score'])}/5")
                    st.markdown(f"**Days listed:** {int(row['days_on_market'])}")
                with c3:
                    st.markdown(f"**Campaigns run:** {int(row['campaigns_run'])}")
                    st.markdown(f"**Tone used:** {row['tone'] or '—'}")
                if row.get("selling_angle"):
                    st.markdown(f"**Winning angle:** {row['selling_angle']}")
                if row.get("target_audience"):
                    st.markdown(f"**Audience:** {row['target_audience']}")

        st.markdown("### Full table")
        st.dataframe(
            view[[
                "bike_id", "trial_num", "date", "title", "brand", "category",
                "price", "sell_difficulty_score", "days_on_market",
                "campaigns_run", "tone", "selling_angle", "target_audience",
                "reason_note",
            ]],
            width="stretch",
            hide_index=True,
        )
