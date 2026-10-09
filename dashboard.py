"""
Fashion Retailer — Loyalty Campaign Dashboard
Designed for the Marketing Director
"""

import sys
sys.path.insert(0, '/Users/azlanghauri/Library/Python/3.9/lib/python/site-packages')

import pathlib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loyalty Campaign Dashboard",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colour palette ────────────────────────────────────────────────────────
PALETTE = {
    "primary":   "#1B2A4A",   # deep navy
    "accent":    "#C8973A",   # warm gold
    "highlight": "#E05C5C",   # coral
    "muted":     "#7B8FA1",   # slate
    "bg":        "#F7F4F0",   # off-white
}

SEGMENT_COLOURS = {
    "VIP Loyalists":       "#1B2A4A",
    "Rising Stars":        "#C8973A",
    "Price-Driven Buyers": "#E05C5C",
    "Casual Browsers":     "#7B8FA1",
}

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F7F4F0; }
    section[data-testid="stSidebar"] { background-color: #1B2A4A; }
    section[data-testid="stSidebar"] * { color: #F7F4F0 !important; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        border-left: 5px solid #C8973A;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 4px;
    }
    .kpi-label  { font-size: 13px; color: #7B8FA1; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
    .kpi-value  { font-size: 32px; color: #1B2A4A; font-weight: 700; line-height: 1.2; }
    .kpi-sub    { font-size: 13px; color: #7B8FA1; }

    /* Section headings */
    h2 { color: #1B2A4A !important; }
    h3 { color: #1B2A4A !important; }

    /* Segment badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        color: white;
    }
    .divider { border-top: 2px solid #E8E2DA; margin: 24px 0; }
    .insight-box {
        background: #EEF3F8;
        border-left: 4px solid #1B2A4A;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
        font-size: 14px;
        color: #1B2A4A;
    }
    .recommendation-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-top: 4px solid #C8973A;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        margin-bottom: 16px;
    }
    .rec-title { font-size: 16px; font-weight: 700; color: #1B2A4A; }
    .rec-body  { font-size: 13px; color: #444; margin-top: 6px; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Data loading & caching ──────────────────────────────────────────────────────
DATA_PATH = pathlib.Path(__file__).parent / "Fashion_Retailer_Customer_Data.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(DATA_PATH)
    return df

@st.cache_data
def build_segments(df: pd.DataFrame):
    """
    K-Means clustering on four behavioural signals:
      • Total 12-month spend
      • Purchase frequency
      • Average order value
      • Days since last purchase (inverted — lower = better)
    Returns df with a 'Segment' column.
    """
    features = ["Total_Spend_12M_AUD", "Purchase_Frequency_12M",
                "Average_Order_Value_AUD", "Days_Since_Last_Purchase"]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reproducible clustering with k=4
    km = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    df = df.copy()
    df["_cluster"] = labels

    # Label clusters by their centroid spend + recency characteristics
    centers = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=features
    )
    centers["cluster"] = range(4)

    # Sort by spend desc, then recency asc to assign meaningful names
    centers_sorted = centers.sort_values(
        ["Total_Spend_12M_AUD", "Days_Since_Last_Purchase"],
        ascending=[False, True]
    ).reset_index(drop=True)

    name_map_raw = {
        0: "VIP Loyalists",
        1: "Rising Stars",
        2: "Price-Driven Buyers",
        3: "Casual Browsers",
    }
    cluster_to_name = {}
    for rank, row in centers_sorted.iterrows():
        cluster_to_name[int(row["cluster"])] = name_map_raw[rank]

    df["Segment"] = df["_cluster"].map(cluster_to_name)
    df.drop(columns=["_cluster"], inplace=True)
    return df

# ── Load ────────────────────────────────────────────────────────────────────────
raw_df = load_data()
df = build_segments(raw_df)

# ── Sidebar filters ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fashion.png", width=60)
    st.title("Filters")
    st.caption("Refine the view using the options below.")

    st.markdown("**Age Band**")
    age_options = sorted(df["Age_Band"].unique())
    sel_age = st.multiselect("", age_options, default=age_options, key="age",
                              label_visibility="collapsed")

    st.markdown("**Location**")
    loc_options = sorted(df["Location_Type"].unique())
    sel_loc = st.multiselect("", loc_options, default=loc_options, key="loc",
                              label_visibility="collapsed")

    st.markdown("**Preferred Channel**")
    chan_options = sorted(df["Preferred_Channel"].unique())
    sel_chan = st.multiselect("", chan_options, default=chan_options, key="chan",
                               label_visibility="collapsed")

    st.markdown("**Loyalty Status**")
    loy_options = sorted(df["Loyalty_Member"].unique())
    sel_loy = st.multiselect("", loy_options, default=loy_options, key="loy",
                               label_visibility="collapsed")

    st.divider()
    st.caption("Data: Fashion_Retailer_Customer_Data.xlsx · 60 customers")

# Apply filters
mask = (
    df["Age_Band"].isin(sel_age) &
    df["Location_Type"].isin(sel_loc) &
    df["Preferred_Channel"].isin(sel_chan) &
    df["Loyalty_Member"].isin(sel_loy)
)
fdf = df[mask].copy()

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#1B2A4A;padding:24px 32px;border-radius:14px;margin-bottom:24px;'>
  <h1 style='color:#F7F4F0;margin:0;font-size:26px;'>👗 Loyalty Campaign Dashboard</h1>
  <p style='color:#C8973A;margin:6px 0 0 0;font-size:15px;'>
    Customer Segmentation & Campaign Recommendations &nbsp;·&nbsp; Marketing Director View
  </p>
</div>
""", unsafe_allow_html=True)

if fdf.empty:
    st.warning("No customers match the current filters. Please adjust the sidebar.")
    st.stop()

# ── KPI Row ─────────────────────────────────────────────────────────────────────
total_customers   = len(fdf)
total_revenue     = fdf["Total_Spend_12M_AUD"].sum()
avg_order_val     = fdf["Average_Order_Value_AUD"].mean()
avg_purchase_freq = fdf["Purchase_Frequency_12M"].mean()
loyalty_pct       = (fdf["Loyalty_Member"] == "Yes").mean() * 100
avg_recency       = fdf["Days_Since_Last_Purchase"].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)

def kpi(col, label, value, sub=""):
    col.markdown(f"""
    <div class='kpi-card'>
      <div class='kpi-label'>{label}</div>
      <div class='kpi-value'>{value}</div>
      <div class='kpi-sub'>{sub}</div>
    </div>""", unsafe_allow_html=True)

kpi(c1, "Customers",       f"{total_customers:,}",       "in filtered view")
kpi(c2, "Total Revenue",   f"${total_revenue:,.0f}",      "12-month spend (AUD)")
kpi(c3, "Avg Order Value", f"${avg_order_val:.0f}",       "per transaction")
kpi(c4, "Avg Purchases",   f"{avg_purchase_freq:.1f}",    "per year")
kpi(c5, "Loyalty Members", f"{loyalty_pct:.0f}%",         "of filtered customers")
kpi(c6, "Avg Recency",     f"{avg_recency:.0f} days",     "since last purchase")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CUSTOMER SEGMENTS
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("🎯 Customer Segments")
st.caption("Four behavioural segments identified via K-Means clustering on spend, frequency, order value and recency.")

seg_summary = fdf.groupby("Segment").agg(
    Customers=("Customer_ID", "count"),
    Avg_Spend=("Total_Spend_12M_AUD", "mean"),
    Avg_Order=("Average_Order_Value_AUD", "mean"),
    Avg_Freq=("Purchase_Frequency_12M", "mean"),
    Avg_Recency=("Days_Since_Last_Purchase", "mean"),
    Avg_App=("App_Engagement_Score", "mean"),
    Discount_Pct=("Discount_Usage_Pct", "mean"),
    Loyalty_Yes=("Loyalty_Member", lambda x: (x == "Yes").sum()),
).reset_index()
seg_summary["Loyalty_%"] = (seg_summary["Loyalty_Yes"] / seg_summary["Customers"] * 100).round(0)

# Segment tile cards
seg_icons = {
    "VIP Loyalists":       ("👑", "#1B2A4A"),
    "Rising Stars":        ("⭐", "#C8973A"),
    "Price-Driven Buyers": ("🏷️", "#E05C5C"),
    "Casual Browsers":     ("🛍️", "#7B8FA1"),
}
seg_descs = {
    "VIP Loyalists":
        "High-spend, frequent, recent buyers. Already loyal — focus on exclusivity rewards and early access.",
    "Rising Stars":
        "Growing spend and frequency. Prime candidates for loyalty programme enrolment and personalised incentives.",
    "Price-Driven Buyers":
        "Moderate spend but heavy discount usage. Respond well to targeted promotions and bundle offers.",
    "Casual Browsers":
        "Low frequency and spend. Re-engagement campaigns, welcome-back vouchers and push notifications.",
}

preferred_order = ["VIP Loyalists", "Rising Stars", "Price-Driven Buyers", "Casual Browsers"]
cols_seg = st.columns(4)
for i, seg_name in enumerate(preferred_order):
    row = seg_summary[seg_summary["Segment"] == seg_name]
    if row.empty:
        continue
    row = row.iloc[0]
    icon, colour = seg_icons.get(seg_name, ("📌", "#888"))
    cols_seg[i].markdown(f"""
    <div style='background:white;border-radius:12px;padding:18px;
                border-top:5px solid {colour};box-shadow:0 2px 8px rgba(0,0,0,0.07);
                min-height:220px;'>
      <div style='font-size:28px'>{icon}</div>
      <div style='font-weight:700;font-size:15px;color:{colour};margin:6px 0 4px 0;'>{seg_name}</div>
      <div style='font-size:22px;font-weight:700;color:#1B2A4A;'>{int(row['Customers'])} customers</div>
      <hr style='border:0;border-top:1px solid #eee;margin:8px 0;'>
      <div style='font-size:12px;color:#555;line-height:1.6;'>
        💰 Avg Spend: <b>${row['Avg_Spend']:,.0f}</b><br>
        🛒 Avg Order: <b>${row['Avg_Order']:,.0f}</b><br>
        🔁 Freq/yr:   <b>{row['Avg_Freq']:.1f}</b><br>
        📅 Recency:   <b>{row['Avg_Recency']:.0f} days</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Segment size + spend side-by-side ───────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    fig_pie = px.pie(
        seg_summary, values="Customers", names="Segment",
        title="Share of Customers by Segment",
        color="Segment",
        color_discrete_map=SEGMENT_COLOURS,
        hole=0.45,
    )
    fig_pie.update_traces(textposition="outside", textinfo="percent+label",
                          textfont_size=12)
    fig_pie.update_layout(
        showlegend=False, margin=dict(t=50, b=10, l=10, r=10),
        title_font_color="#1B2A4A",
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    seg_bar = seg_summary.copy()
    seg_bar["Segment_ord"] = pd.Categorical(
        seg_bar["Segment"], categories=preferred_order, ordered=True)
    seg_bar = seg_bar.sort_values("Segment_ord")

    fig_spend = px.bar(
        seg_bar, x="Segment", y="Avg_Spend",
        title="Average 12-Month Spend per Segment (AUD)",
        color="Segment",
        color_discrete_map=SEGMENT_COLOURS,
        text=seg_bar["Avg_Spend"].apply(lambda v: f"${v:,.0f}"),
    )
    fig_spend.update_traces(textposition="outside")
    fig_spend.update_layout(
        showlegend=False, yaxis_title="Avg Spend (AUD)", xaxis_title="",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    )
    st.plotly_chart(fig_spend, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SPENDING & BEHAVIOUR DEEP DIVE
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("💰 Spending & Behaviour Deep Dive")

col_left, col_right = st.columns(2)

# Spend by Age Band
with col_left:
    age_order = ["18-24", "25-34", "35-49", "50-64", "65+"]
    age_spend = fdf.groupby("Age_Band")["Total_Spend_12M_AUD"].mean().reindex(age_order).reset_index()
    fig_age = px.bar(
        age_spend, x="Age_Band", y="Total_Spend_12M_AUD",
        title="Average Annual Spend by Age Band",
        color="Total_Spend_12M_AUD",
        color_continuous_scale=[[0, "#E8E2DA"], [1, "#1B2A4A"]],
        text=age_spend["Total_Spend_12M_AUD"].apply(lambda v: f"${v:,.0f}"),
    )
    fig_age.update_traces(textposition="outside")
    fig_age.update_layout(
        coloraxis_showscale=False, showlegend=False,
        yaxis_title="Avg Spend (AUD)", xaxis_title="Age Band",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    )
    st.plotly_chart(fig_age, use_container_width=True)

# Channel breakdown
with col_right:
    chan_seg = fdf.groupby(["Preferred_Channel", "Segment"])["Customer_ID"].count().reset_index()
    chan_seg.columns = ["Channel", "Segment", "Count"]
    fig_chan = px.bar(
        chan_seg, x="Channel", y="Count", color="Segment",
        title="Customer Distribution: Channel × Segment",
        color_discrete_map=SEGMENT_COLOURS,
        barmode="stack",
    )
    fig_chan.update_layout(
        yaxis_title="Customers", xaxis_title="Preferred Channel",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
        legend_title_text="Segment",
    )
    st.plotly_chart(fig_chan, use_container_width=True)

# Spend distribution box
col_c, col_d = st.columns(2)
with col_c:
    fig_box = px.box(
        fdf, x="Segment", y="Total_Spend_12M_AUD",
        title="Spend Distribution by Segment",
        color="Segment", color_discrete_map=SEGMENT_COLOURS,
        category_orders={"Segment": preferred_order},
        points="all",
    )
    fig_box.update_layout(
        showlegend=False, yaxis_title="Annual Spend (AUD)", xaxis_title="",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    )
    st.plotly_chart(fig_box, use_container_width=True)

# Recency vs Spend scatter
with col_d:
    fig_scatter = px.scatter(
        fdf,
        x="Days_Since_Last_Purchase",
        y="Total_Spend_12M_AUD",
        color="Segment",
        size="Purchase_Frequency_12M",
        hover_name="Customer_ID",
        hover_data=["Age_Band", "Preferred_Channel"],
        title="Recency vs Spend (bubble = purchase frequency)",
        color_discrete_map=SEGMENT_COLOURS,
        category_orders={"Segment": preferred_order},
    )
    fig_scatter.update_layout(
        xaxis_title="Days Since Last Purchase",
        yaxis_title="Annual Spend (AUD)",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
        legend_title_text="Segment",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 3 — APP ENGAGEMENT & DISCOUNT BEHAVIOUR
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("📱 App Engagement & Discount Sensitivity")

col_e, col_f = st.columns(2)

with col_e:
    fig_app = px.violin(
        fdf, x="Segment", y="App_Engagement_Score",
        color="Segment", color_discrete_map=SEGMENT_COLOURS,
        box=True, points="all",
        title="App Engagement Score by Segment",
        category_orders={"Segment": preferred_order},
    )
    fig_app.update_layout(
        showlegend=False, yaxis_title="App Engagement (0–100)", xaxis_title="",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    )
    st.plotly_chart(fig_app, use_container_width=True)

with col_f:
    fig_disc = px.bar(
        seg_summary.sort_values("Discount_Pct", ascending=False),
        x="Segment", y="Discount_Pct",
        color="Segment", color_discrete_map=SEGMENT_COLOURS,
        text=seg_summary.sort_values("Discount_Pct", ascending=False)["Discount_Pct"].apply(lambda v: f"{v:.0f}%"),
        title="Average Discount Usage % by Segment",
    )
    fig_disc.update_traces(textposition="outside")
    fig_disc.update_layout(
        showlegend=False, yaxis_title="Avg Discount Usage (%)", xaxis_title="",
        margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    )
    st.plotly_chart(fig_disc, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PRODUCT PREFERENCES
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("👚 Product Preferences by Segment")

prod_seg = fdf.groupby(["Segment", "Product_Preference"])["Customer_ID"].count().reset_index()
prod_seg.columns = ["Segment", "Product", "Count"]

fig_prod = px.bar(
    prod_seg, x="Segment", y="Count", color="Product",
    title="Product Category Preferences across Segments",
    barmode="group",
    category_orders={"Segment": preferred_order},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig_prod.update_layout(
    yaxis_title="Customers", xaxis_title="",
    margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    legend_title_text="Product",
)
st.plotly_chart(fig_prod, use_container_width=True)

# Location heatmap
loc_seg = fdf.groupby(["Location_Type", "Segment"])["Customer_ID"].count().unstack(fill_value=0)
fig_heat = px.imshow(
    loc_seg,
    title="Customer Count: Location × Segment",
    color_continuous_scale=[[0, "#F7F4F0"], [1, "#1B2A4A"]],
    text_auto=True,
    aspect="auto",
)
fig_heat.update_layout(
    xaxis_title="Segment", yaxis_title="Location",
    margin=dict(t=50, b=10), title_font_color="#1B2A4A",
    coloraxis_showscale=False,
)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 5 — LOYALTY CAMPAIGN RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("🚀 Loyalty Campaign Recommendations")
st.caption("Tailored actions for each segment to maximise ROI on the upcoming campaign.")

rec_data = {
    "VIP Loyalists": {
        "icon": "👑",
        "colour": "#1B2A4A",
        "priority": "High — Protect & Elevate",
        "tactics": [
            "Exclusive Tier: Launch a 'Platinum' loyalty tier with early access to new collections.",
            "Personal Shopping: Assign a dedicated style advisor or virtual concierge.",
            "Birthday Moments: Premium gift-with-purchase, personalised packaging.",
            "Referral Boost: Double loyalty points for every friend referred.",
        ],
        "channel": "Email + App push (high engagement score) + in-store invitation events.",
        "why": "These customers already spend the most and buy frequently. Small investments in exclusivity yield outsized retention and word-of-mouth value.",
    },
    "Rising Stars": {
        "icon": "⭐",
        "colour": "#C8973A",
        "priority": "High — Accelerate Loyalty",
        "tactics": [
            "Fast-Track Enrolment: Offer double points for first 3 months of loyalty membership.",
            "Milestone Rewards: Surprise unlocks at spend thresholds ($500, $1,000, $1,500).",
            "Cross-Sell: Personalised 'complete the look' recommendations tied to past purchases.",
            "App Incentive: Exclusive in-app flash sales to boost app engagement.",
        ],
        "channel": "App push + targeted email + social retargeting.",
        "why": "Spend and frequency are growing. These customers are on the cusp of VIP — a well-timed nudge converts them into long-term high-value buyers.",
    },
    "Price-Driven Buyers": {
        "icon": "🏷️",
        "colour": "#E05C5C",
        "priority": "Medium — Shift Behaviour",
        "tactics": [
            "Smart Bundles: Bundle promotions that deliver perceived value without heavy discounting.",
            "Points for Full-Price Buys: Extra loyalty points on non-discounted purchases.",
            "Seasonal Previews: Give early access to sales — rewards loyalty, reduces dependency on public promotions.",
            "Value Content: 'Style for less' editorial content that showcases quality over price.",
        ],
        "channel": "Email with strong subject lines + SMS for time-limited deals.",
        "why": "Heavy discount usage compresses margins. The goal is to gradually shift this group towards value-based purchasing without alienating them.",
    },
    "Casual Browsers": {
        "icon": "🛍️",
        "colour": "#7B8FA1",
        "priority": "Medium — Re-engage",
        "tactics": [
            "Win-Back Voucher: '$20 off your next purchase' triggered after 30 days of inactivity.",
            "Trend Alerts: 'Just In' notifications aligned to their preferred product category.",
            "Low-Barrier Loyalty: 'Earn points on your first purchase this month' — removes friction.",
            "Social Proof: Show popular items and customer reviews to build purchase confidence.",
        ],
        "channel": "Email (low app engagement) + SMS for re-engagement offers.",
        "why": "These customers haven't fully committed yet. Re-engagement is cost-effective — they know the brand. The focus is reducing friction and providing a reason to return.",
    },
}

for seg_name in preferred_order:
    rec = rec_data[seg_name]
    row = seg_summary[seg_summary["Segment"] == seg_name]
    if row.empty:
        continue
    row = row.iloc[0]
    st.markdown(f"""
    <div class='recommendation-card'>
      <div style='display:flex;align-items:center;gap:10px;'>
        <span style='font-size:28px'>{rec['icon']}</span>
        <div>
          <div class='rec-title'>{seg_name}</div>
          <span style='background:{rec['colour']};color:white;font-size:11px;
                        font-weight:700;padding:2px 10px;border-radius:20px;'>
            Priority: {rec['priority']}
          </span>
        </div>
        <div style='margin-left:auto;text-align:right;'>
          <div style='font-size:12px;color:#888;'>Avg Annual Spend</div>
          <div style='font-size:20px;font-weight:700;color:{rec["colour"]};'>${row['Avg_Spend']:,.0f}</div>
        </div>
      </div>
      <hr style='border:0;border-top:1px solid #eee;margin:12px 0;'>
      <div class='rec-body'>
        <b>Campaign Tactics:</b>
        <ul style='margin:6px 0 10px 0;padding-left:18px;'>
          {''.join(f"<li>{t}</li>" for t in rec['tactics'])}
        </ul>
        <b>Recommended Channel:</b> {rec['channel']}<br><br>
        <b>Why this works:</b> {rec['why']}
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 6 — RADAR CHART SEGMENT COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("📊 Segment Comparison (Radar)")
st.caption("Normalised scores across five dimensions — higher = better for marketing attractiveness.")

radar_metrics = {
    "Spend":     "Total_Spend_12M_AUD",
    "Frequency": "Purchase_Frequency_12M",
    "Recency":   "Days_Since_Last_Purchase",   # inverted
    "App":       "App_Engagement_Score",
    "Order Val": "Average_Order_Value_AUD",
}

seg_radar = fdf.groupby("Segment")[list(radar_metrics.values())].mean()
# Invert recency (fewer days = better)
seg_radar["Days_Since_Last_Purchase"] = seg_radar["Days_Since_Last_Purchase"].max() - seg_radar["Days_Since_Last_Purchase"]

# Normalise 0-1
for col in seg_radar.columns:
    mn, mx = seg_radar[col].min(), seg_radar[col].max()
    if mx > mn:
        seg_radar[col] = (seg_radar[col] - mn) / (mx - mn)

categories = list(radar_metrics.keys())

fig_radar = go.Figure()
for seg_name in preferred_order:
    if seg_name not in seg_radar.index:
        continue
    vals = [seg_radar.loc[seg_name, v] for v in radar_metrics.values()]
    vals += [vals[0]]  # close the loop
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=categories + [categories[0]],
        fill="toself",
        name=seg_name,
        line_color=SEGMENT_COLOURS.get(seg_name, "#888"),
        opacity=0.7,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
    showlegend=True,
    title="Segment Profiles — Normalised (0 = lowest, 1 = highest in each dimension)",
    title_font_color="#1B2A4A",
    margin=dict(t=60, b=20),
    height=480,
)
st.plotly_chart(fig_radar, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RAW DATA EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with st.expander("🗂️ View Customer Data Table", expanded=False):
    display_df = fdf.drop(columns=["_cluster"] if "_cluster" in fdf.columns else [])
    seg_filter = st.multiselect(
        "Filter by segment",
        options=preferred_order,
        default=preferred_order,
        key="table_seg",
    )
    table_df = display_df[display_df["Segment"].isin(seg_filter)]
    st.dataframe(
        table_df.reset_index(drop=True),
        use_container_width=True,
        height=350,
    )
    st.caption(f"Showing {len(table_df)} customers")

# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:20px;color:#7B8FA1;font-size:12px;margin-top:20px;'>
  Fashion Retailer Loyalty Dashboard · Built with Streamlit & Plotly ·
  Segmentation via K-Means clustering · Data: 60 customers, 12-month window
</div>
""", unsafe_allow_html=True)
