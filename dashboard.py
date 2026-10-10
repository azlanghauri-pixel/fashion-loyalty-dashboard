"""
Fashion Retailer — Loyalty Campaign Dashboard
Decision-support tool only. Campaign recommendations require human review
and pilot testing before implementation.
"""
import pathlib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Loyalty Campaign Dashboard",
    page_icon="👗",
    layout="wide",
)

# ── Colour palette ─────────────────────────────────────────────────────────────
SEGMENT_COLOURS = {
    "Segment A — High-value active members":      "#1B2A4A",
    "Segment B — High-value active non-members":  "#C8973A",
    "Segment C — Members for re-engagement":      "#4A90A4",
    "Not currently prioritised":                  "#C8C8C8",
}
preferred_order = [
    "Segment A — High-value active members",
    "Segment B — High-value active non-members",
    "Segment C — Members for re-engagement",
    "Not currently prioritised",
]

# ── Load & segment ─────────────────────────────────────────────────────────────
DATA_PATH = pathlib.Path(__file__).parent / "Fashion_Retailer_Customer_Data.xlsx"

@st.cache_data
def load_and_segment():
    df = pd.read_excel(DATA_PATH)

    median_spend = df["Total_Spend_12M_AUD"].median()       # expected: 1175
    median_freq  = df["Purchase_Frequency_12M"].median()    # expected: 12

    def assign_segment(row):
        high_spend  = row["Total_Spend_12M_AUD"] >= median_spend
        high_freq   = row["Purchase_Frequency_12M"] >= median_freq
        recent      = row["Days_Since_Last_Purchase"] < 30
        member      = row["Loyalty_Member"] == "Yes"

        if member and high_spend and high_freq and recent:
            return "Segment A — High-value active members"
        elif (not member) and high_spend and high_freq and recent:
            return "Segment B — High-value active non-members"
        elif member and row["Days_Since_Last_Purchase"] >= 30:
            return "Segment C — Members for re-engagement"
        else:
            return "Not currently prioritised"

    df["Segment"] = df.apply(assign_segment, axis=1)
    df["_median_spend"] = median_spend
    df["_median_freq"]  = median_freq
    return df

df = load_and_segment()
MEDIAN_SPEND = df["_median_spend"].iloc[0]
MEDIAN_FREQ  = df["_median_freq"].iloc[0]

# ── Prominent notice ───────────────────────────────────────────────────────────
st.warning(
    "⚠️ **Decision-support tool only.** "
    "Campaign recommendations require human review and pilot testing before implementation."
)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("👗 Loyalty Campaign Dashboard")
st.caption("Customer Segmentation & Campaign Recommendations · Marketing Director View")
st.divider()

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("👗 Filters")
    sel_age  = st.multiselect("Age Band",  sorted(df["Age_Band"].unique()),
                               default=sorted(df["Age_Band"].unique()))
    sel_loc  = st.multiselect("Location",  sorted(df["Location_Type"].unique()),
                               default=sorted(df["Location_Type"].unique()))
    sel_chan = st.multiselect("Channel",   sorted(df["Preferred_Channel"].unique()),
                               default=sorted(df["Preferred_Channel"].unique()))
    sel_seg  = st.multiselect("Segment",   preferred_order, default=preferred_order)
    st.divider()
    st.caption("Data: 60 customers · 12-month window")

fdf = df[
    df["Age_Band"].isin(sel_age) &
    df["Location_Type"].isin(sel_loc) &
    df["Preferred_Channel"].isin(sel_chan) &
    df["Segment"].isin(sel_seg)
].copy()

if fdf.empty:
    st.warning("No customers match the current filters.")
    st.stop()

# ── KPIs — full dataset ────────────────────────────────────────────────────────
prioritised = df[df["Segment"] != "Not currently prioritised"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Total Customers",       f"{len(df):,}")
c2.metric("Prioritised (A+B+C)",   f"{len(prioritised):,}")
c3.metric("Combined Spend A+B+C",  f"${prioritised['Total_Spend_12M_AUD'].sum():,.0f}")
c4.metric("Loyalty Members",       f"{(df['Loyalty_Member']=='Yes').sum()}")
c5.metric("Median Annual Spend",   f"${MEDIAN_SPEND:,.0f}")
c6.metric("Median Purchase Freq",  f"{MEDIAN_FREQ:.0f}/yr")
st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SEGMENTATION METHODOLOGY
# ════════════════════════════════════════════════════════════════════════════════
with st.expander("📐 Segmentation Methodology", expanded=True):
    st.markdown("""
**Segments are assigned using deterministic, rule-based logic — no machine learning or probabilistic clustering.**

| Segment | Loyalty Member | Annual Spend | Purchase Frequency | Days Since Last Purchase |
|---|---|---|---|---|
| **A — High-value active members** | Yes | ≥ median (A$1,175) | ≥ median (12/yr) | < 30 days |
| **B — High-value active non-members** | No | ≥ median (A$1,175) | ≥ median (12/yr) | < 30 days |
| **C — Members for re-engagement** | Yes | Any | Any | ≥ 30 days |
| **Not currently prioritised** | Any | Does not meet A/B/C | — | — |

**Thresholds used in this dataset:**
- Median annual spend: **A$1,175**
- Median purchase frequency: **12 purchases/year**
- Recency cutoff: **30 days**

**Rules are applied in order** (A → B → C). A customer qualifying for Segment A will not also appear in C, even if they are a loyalty member.

**What these segments mean for campaign targeting:**
- **Segment A** already shows the behaviours the loyalty programme is designed to reward — high spend, frequent, recent, enrolled.
- **Segment B** shows the same high-value active behaviour without programme membership — the clearest enrolment conversion opportunity.
- **Segment C** are enrolled members who have not purchased recently — re-engagement before lapse is more cost-effective than reacquisition.
""")

# ── Segment summary cards ──────────────────────────────────────────────────────
st.subheader("🎯 Segment Summary")

seg_counts = df.groupby("Segment").agg(
    Customers=("Customer_ID","count"),
    Avg_Spend=("Total_Spend_12M_AUD","mean"),
    Total_Spend=("Total_Spend_12M_AUD","sum"),
    Avg_Order=("Average_Order_Value_AUD","mean"),
    Avg_Freq=("Purchase_Frequency_12M","mean"),
    Avg_Recency=("Days_Since_Last_Purchase","mean"),
    Avg_App=("App_Engagement_Score","mean"),
    Avg_Discount=("Discount_Usage_Pct","mean"),
).reset_index()

icons = {
    "Segment A — High-value active members":     "👑",
    "Segment B — High-value active non-members": "⭐",
    "Segment C — Members for re-engagement":     "🔄",
    "Not currently prioritised":                 "—",
}

display_segments = [s for s in preferred_order if s != "Not currently prioritised"]
cols = st.columns(3)
for i, name in enumerate(display_segments):
    r = seg_counts[seg_counts["Segment"] == name]
    if r.empty:
        continue
    r = r.iloc[0]
    with cols[i]:
        st.markdown(f"### {icons.get(name,'')} {name.split('—')[1].strip()}")
        st.metric("Customers",        int(r["Customers"]))
        st.metric("Avg Annual Spend", f"${r['Avg_Spend']:,.0f}")
        st.metric("Avg Order Value",  f"${r['Avg_Order']:,.0f}")
        st.metric("Purchases/yr",     f"{r['Avg_Freq']:.1f}")
        st.metric("Avg Recency",      f"{r['Avg_Recency']:.0f} days")

not_pri = seg_counts[seg_counts["Segment"] == "Not currently prioritised"]
if not not_pri.empty:
    r = not_pri.iloc[0]
    st.info(f"**Not currently prioritised:** {int(r['Customers'])} customers · "
            f"Avg spend ${r['Avg_Spend']:,.0f} · Avg recency {r['Avg_Recency']:.0f} days")

st.divider()

# ── Charts row 1 ──────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    fig = px.pie(
        seg_counts, values="Customers", names="Segment",
        title="Customer Count by Segment",
        color="Segment", color_discrete_map=SEGMENT_COLOURS, hole=0.4,
    )
    fig.update_traces(textinfo="value+percent")
    fig.update_layout(showlegend=True, legend_title_text="Segment")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    fig = px.bar(
        seg_counts, x="Segment", y="Avg_Spend",
        title="Average Annual Spend by Segment (AUD)",
        color="Segment", color_discrete_map=SEGMENT_COLOURS,
        text=seg_counts["Avg_Spend"].apply(lambda v: f"${v:,.0f}"),
        category_orders={"Segment": preferred_order},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Avg Spend (AUD)", xaxis_title="",
                      xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
st.subheader("💰 Spending & Behaviour")
col_c, col_d = st.columns(2)

with col_c:
    age_order = ["18-24","25-34","35-49","50-64","65+"]
    age_df = fdf.groupby("Age_Band")["Total_Spend_12M_AUD"].mean().reindex(age_order).reset_index()
    fig = px.bar(
        age_df, x="Age_Band", y="Total_Spend_12M_AUD",
        title="Average Annual Spend by Age Band",
        color="Total_Spend_12M_AUD",
        color_continuous_scale=["#E8E2DA","#1B2A4A"],
        text=age_df["Total_Spend_12M_AUD"].apply(lambda v: f"${v:,.0f}"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="Avg Spend (AUD)",
                      xaxis_title="Age Band")
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    fig = px.scatter(
        fdf,
        x="Days_Since_Last_Purchase", y="Total_Spend_12M_AUD",
        color="Segment", size="Purchase_Frequency_12M",
        hover_name="Customer_ID",
        hover_data=["Age_Band","Preferred_Channel","Loyalty_Member"],
        title="Recency vs Annual Spend (bubble = purchase frequency)",
        color_discrete_map=SEGMENT_COLOURS,
        category_orders={"Segment": preferred_order},
    )
    fig.update_layout(
        xaxis_title="Days Since Last Purchase",
        yaxis_title="Annual Spend (AUD)",
        legend_title_text="Segment",
    )
    st.plotly_chart(fig, use_container_width=True)

# Spend distribution by segment
col_e, col_f = st.columns(2)

with col_e:
    fig = px.box(
        fdf, x="Segment", y="Total_Spend_12M_AUD",
        color="Segment", color_discrete_map=SEGMENT_COLOURS,
        title="Spend Distribution by Segment",
        category_orders={"Segment": preferred_order}, points="all",
    )
    fig.update_layout(showlegend=False, yaxis_title="Annual Spend (AUD)",
                      xaxis_title="", xaxis_tickangle=-15)
    st.plotly_chart(fig, use_container_width=True)

with col_f:
    chan_seg = fdf.groupby(["Preferred_Channel","Segment"])["Customer_ID"].count().reset_index()
    chan_seg.columns = ["Channel","Segment","Count"]
    fig = px.bar(
        chan_seg, x="Channel", y="Count", color="Segment",
        title="Customer Count: Channel × Segment",
        color_discrete_map=SEGMENT_COLOURS, barmode="stack",
        category_orders={"Segment": preferred_order},
    )
    fig.update_layout(yaxis_title="Customers", legend_title_text="Segment")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Product preferences ────────────────────────────────────────────────────────
st.subheader("👚 Product Preferences by Segment")
prod = fdf.groupby(["Segment","Product_Preference"])["Customer_ID"].count().reset_index()
prod.columns = ["Segment","Product","Count"]
fig = px.bar(
    prod, x="Segment", y="Count", color="Product",
    title="Product Category Preferences by Segment",
    barmode="group",
    category_orders={"Segment": preferred_order},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(yaxis_title="Customers", xaxis_title="", xaxis_tickangle=-15)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ── Discount usage ─────────────────────────────────────────────────────────────
st.subheader("🏷️ Discount Usage by Segment")
fig = px.box(
    fdf, x="Segment", y="Discount_Usage_Pct",
    color="Segment", color_discrete_map=SEGMENT_COLOURS,
    title="Discount Usage % by Segment",
    category_orders={"Segment": preferred_order}, points="all",
)
fig.update_layout(showlegend=False, yaxis_title="Discount Usage (%)",
                  xaxis_title="", xaxis_tickangle=-15)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION — CAMPAIGN RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════════
st.subheader("🚀 Campaign Recommendations")
st.caption("Directional suggestions only. Each requires human review and pilot testing before implementation.")

rec_data = {
    "Segment A — High-value active members": {
        "icon": "👑",
        "label": "High-value active members",
        "tactics": [
            "Recognise and reward existing loyalty status — personalised thank-you communications",
            "Offer early access to new collections or exclusive member events",
            "Invite to refer friends with a loyalty points incentive",
            "Ensure communications match their preferred channel (in-store, online or mixed)",
        ],
        "channel_note": "Channel preference varies — check individual preferred channel before outreach.",
        "caution": "These customers are already engaged. Over-communication or irrelevant offers risk reducing satisfaction.",
    },
    "Segment B — High-value active non-members": {
        "icon": "⭐",
        "label": "High-value active non-members",
        "tactics": [
            "Present a clear, low-friction loyalty programme enrolment offer",
            "Highlight concrete membership benefits relevant to their spend level",
            "Avoid assuming a single reason for non-membership — test messaging across sub-groups",
        ],
        "channel_note": "Channel preference varies — check individual preferred channel before outreach.",
        "caution": "Non-membership may reflect a deliberate choice. Pilot test enrolment messaging before broad rollout.",
    },
    "Segment C — Members for re-engagement": {
        "icon": "🔄",
        "label": "Members for re-engagement",
        "tactics": [
            "Send a re-engagement communication acknowledging their loyalty membership",
            "Highlight what is new since their last visit",
            "Offer a time-limited incentive relevant to their product preference",
            "Allow easy opt-out — forced re-engagement can accelerate churn",
        ],
        "channel_note": "App engagement is lower on average in this segment. Prefer email or in-store for initial contact.",
        "caution": "Segment C represents an opportunity to test re-engagement before customers become less active. The dataset does not contain acquisition-cost data, so the relative cost-effectiveness of re-engagement versus reacquisition cannot be determined.",
    },
}

for name in display_segments:
    rec = rec_data.get(name)
    if not rec:
        continue
    r = seg_counts[seg_counts["Segment"] == name]
    if r.empty:
        continue
    r = r.iloc[0]
    with st.expander(
        f"{rec['icon']} {rec['label']}  ·  {int(r['Customers'])} customers  ·  Avg spend A${r['Avg_Spend']:,.0f}",
        expanded=True,
    ):
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Suggested tactics:**")
            for t in rec["tactics"]:
                st.markdown(f"- {t}")
            st.markdown(f"**Channel guidance:** {rec['channel_note']}")
            st.info(f"⚠️ **Caution:** {rec['caution']}")
        with col2:
            st.metric("Customers",         int(r["Customers"]))
            st.metric("Avg Annual Spend",  f"${r['Avg_Spend']:,.0f}")
            st.metric("Total Spend",       f"${r['Total_Spend']:,.0f}")
            st.metric("Avg Order Value",   f"${r['Avg_Order']:,.0f}")
            st.metric("Avg Recency",       f"{r['Avg_Recency']:.0f} days")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION — FAIRNESS & RELIABILITY AUDIT
# ════════════════════════════════════════════════════════════════════════════════
with st.expander("⚖️ Fairness & Reliability Audit", expanded=False):

    st.markdown("""
### Fairness & Reliability Audit

This section documents key checks to ensure the segmentation and recommendations do not
introduce bias or unsupported inferences.
""")

    # In-store vs app engagement
    st.markdown("#### 1. App Engagement vs In-Store Customer Value")
    channel_app = df.groupby("Preferred_Channel").agg(
        Avg_Spend=("Total_Spend_12M_AUD","mean"),
        Avg_App=("App_Engagement_Score","mean"),
        Count=("Customer_ID","count"),
    ).reset_index()

    fig = px.scatter(
        channel_app, x="Avg_App", y="Avg_Spend",
        text="Preferred_Channel", size="Count",
        title="Average App Engagement vs Average Annual Spend by Channel",
        labels={"Avg_App":"Avg App Engagement Score","Avg_Spend":"Avg Annual Spend (AUD)"},
        color="Preferred_Channel",
        color_discrete_sequence=["#1B2A4A","#C8973A","#4A90A4"],
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**Finding:** In-store customers may have lower app engagement scores but comparable or higher
annual spend. **Low app engagement must not be treated as low customer value.**

- App engagement is used in this dashboard **only** to inform communication channel selection
  (e.g. whether to use app push notifications vs email vs in-store invitation).
- It is **not** used as a criterion for segment eligibility or campaign priority.
""")

    st.markdown("#### 2. Age is Not Used for Eligibility")
    st.markdown("""
- Age band is included in the dashboard for descriptive analysis only.
- **Age is not a criterion in any segment rule.** No customer is excluded from or prioritised
  for a campaign based on their age group.
- Age-band charts show observed spending patterns to help tailor communication tone,
  not to gate access to campaign benefits.
""")

    st.markdown("#### 3. App Engagement — Role Clarification")
    st.markdown("""
- App engagement score is available in this dataset but is **not used to determine segment membership**.
- Its only sanctioned use is **communication channel guidance**: customers with high app engagement
  may be more receptive to in-app messages; customers with low engagement should be reached
  via email, SMS or in-store channels instead.
""")

    st.markdown("#### 4. Correlation ≠ Causation")
    st.markdown("""
- Observed associations in this dataset (e.g. between spend level and purchase frequency)
  describe patterns in historical data only.
- **No causal claims are made.** The dashboard does not assert that loyalty membership
  *causes* higher spend, or that re-engagement campaigns *will* increase purchase frequency.
- Any campaign outcomes should be measured through controlled pilots before being attributed
  to specific interventions.
""")

st.divider()

# ════════════════════════════════════════════════════════════════════════════════
# SECTION — DATA LIMITATIONS
# ════════════════════════════════════════════════════════════════════════════════
with st.expander("⚠️ Data Limitations", expanded=False):
    st.markdown("""
### Data Limitations

This dashboard is built on a single dataset of 60 customers covering a 12-month spend window.
The following data are **absent from this dataset** and limit the conclusions that can be drawn:

| Missing Data | Implication |
|---|---|
| **Campaign response history** | Cannot predict which customers will respond to which offer type |
| **Profit margins by product** | High spend does not necessarily equal high profit; margin data is needed before allocating campaign budget |
| **Campaign costs** | ROI cannot be calculated without cost data; "high-value" targeting cannot be confirmed as cost-effective |
| **Customer acquisition costs** | Cannot assess whether re-engagement is cheaper than reacquisition without acquisition cost data |
| **Communication consent / permissions** | Regulatory compliance (e.g. spam laws, privacy legislation) must be verified before any outreach |
| **Longitudinal purchase history** | A single 12-month window is insufficient to establish behavioural trends or predict future spend with confidence |

**Sample size note:** 60 customers is a small sample. Segment counts (particularly Segment B at 9 customers)
are too small to draw statistically reliable conclusions. Findings should be treated as directional only.

**Recommended next steps before implementation:**
1. Obtain consent/permissions data before any outreach
2. Validate segment rules against a larger or more recent dataset
3. Run a small controlled pilot for each segment before broad rollout
4. Measure actual campaign response rates and margin impact
""")

st.divider()

# ── Data table ─────────────────────────────────────────────────────────────────
with st.expander("🗂️ View Customer Data", expanded=False):
    seg_filter = st.multiselect(
        "Filter by segment", preferred_order, default=preferred_order, key="tbl"
    )
    show_df = fdf[fdf["Segment"].isin(seg_filter)].reset_index(drop=True)
    st.dataframe(show_df.drop(columns=["_median_spend","_median_freq"]), use_container_width=True)
    st.caption(f"Showing {len(show_df)} customers")

st.caption(
    "Fashion Retailer Loyalty Dashboard · Streamlit & Plotly · "
    "Rule-based segmentation · 60 customers · 12-month window · "
    "Decision-support tool only — not for direct implementation without human review."
)
