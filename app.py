"""
app.py  –  Tech Layoffs Intelligence Dashboard
Run with:  streamlit run app.py

Auto-trains ML models on first launch if models/ directory is missing or incomplete.
This makes it fully deployable on Streamlit Cloud without pre-committing .pkl files.
"""

import os, re, warnings, uuid
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "tech_layoffs_hiring_trends_elite_v2.csv")
MODELS_DIR  = os.path.join(BASE, "models")

# ── Auto-train bootstrap ───────────────────────────────────────────────────────
def _models_ready() -> bool:
    required = [
        "hiring_trend_classifier.pkl", "layoff_count_regressor.pkl",
        "ai_risk_classifier.pkl",       "job_security_regressor.pkl",
        "kmeans_cluster.pkl",           "label_encoders.pkl",
        "feature_list.pkl",             "model_metrics.pkl",
    ]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required)

if not _models_ready():
    import subprocess, sys
    train_script = os.path.join(BASE, "train_models.py")
    with st.spinner("⏳ First launch detected — training ML models (takes ~60 seconds)…"):
        result = subprocess.run(
            [sys.executable, train_script],
            capture_output=True, text=True
        )
    if result.returncode != 0:
        st.error("Model training failed. Check train_models.py logs.")
        st.code(result.stderr)
        st.stop()
    st.success("✅ Models trained successfully! Loading dashboard…")
    st.rerun()

# ── Load everything once ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    def m(name): return joblib.load(os.path.join(MODELS_DIR, name))
    return {
        "encoders":         m("label_encoders.pkl"),
        "ht_clf":           m("hiring_trend_classifier.pkl"),
        "all_features":     m("feature_list.pkl"),
        "lc_reg":           m("layoff_count_regressor.pkl"),
        "reg_features":     m("reg_feature_list.pkl"),
        "risk_clf":         m("ai_risk_classifier.pkl"),
        "jss_reg":          m("job_security_regressor.pkl"),
        "jss_features":     m("jss_feature_list.pkl"),
        "km":               m("kmeans_cluster.pkl"),
        "scaler":           m("cluster_scaler.pkl"),
        "cluster_features": m("cluster_features.pkl"),
        "cluster_map":      m("cluster_map.pkl"),
        "metrics":          m("model_metrics.pkl"),
        "feat_imp":         pd.read_csv(
                                os.path.join(MODELS_DIR, "feature_importances.csv"),
                                index_col=0, header=0,
                                names=["feature", "importance"],
                                skiprows=1
                            ),
    }

df   = load_data()
mods = load_models()

# ── Column shortlists ──────────────────────────────────────────────────────────
NUM_COLS = [
    "layoffs_count", "layoff_percentage", "ai_automation_impact",
    "ai_replacement_risk", "open_roles", "remote_jobs_percentage",
    "stock_growth_percent", "revenue_growth_percent",
    "salary_budget_change", "ai_adoption_level",
    "employee_sentiment", "job_security_score",
]
CAT_COLS = [
    "industry", "country", "company_size", "hiring_trend",
    "reason_for_layoffs", "market_condition",
]

# ── Helper: encode a row dict into model features ──────────────────────────────
def encode_row(row_dict: dict, feature_list: list) -> np.ndarray:
    enc = mods["encoders"]
    cat_map = {
        "industry":          "industry_enc",
        "country":           "country_enc",
        "company_size":      "company_size_enc",
        "hiring_trend":      "hiring_trend_enc",
        "reason_for_layoffs":"reason_for_layoffs_enc",
        "market_condition":  "market_condition_enc",
        "month":             "month_enc",
        "top_hiring_role":   "top_hiring_role_enc",
    }
    row = dict(row_dict)
    for raw, encoded in cat_map.items():
        if raw in row:
            le = enc.get(raw)
            if le:
                try:
                    row[encoded] = le.transform([str(row[raw])])[0]
                except ValueError:
                    row[encoded] = 0
    return np.array([row.get(f, 0) for f in feature_list]).reshape(1, -1)

# ── Question-Answering Engine ──────────────────────────────────────────────────
INTENTS = {
    "layoffs_by_industry":  r"layoff.*(industry|sector)|industry.*layoff",
    "layoffs_by_country":   r"layoff.*(country|region|nation)|country.*layoff",
    "layoffs_trend":        r"layoff.*(trend|over time|year|month)|trend.*layoff",
    "top_companies":        r"top|most layoff|highest layoff|company.*layoff",
    "hiring_trend":         r"hiring.*(trend|pattern)|trend.*hiring",
    "ai_impact":            r"ai.*(impact|automation|risk|replace)|automation.*(impact|risk)",
    "remote_work":          r"remote|work from home|wfh",
    "market_condition":     r"market.*(condition|bull|recession)|bull|recession",
    "company_size":         r"company.?(size|type)|startup|enterprise|big tech|mid.?size",
    "salary_budget":        r"salary|budget|compensation|pay",
    "employee_sentiment":   r"sentiment|morale|employee.*(feel|opinion|satisfaction)",
    "job_security":         r"job security|secure|safe job",
    "reason_layoffs":       r"reason|why.*layoff|cause.*layoff|layoff.*reason",
    "stock_revenue":        r"stock|revenue|financial|growth",
    "predict_hiring":       r"predict|forecast|will|what.*hiring|hiring.*predict",
    "predict_layoff_count": r"predict.*layoff count|forecast.*layoff|how many.*layoff",
    "cluster_analysis":     r"cluster|group|segm|profile|categori",
    "feature_importance":   r"important|factor|driver|feature.*important",
    "overview":             r"overview|summary|summar|describe|what is|tell me about|show all",
}

def detect_intent(q: str) -> str:
    q_low = q.lower()
    for intent, pattern in INTENTS.items():
        if re.search(pattern, q_low):
            return intent
    return "overview"

def answer(question: str):
    """Returns (text_answer, fig_or_None)."""
    intent = detect_intent(question)
    text   = ""
    fig    = None

    # ── Layoffs by industry ──────────────────────────────────────────────────
    if intent == "layoffs_by_industry":
        grp = (df.groupby("industry")["layoffs_count"]
                 .sum().sort_values(ascending=False).reset_index())
        grp.columns = ["Industry", "Total Layoffs"]
        text = (
            "**Total layoffs by industry:**\n\n"
            + grp.to_markdown(index=False)
            + f"\n\n🔺 Hardest hit: **{grp.iloc[0]['Industry']}** "
            + f"with **{grp.iloc[0]['Total Layoffs']:,}** layoffs."
        )
        fig = px.bar(
            grp, x="Industry", y="Total Layoffs",
            color="Total Layoffs", color_continuous_scale="Reds",
            title="Total Layoffs by Industry", template="plotly_dark"
        )

    # ── Layoffs by country ───────────────────────────────────────────────────
    elif intent == "layoffs_by_country":
        grp = (df.groupby("country")["layoffs_count"]
                 .sum().sort_values(ascending=False).reset_index())
        grp.columns = ["Country", "Total Layoffs"]
        text = (
            "**Layoffs by country:**\n\n"
            + grp.to_markdown(index=False)
            + f"\n\n🌍 Most affected: **{grp.iloc[0]['Country']}**."
        )
        fig = px.choropleth(
            grp, locations="Country", locationmode="country names",
            color="Total Layoffs", color_continuous_scale="OrRd",
            title="Global Layoff Distribution", template="plotly_dark"
        )

    # ── Layoff trend over time ───────────────────────────────────────────────
    elif intent == "layoffs_trend":
        grp = df.groupby("year")["layoffs_count"].sum().reset_index()
        grp.columns = ["Year", "Total Layoffs"]
        text = "**Layoff trend across years:**\n\n" + grp.to_markdown(index=False)
        fig = px.line(
            grp, x="Year", y="Total Layoffs", markers=True,
            title="Layoff Trend Over Years", template="plotly_dark",
            line_shape="spline"
        )
        fig.update_traces(line_color="#ff4b4b", line_width=3)

    # ── Top companies ────────────────────────────────────────────────────────
    elif intent == "top_companies":
        top = df.nlargest(10, "layoffs_count")[
            ["company_name", "industry", "country", "layoffs_count", "year"]
        ]
        text = "**Top 10 companies with highest layoffs:**\n\n" + top.to_markdown(index=False)
        fig = px.bar(
            top.sort_values("layoffs_count"),
            x="layoffs_count", y="company_name", orientation="h",
            color="industry", title="Top 10 Companies by Layoff Count",
            template="plotly_dark",
            labels={"layoffs_count": "Layoffs", "company_name": "Company"}
        )

    # ── Hiring trends ────────────────────────────────────────────────────────
    elif intent == "hiring_trend":
        grp = df["hiring_trend"].value_counts().reset_index()
        grp.columns = ["Hiring Trend", "Count"]
        pct = grp.copy()
        pct["Percentage"] = (pct["Count"] / pct["Count"].sum() * 100).round(1)
        text = "**Hiring trend distribution:**\n\n" + pct.to_markdown(index=False)
        fig = px.pie(
            grp, names="Hiring Trend", values="Count",
            title="Distribution of Hiring Trends",
            template="plotly_dark", hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )

    # ── AI impact ───────────────────────────────────────────────────────────
    elif intent == "ai_impact":
        grp = (df.groupby("industry")
                 [["ai_automation_impact", "ai_replacement_risk", "ai_adoption_level"]]
                 .mean().round(2).reset_index())
        text = (
            "**AI impact metrics by industry (avg):**\n\n"
            + grp.to_markdown(index=False)
            + "\n\n_Higher scores = greater AI disruption._"
        )
        fig = px.bar(
            grp, x="industry",
            y=["ai_automation_impact", "ai_replacement_risk", "ai_adoption_level"],
            barmode="group", title="AI Impact by Industry", template="plotly_dark",
            labels={"value": "Score", "variable": "Metric", "industry": "Industry"}
        )

    # ── Remote work ─────────────────────────────────────────────────────────
    elif intent == "remote_work":
        grp = (df.groupby("industry")["remote_jobs_percentage"]
                 .mean().round(1).sort_values(ascending=False).reset_index())
        grp.columns = ["Industry", "Avg Remote %"]
        text = "**Average remote job percentage by industry:**\n\n" + grp.to_markdown(index=False)
        fig = px.bar(
            grp, x="Industry", y="Avg Remote %",
            color="Avg Remote %", color_continuous_scale="Blues",
            title="Remote Work Percentage by Industry", template="plotly_dark"
        )

    # ── Market condition ─────────────────────────────────────────────────────
    elif intent == "market_condition":
        grp = df.groupby("market_condition").agg(
            total_layoffs=("layoffs_count", "sum"),
            avg_hiring_companies=("open_roles", "mean"),
            avg_sentiment=("employee_sentiment", "mean")
        ).round(2).reset_index()
        text = "**Metrics by market condition:**\n\n" + grp.to_markdown(index=False)
        fig = px.bar(
            grp, x="market_condition", y="total_layoffs",
            color="market_condition", title="Total Layoffs by Market Condition",
            template="plotly_dark",
            labels={"total_layoffs": "Total Layoffs", "market_condition": "Market Condition"}
        )

    # ── Company size ─────────────────────────────────────────────────────────
    elif intent == "company_size":
        grp = df.groupby("company_size").agg(
            total_layoffs=("layoffs_count", "sum"),
            avg_layoff_pct=("layoff_percentage", "mean"),
            count=("record_id", "count")
        ).round(2).reset_index()
        text = "**Layoff analysis by company size:**\n\n" + grp.to_markdown(index=False)
        fig = px.scatter(
            df.sample(min(2000, len(df))),
            x="layoff_percentage", y="open_roles",
            color="company_size", size="layoffs_count",
            title="Layoff % vs Open Roles by Company Size",
            template="plotly_dark", opacity=0.6
        )

    # ── Salary / budget ──────────────────────────────────────────────────────
    elif intent == "salary_budget":
        grp = (df.groupby("industry")["salary_budget_change"]
                 .mean().round(2).sort_values().reset_index())
        grp.columns = ["Industry", "Avg Salary Budget Change (%)"]
        text = (
            "**Average salary budget change by industry:**\n\n"
            + grp.to_markdown(index=False)
            + "\n\n_Negative = budget cuts; Positive = increases._"
        )
        fig = px.bar(
            grp, x="Industry", y="Avg Salary Budget Change (%)",
            color="Avg Salary Budget Change (%)", color_continuous_scale="RdYlGn",
            title="Salary Budget Change by Industry", template="plotly_dark"
        )

    # ── Employee sentiment ───────────────────────────────────────────────────
    elif intent == "employee_sentiment":
        grp = (df.groupby("industry")["employee_sentiment"]
                 .mean().round(2).sort_values().reset_index())
        grp.columns = ["Industry", "Avg Sentiment Score"]
        text = (
            "**Average employee sentiment score by industry (0-10):**\n\n"
            + grp.to_markdown(index=False)
            + "\n\n_Higher = more positive; lower = distressed workforce._"
        )
        fig = px.bar(
            grp, x="Avg Sentiment Score", y="Industry", orientation="h",
            color="Avg Sentiment Score", color_continuous_scale="RdYlGn",
            title="Employee Sentiment by Industry", template="plotly_dark"
        )

    # ── Job security ─────────────────────────────────────────────────────────
    elif intent == "job_security":
        grp = (df.groupby("industry")["job_security_score"]
                 .mean().round(2).sort_values(ascending=False).reset_index())
        grp.columns = ["Industry", "Avg Job Security Score"]
        text = (
            "**Average job security score by industry (0-10):**\n\n"
            + grp.to_markdown(index=False)
        )
        fig = px.bar(
            grp, x="Industry", y="Avg Job Security Score",
            color="Avg Job Security Score", color_continuous_scale="Greens",
            title="Job Security Score by Industry", template="plotly_dark"
        )

    # ── Reason for layoffs ───────────────────────────────────────────────────
    elif intent == "reason_layoffs":
        grp = (df.groupby("reason_for_layoffs")["layoffs_count"]
                 .sum().sort_values(ascending=False).reset_index())
        grp.columns = ["Reason", "Total Layoffs"]
        text = "**Total layoffs by reason:**\n\n" + grp.to_markdown(index=False)
        fig = px.bar(
            grp, x="Total Layoffs", y="Reason", orientation="h",
            color="Total Layoffs", color_continuous_scale="Oranges",
            title="Layoff Reasons Analysis", template="plotly_dark"
        )

    # ── Stock / revenue ──────────────────────────────────────────────────────
    elif intent == "stock_revenue":
        corr_s = df["stock_growth_percent"].corr(df["layoffs_count"])
        corr_r = df["revenue_growth_percent"].corr(df["layoffs_count"])
        text = (
            f"**Financial indicators vs layoffs:**\n\n"
            f"- Correlation: stock growth ↔ layoffs = **{corr_s:.3f}**\n"
            f"- Correlation: revenue growth ↔ layoffs = **{corr_r:.3f}**\n\n"
            "_Negative correlation means growth reduces layoffs._"
        )
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Stock Growth vs Layoffs", "Revenue Growth vs Layoffs"]
        )
        sample = df.sample(min(1500, len(df)))
        fig.add_trace(
            go.Scatter(
                x=sample["stock_growth_percent"], y=sample["layoffs_count"],
                mode="markers", marker=dict(color="gold", opacity=0.4, size=4),
                name="Stock"
            ), row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=sample["revenue_growth_percent"], y=sample["layoffs_count"],
                mode="markers", marker=dict(color="cyan", opacity=0.4, size=4),
                name="Revenue"
            ), row=1, col=2
        )
        fig.update_layout(
            template="plotly_dark",
            title="Financial Growth vs Layoff Count",
            showlegend=False
        )

    # ── Feature importance ───────────────────────────────────────────────────
    elif intent == "feature_importance":
        fi = mods["feat_imp"].head(15).reset_index()
        fi.columns = ["Feature", "Importance"]
        text = (
            "**Top 15 features driving Hiring Trend prediction:**\n\n"
            + fi.to_markdown(index=False)
        )
        fig = px.bar(
            fi.sort_values("Importance"),
            x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Viridis",
            title="Top 15 Feature Importances", template="plotly_dark"
        )

    # ── Cluster analysis ─────────────────────────────────────────────────────
    elif intent == "cluster_analysis":
        cf     = mods["cluster_features"]
        scaler = mods["scaler"]
        km     = mods["km"]
        cmap   = mods["cluster_map"]
        X_c    = scaler.transform(df[cf])
        labels = km.predict(X_c)
        df_c   = df.copy()
        df_c["Cluster"] = [cmap.get(l, f"Cluster {l}") for l in labels]
        grp = df_c.groupby("Cluster")[
            ["layoffs_count", "ai_replacement_risk", "employee_sentiment", "job_security_score"]
        ].mean().round(2)
        text = (
            "**Company Risk Cluster Profiles (KMeans, k=4):**\n\n"
            + grp.to_markdown()
            + "\n\nClusters group companies by risk & stability profile."
        )
        fig = px.scatter(
            df_c.sample(min(2000, len(df_c))),
            x="ai_replacement_risk", y="layoffs_count",
            color="Cluster", size="open_roles",
            hover_data=["company_name", "industry"],
            title="Company Clusters: AI Risk vs Layoffs",
            template="plotly_dark", opacity=0.7
        )

    # ── ML: Predict hiring trend ─────────────────────────────────────────────
    elif intent == "predict_hiring":
        text = (
            "**Hiring Trend Predictor** (ML Model — RandomForest, Acc: 100%)\n\n"
            "Fill in the sidebar form and click **🎯 Predict Hiring Trend** "
            "to get the ML prediction for a custom company scenario."
        )
        st.session_state["show_predictor"] = "hiring"

    # ── ML: Predict layoff count ─────────────────────────────────────────────
    elif intent == "predict_layoff_count":
        text = (
            "**Layoff Count Predictor** (ML Model — RandomForest Regressor, R²: 0.55)\n\n"
            "Fill in the sidebar form and click **📉 Predict Layoff Count** "
            "to estimate expected layoffs for a company scenario."
        )
        st.session_state["show_predictor"] = "layoff"

    # ── Overview / default ───────────────────────────────────────────────────
    else:
        tot_lay = df["layoffs_count"].sum()
        avg_pct = df["layoff_percentage"].mean()
        top_ind = df.groupby("industry")["layoffs_count"].sum().idxmax()
        top_cty = df.groupby("country")["layoffs_count"].sum().idxmax()
        text = (
            f"**📊 Dataset Overview**\n\n"
            f"- **Total Records:** {len(df):,}\n"
            f"- **Total Layoffs:** {tot_lay:,}\n"
            f"- **Avg Layoff %:** {avg_pct:.1f}%\n"
            f"- **Most Affected Industry:** {top_ind}\n"
            f"- **Most Affected Country:** {top_cty}\n"
            f"- **Years Covered:** {df['year'].min()}–{df['year'].max()}\n"
            f"- **Industries:** {', '.join(df['industry'].unique())}\n"
            f"- **Countries:** {', '.join(df['country'].unique())}\n\n"
            "_Ask me specific questions about layoffs, hiring trends, AI impact, predictions, and more!_"
        )
        corr = df[NUM_COLS].corr().round(2)
        fig  = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
            template="plotly_dark", aspect="auto"
        )

    return text, fig


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Tech Layoffs Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"]          { background: #161b22; }
.chat-bubble-user {
    background: #1f6feb; color: white;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 16px; margin: 8px 0;
    max-width: 80%; margin-left: auto; text-align: right;
}
.chat-bubble-ai {
    background: #21262d; color: #e6edf3;
    border-radius: 2px 12px 12px 12px;
    padding: 12px 16px; margin: 8px 0;
    max-width: 85%; border-left: 3px solid #388bfd;
}
.metric-card {
    background: #161b22; border-radius: 10px; padding: 14px;
    border: 1px solid #30363d; text-align: center;
}
.stButton > button {
    background: #238636; color: white; border: none;
    border-radius: 8px; padding: 8px 20px; font-weight: 600;
}
.stButton > button:hover { background: #2ea043; }
h1, h2, h3 { color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Tech Layoffs Intelligence Dashboard")
st.markdown("*Ask anything about tech layoffs, hiring trends, AI impact, and predictions*")
st.divider()

# ── KPI row ────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(f"<div class='metric-card'><h3>{len(df):,}</h3><p>Records</p></div>",
                unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='metric-card'><h3>{df['layoffs_count'].sum():,}</h3><p>Total Layoffs</p></div>",
                unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='metric-card'><h3>{df['industry'].nunique()}</h3><p>Industries</p></div>",
                unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='metric-card'><h3>{df['country'].nunique()}</h3><p>Countries</p></div>",
                unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='metric-card'><h3>{df['year'].min()}–{df['year'].max()}</h3><p>Years</p></div>",
                unsafe_allow_html=True)

st.markdown("")

# ── Session state init ─────────────────────────────────────────────────────────
if "messages"       not in st.session_state:
    st.session_state.messages       = []
if "show_predictor" not in st.session_state:
    st.session_state.show_predictor = None

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔮 ML Predictor")
    st.markdown("---")

    industry_opts = sorted(df["industry"].unique())
    country_opts  = sorted(df["country"].unique())
    size_opts     = sorted(df["company_size"].unique())
    reason_opts   = sorted(df["reason_for_layoffs"].unique())
    market_opts   = sorted(df["market_condition"].unique())
    ht_opts       = sorted(df["hiring_trend"].unique())
    role_opts     = sorted(df["top_hiring_role"].unique())
    month_opts    = list(df["month"].unique())

    industry   = st.selectbox("Industry",        industry_opts)
    country    = st.selectbox("Country",         country_opts)
    size       = st.selectbox("Company Size",    size_opts)
    reason     = st.selectbox("Layoff Reason",   reason_opts)
    market     = st.selectbox("Market Condition",market_opts)
    top_role   = st.selectbox("Top Hiring Role", role_opts)
    month      = st.selectbox("Month",           month_opts)

    year       = st.slider("Year",                          2024,  2026,  2025)
    layoffs_c  = st.slider("Layoffs Count",                    0, 20000,  2000, step=100)
    layoff_pct = st.slider("Layoff %",                       0.0,  40.0,  10.0, step=0.5)
    ai_auto    = st.slider("AI Automation Impact (0-10)",    0.0,  10.0,   5.0, step=0.1)
    ai_risk    = st.slider("AI Replacement Risk (0-10)",     0.0,  10.0,   5.0, step=0.1)
    open_roles = st.slider("Open Roles",                      50, 10000,  2000, step=50)
    remote_pct = st.slider("Remote Jobs %",                 10.0,  90.0,  50.0, step=1.0)
    stock_g    = st.slider("Stock Growth %",               -45.0,  90.0,  10.0, step=0.5)
    rev_g      = st.slider("Revenue Growth %",             -25.0,  60.0,  10.0, step=0.5)
    sal_b      = st.slider("Salary Budget Change %",       -28.0,  38.0,   5.0, step=0.5)
    ai_adop    = st.slider("AI Adoption Level (0-10)",       0.0,  10.0,   5.0, step=0.1)
    emp_sent   = st.slider("Employee Sentiment (0-10)",      1.0,  10.0,   6.5, step=0.1)
    job_sec    = st.slider("Job Security Score (0-10)",      1.0,  10.0,   5.5, step=0.1)

    row_dict = {
        "year": year, "layoffs_count": layoffs_c, "layoff_percentage": layoff_pct,
        "ai_automation_impact": ai_auto, "ai_replacement_risk": ai_risk,
        "open_roles": open_roles, "remote_jobs_percentage": remote_pct,
        "stock_growth_percent": stock_g, "revenue_growth_percent": rev_g,
        "salary_budget_change": sal_b, "ai_adoption_level": ai_adop,
        "employee_sentiment": emp_sent, "job_security_score": job_sec,
        "industry": industry, "country": country, "company_size": size,
        "reason_for_layoffs": reason, "market_condition": market,
        "hiring_trend": ht_opts[0], "top_hiring_role": top_role, "month": month,
    }

    st.markdown("---")
    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        if st.button("🎯 Predict\nHiring Trend"):
            X     = encode_row(row_dict, mods["all_features"])
            pred  = mods["ht_clf"].predict(X)[0]
            proba = mods["ht_clf"].predict_proba(X)[0]
            le    = mods["encoders"]["hiring_trend"]
            label = le.inverse_transform([pred])[0]
            conf  = round(max(proba) * 100, 1)
            st.success(f"**{label}**\n\nConfidence: {conf}%")

    with pred_col2:
        if st.button("📉 Predict\nLayoff Count"):
            X    = encode_row(row_dict, mods["reg_features"])
            pred = mods["lc_reg"].predict(X)[0]
            st.info(f"**~{int(pred):,}** layoffs")

    st.markdown("---")

    if st.button("🤖 Predict AI Risk Level"):
        X     = encode_row(row_dict, mods["all_features"])
        pred  = mods["risk_clf"].predict(X)[0]
        le    = mods["encoders"]["risk_category"]
        label = le.inverse_transform([pred])[0]
        color = {"High": "error", "Medium": "warning", "Low": "success"}.get(label, "info")
        getattr(st, color)(f"AI Replacement Risk: **{label}**")

    if st.button("🔒 Predict Job Security"):
        X    = encode_row(row_dict, mods["jss_features"])
        pred = mods["jss_reg"].predict(X)[0]
        st.info(f"Job Security Score: **{pred:.2f} / 10**")

    st.markdown("---")
    st.markdown("**Model Performance**")
    metrics = mods["metrics"]
    for mname, mval in metrics.items():
        short = mname.replace("_", " ").title()
        if mval["type"] == "classification":
            st.caption(f"✅ {short}: Acc={mval['accuracy']}")
        elif mval["type"] == "regression":
            st.caption(f"📈 {short}: R²={mval['r2']}")
        else:
            st.caption(f"🔵 {short}: Sil={mval.get('silhouette', 'N/A')}")

# ── Suggested questions ────────────────────────────────────────────────────────
st.markdown("#### 💡 Suggested Questions")
suggestions = [
    "Show layoffs by industry",
    "Which countries have most layoffs?",
    "What is the AI impact on tech jobs?",
    "Show hiring trend distribution",
    "What are the top reasons for layoffs?",
    "Show employee sentiment by industry",
    "Show feature importances",
    "Cluster analysis of companies",
]
sugg_cols = st.columns(4)
for idx, sugg in enumerate(suggestions):
    with sugg_cols[idx % 4]:
        if st.button(sugg, key=f"sugg_{idx}"):
            # ✅ Build and store the message with a permanent chart key
            st.session_state.messages.append({"role": "user", "content": sugg})
            ans_text, ans_fig = answer(sugg)
            st.session_state.messages.append({
                "role":      "assistant",
                "content":   ans_text,
                "fig":       ans_fig,
                # ✅ Key generated ONCE and stored permanently with the message
                "fig_key":   f"chart_{uuid.uuid4()}" if ans_fig is not None else None,
            })
            st.rerun()   # ✅ Refresh so the new message appears immediately

st.markdown("---")

# ── Chat history ───────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-bubble-user'>🧑 {msg['content']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='chat-bubble-ai'>🤖 {msg['content']}</div>",
            unsafe_allow_html=True
        )
        # ✅ Only render chart if fig exists AND has a stored key
        if msg.get("fig") is not None and msg.get("fig_key"):
            st.plotly_chart(
                msg["fig"],
                use_container_width=True,
                key=msg["fig_key"]   # ✅ stable, unique, stored with message
            )

# ── Chat input form ────────────────────────────────────────────────────────────
st.markdown("---")
with st.form("chat_form", clear_on_submit=True):
    cols = st.columns([5, 1])
    with cols[0]:
        user_input = st.text_input(
            "Ask a question about the data…",
            label_visibility="collapsed",
            placeholder="e.g. Which industry has the most layoffs in 2025?"
        )
    with cols[1]:
        submitted = st.form_submit_button("Send 🚀")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    ans_text, ans_fig = answer(user_input)
    st.session_state.messages.append({
        "role":    "assistant",
        "content": ans_text,
        "fig":     ans_fig,
        # ✅ Key generated ONCE and stored permanently with the message
        "fig_key": f"chart_{uuid.uuid4()}" if ans_fig is not None else None,
    })
    st.rerun()

# ── Clear chat ─────────────────────────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()