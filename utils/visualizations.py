"""
utils/visualizations.py
Reusable Plotly chart factories for the Tech Layoffs dashboard.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

TEMPLATE = "plotly_dark"


def bar_layoffs_by_group(df: pd.DataFrame, group_col: str, title: str) -> go.Figure:
    grp = (
        df.groupby(group_col)["layoffs_count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    grp.columns = [group_col, "Total Layoffs"]
    return px.bar(
        grp, x=group_col, y="Total Layoffs",
        color="Total Layoffs", color_continuous_scale="Reds",
        title=title, template=TEMPLATE,
    )


def choropleth_countries(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby("country")["layoffs_count"]
        .sum()
        .reset_index()
    )
    grp.columns = ["Country", "Total Layoffs"]
    return px.choropleth(
        grp, locations="Country", locationmode="country names",
        color="Total Layoffs", color_continuous_scale="OrRd",
        title="Global Layoff Distribution", template=TEMPLATE,
    )


def line_trend(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    grp = df.groupby(x_col)[y_col].sum().reset_index()
    fig = px.line(grp, x=x_col, y=y_col, markers=True, title=title,
                  template=TEMPLATE, line_shape="spline")
    fig.update_traces(line_color="#ff4b4b", line_width=3)
    return fig


def pie_distribution(df: pd.DataFrame, col: str, title: str) -> go.Figure:
    grp = df[col].value_counts().reset_index()
    grp.columns = [col, "Count"]
    return px.pie(
        grp, names=col, values="Count", title=title,
        template=TEMPLATE, hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3,
    )


def grouped_bar_ai(df: pd.DataFrame) -> go.Figure:
    grp = (
        df.groupby("industry")[
            ["ai_automation_impact", "ai_replacement_risk", "ai_adoption_level"]
        ]
        .mean()
        .round(2)
        .reset_index()
    )
    return px.bar(
        grp, x="industry",
        y=["ai_automation_impact", "ai_replacement_risk", "ai_adoption_level"],
        barmode="group", title="AI Impact by Industry", template=TEMPLATE,
        labels={"value": "Score", "variable": "Metric", "industry": "Industry"},
    )


def scatter_cluster(df_with_cluster: pd.DataFrame) -> go.Figure:
    return px.scatter(
        df_with_cluster.sample(min(2000, len(df_with_cluster))),
        x="ai_replacement_risk", y="layoffs_count",
        color="Cluster", size="open_roles",
        hover_data=["company_name", "industry"],
        title="Company Clusters: AI Risk vs Layoffs",
        template=TEMPLATE, opacity=0.7,
    )


def heatmap_correlation(df: pd.DataFrame, num_cols: list) -> go.Figure:
    corr = df[num_cols].corr().round(2)
    return px.imshow(
        corr, text_auto=True, color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap", template=TEMPLATE, aspect="auto",
    )


def horizontal_bar(df_grp: pd.DataFrame, x_col: str, y_col: str,
                   title: str, color_scale: str = "Oranges") -> go.Figure:
    return px.bar(
        df_grp, x=x_col, y=y_col, orientation="h",
        color=x_col, color_continuous_scale=color_scale,
        title=title, template=TEMPLATE,
    )


def financial_scatter(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Stock Growth vs Layoffs", "Revenue Growth vs Layoffs"],
    )
    sample = df.sample(min(1500, len(df)))
    fig.add_trace(
        go.Scatter(
            x=sample["stock_growth_percent"], y=sample["layoffs_count"],
            mode="markers", marker=dict(color="gold", opacity=0.4, size=4),
            name="Stock",
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sample["revenue_growth_percent"], y=sample["layoffs_count"],
            mode="markers", marker=dict(color="cyan", opacity=0.4, size=4),
            name="Revenue",
        ),
        row=1, col=2,
    )
    fig.update_layout(
        template=TEMPLATE, title="Financial Growth vs Layoff Count", showlegend=False,
    )
    return fig
