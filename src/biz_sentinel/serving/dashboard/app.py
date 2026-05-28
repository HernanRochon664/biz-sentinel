"""BizSentinel Dash Dashboard.

Reads scored data from parquet files and displays three views:
- Overview: KPI cards (anomaly rate, high-risk customers, top segment)
- Anomalies: table of flagged customers with scores
- Segments: bar chart of segment distribution with profile table
"""

import os

import pandas as pd
import plotly.express as px  # type: ignore[import-untyped]
from dash import Dash, Input, Output, callback, dash_table, dcc, html

# --- Data paths (from env with defaults) ---
ANOMALY_SCORES_PATH = os.getenv(
    "ANOMALY_SCORES_PATH", "data/07_model_output/anomaly_scores.parquet"
)
CHURN_SCORES_PATH = os.getenv("CHURN_SCORES_PATH", "data/07_model_output/churn_scores.parquet")
SEGMENT_PROFILES_PATH = os.getenv(
    "SEGMENT_PROFILES_PATH", "data/07_model_output/segment_profiles.parquet"
)
SEGMENT_ASSIGNMENTS_PATH = os.getenv(
    "SEGMENT_ASSIGNMENTS_PATH", "data/07_model_output/segment_assignments.parquet"
)

# --- Data loading functions ---


def load_parquet_safe(path: str) -> pd.DataFrame:
    """Load parquet file, return empty DataFrame if file not found."""
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        return pd.DataFrame()


# --- Color scheme ---
COLORS = {
    "primary": "#2C3E50",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "success": "#27AE60",
    "background": "#F8F9FA",
    "card": "#FFFFFF",
    "border": "#DEE2E6",
}


# --- KPI Card component ---
def kpi_card(title: str, value: str, color: str = COLORS["primary"]) -> html.Div:
    return html.Div(
        [
            html.H4(title, style={"color": "#6C757D", "fontSize": "14px", "marginBottom": "8px"}),
            html.H2(value, style={"color": color, "fontSize": "32px", "margin": "0"}),
        ],
        style={
            "background": COLORS["card"],
            "border": f"1px solid {COLORS['border']}",
            "borderRadius": "8px",
            "padding": "20px",
            "flex": "1",
            "minWidth": "200px",
        },
    )


# --- App layout ---
app = Dash(__name__, title="BizSentinel")

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>body{margin:0}</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = html.Div(
    [
        # Header
        html.Div(
            [
                html.A(
                    "← Back",
                    href="http://localhost:8055",
                    style={
                        "color": "#6C757D",
                        "textDecoration": "none",
                        "fontSize": "14px",
                        "marginRight": "16px",
                    },
                ),
                html.H1("BizSentinel", style={"color": COLORS["primary"], "margin": "0 0 12px 0"}),
                html.P(
                    "Business Intelligence & Anomaly Detection",
                    style={"color": "#6C757D", "margin": "4px 0 0 16px"},
                ),
                html.Button(
                    "Refresh Data",
                    id="refresh-btn",
                    n_clicks=0,
                    style={
                        "marginLeft": "auto",
                        "padding": "8px 16px",
                        "background": COLORS["primary"],
                        "color": "white",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "padding": "20px 32px",
                "background": COLORS["card"],
                "borderBottom": f"1px solid {COLORS['border']}",
            },
        ),
        # Tabs
        dcc.Tabs(  # type: ignore[attr-defined]
            id="tabs",
            value="overview",
            children=[
                dcc.Tab(label="Overview", value="overview"),  # type: ignore[attr-defined]
                dcc.Tab(label="Anomalies", value="anomalies"),  # type: ignore[attr-defined]
                dcc.Tab(label="Segments", value="segments"),  # type: ignore[attr-defined]
            ],
            style={"margin": "0 32px"},
        ),
        # Tab content
        html.Div(id="tab-content", style={"padding": "24px 32px"}),
    ],
    style={
        "background": COLORS["background"],
        "minHeight": "100vh",
        "fontFamily": "system-ui, -apple-system, sans-serif",
    },
)


# --- Callbacks ---


@callback(
    Output("tab-content", "children"),
    Input("tabs", "value"),
    Input("refresh-btn", "n_clicks"),
)
def render_tab(tab: str, _n_clicks: int) -> html.Div | html.P:
    """Render content for selected tab. Re-loads data on refresh."""

    anomaly_df = load_parquet_safe(ANOMALY_SCORES_PATH)
    churn_df = load_parquet_safe(CHURN_SCORES_PATH)
    segment_profiles_df = load_parquet_safe(SEGMENT_PROFILES_PATH)
    segment_df = load_parquet_safe(SEGMENT_ASSIGNMENTS_PATH)

    if tab == "overview":
        # KPI calculations
        n_anomalous = (
            int((anomaly_df["anomaly_flag"] == "anomalous").sum()) if not anomaly_df.empty else 0
        )
        n_high_risk = int((churn_df["churn_probability"] >= 0.6).sum()) if not churn_df.empty else 0
        anomaly_rate = f"{n_anomalous / len(anomaly_df):.1%}" if not anomaly_df.empty else "N/A"
        top_segment = segment_df["segment_label"].mode()[0] if not segment_df.empty else "N/A"

        return html.Div(
            [
                html.H3("Overview", style={"margin": "0 0 20px 0"}),
                html.Div(
                    [
                        kpi_card("Anomaly Rate", anomaly_rate, COLORS["danger"]),
                        kpi_card("High-Risk Customers", str(n_high_risk), COLORS["warning"]),
                        kpi_card("Flagged Anomalies", str(n_anomalous), COLORS["danger"]),
                        kpi_card("Largest Segment", top_segment, COLORS["success"]),
                    ],
                    style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                ),
            ]
        )

    elif tab == "anomalies":
        if anomaly_df.empty:
            return html.P("No anomaly data available. Run the training pipeline first.")

        flagged = anomaly_df[anomaly_df["anomaly_flag"] != "normal"].copy()
        flagged = flagged.sort_values("anomaly_score", ascending=False)  # type: ignore[call-overload]
        flagged["anomaly_score"] = flagged["anomaly_score"].round(4)

        # Merge churn data if available
        if not churn_df.empty:
            flagged = flagged.merge(
                churn_df[["customer_hash", "churn_probability"]], on="customer_hash", how="left"
            )
            flagged["churn_probability"] = flagged["churn_probability"].round(4)

        display_cols = [
            c
            for c in ["customer_hash", "anomaly_score", "anomaly_flag", "churn_probability"]
            if c in flagged.columns
        ]

        return html.Div(
            [
                html.H3(
                    f"Flagged Customers ({len(flagged)} total)", style={"marginBottom": "16px"}
                ),
                dash_table.DataTable(  # type: ignore[attr-defined]
                    data=flagged[display_cols].to_dict("records"),  # type: ignore[call-overload]
                    columns=[{"name": c.replace("_", " ").title(), "id": c} for c in display_cols],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px"},
                    style_header={
                        "background": COLORS["primary"],
                        "color": "white",
                        "fontWeight": "bold",
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{anomaly_flag} = anomalous"},
                            "backgroundColor": "#FFE8E8",
                        }
                    ],
                    page_size=20,
                ),
            ]
        )

    elif tab == "segments":
        if segment_profiles_df.empty:
            return html.P("No segment data available. Run the training pipeline first.")

        fig = px.bar(
            segment_profiles_df,
            x="segment_label",
            y="count" if "count" in segment_profiles_df.columns else segment_profiles_df.columns[1],
            color="segment_label",
            title="Customers per Segment",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white")

        display_cols = [c for c in segment_profiles_df.columns if c != "segment_label"]
        rounded = segment_profiles_df.copy()
        for col in display_cols:
            if rounded[col].dtype in ["float64", "float32"]:
                rounded[col] = rounded[col].round(2)

        return html.Div(
            [
                html.H3("Customer Segments", style={"marginBottom": "16px"}),
                dcc.Graph(figure=fig, style={"marginBottom": "24px"}),  # type: ignore[attr-defined]
                html.H4("Segment Profiles"),
                dash_table.DataTable(  # type: ignore[attr-defined]
                    data=rounded.to_dict("records"),  # type: ignore[call-overload]
                    columns=[
                        {"name": c.replace("_", " ").title(), "id": c}
                        for c in segment_profiles_df.columns
                    ],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "left", "padding": "8px"},
                    style_header={
                        "background": COLORS["primary"],
                        "color": "white",
                        "fontWeight": "bold",
                    },
                ),
            ]
        )

    return html.P("Select a tab")


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8050"))
    debug = os.getenv("ENVIRONMENT", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
