"""BizSentinel Landing Page.

Entry point for the BizSentinel system. Provides access to the Visual Dashboard
(port 8050) and the AI Assistant (port 8060).
"""

import os

from dash import Dash, html

DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:8050")
CHAT_URL = os.getenv("CHAT_URL", "http://localhost:8060")
CHAT_ENABLED = os.getenv("CHAT_ENABLED", "true").lower() == "true"

# --- Color scheme (matches app.py) ---
COLORS = {
    "primary": "#2C3E50",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "success": "#27AE60",
    "background": "#F8F9FA",
    "card": "#FFFFFF",
    "border": "#DEE2E6",
}

# --- Shared styles ---
FONT_FAMILY = "system-ui, -apple-system, sans-serif"

CARD_STYLE = {
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "padding": "32px",
    "flex": "1",
    "minWidth": "300px",
    "maxWidth": "480px",
    "display": "flex",
    "flexDirection": "column",
}

BUTTON_BASE: dict = {
    "display": "inline-block",
    "padding": "12px 24px",
    "borderRadius": "6px",
    "textDecoration": "none",
    "fontWeight": "600",
    "fontSize": "16px",
    "textAlign": "center",
    "cursor": "pointer",
    "marginTop": "auto",
    "width": "100%",
    "boxSizing": "border-box",
}

# --- App ---
app = Dash(__name__, title="BizSentinel")

app.layout = html.Div(
    [
        # -- Mobile viewport --
        html.Meta(name="viewport", content="width=device-width, initial-scale=1"),
        # -- Header --
        html.Div(
            [
                html.H1(
                    "BIZSENTINEL",
                    style={
                        "color": COLORS["primary"],
                        "fontSize": "clamp(32px, 6vw, 52px)",
                        "fontWeight": "800",
                        "letterSpacing": "4px",
                        "margin": "0",
                    },
                ),
                html.P(
                    "Business Intelligence & Anomaly Detection for E-commerce",
                    style={
                        "color": COLORS["primary"],
                        "fontSize": "clamp(14px, 2.5vw, 18px)",
                        "margin": "8px 0 4px",
                        "opacity": "0.8",
                    },
                ),
                html.P(
                    "97,896 customers analyzed · 3 ML models · Real-time insights",
                    style={
                        "color": "#6C757D",
                        "fontSize": "clamp(12px, 2vw, 14px)",
                        "margin": "4px 0 0",
                    },
                ),
            ],
            style={
                "textAlign": "center",
                "padding": "48px 24px 32px",
                "borderBottom": f"1px solid {COLORS['border']}",
            },
        ),
        # -- Cards --
        html.Div(
            [
                # Card 1 - Visual Dashboard
                html.Div(
                    [
                        html.Div(
                            "📊",
                            style={
                                "fontSize": "48px",
                                "marginBottom": "12px",
                            },
                        ),
                        html.H2(
                            "Visual Dashboard",
                            style={
                                "color": COLORS["primary"],
                                "margin": "0 0 8px",
                                "fontSize": "22px",
                            },
                        ),
                        html.P(
                            "Explore anomalies, customer segments, and churn risk "
                            "through interactive charts and tables. Best for data "
                            "exploration and monitoring.",
                            style={
                                "color": "#6C757D",
                                "fontSize": "14px",
                                "lineHeight": "1.6",
                                "margin": "0 0 20px",
                            },
                        ),
                        html.Ul(
                            [
                                html.Li("Anomaly detection overview"),
                                html.Li("Customer segment profiles"),
                                html.Li("Churn risk rankings"),
                                html.Li("Alert management"),
                            ],
                            style={
                                "color": "#6C757D",
                                "fontSize": "14px",
                                "lineHeight": "1.8",
                                "paddingLeft": "20px",
                                "margin": "0 0 24px",
                            },
                        ),
                        html.A(
                            "Open Dashboard",
                            href=DASHBOARD_URL,
                            target="_blank",
                            rel="noopener noreferrer",
                            style={
                                **BUTTON_BASE,
                                "background": COLORS["primary"],
                                "color": "white",
                                "border": "none",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
                # Card 2 - AI Assistant
                html.Div(
                    [
                        html.Div(
                            "🤖",
                            style={
                                "fontSize": "48px",
                                "marginBottom": "12px",
                            },
                        ),
                        html.H2(
                            "AI Assistant",
                            style={
                                "color": COLORS["primary"],
                                "margin": "0 0 8px",
                                "fontSize": "22px",
                            },
                        ),
                        html.P(
                            "Ask questions in natural language. The assistant uses "
                            "local AI (Ollama + Qwen) to query the system and explain "
                            "insights. Best for quick answers and recommendations.",
                            style={
                                "color": "#6C757D",
                                "fontSize": "14px",
                                "lineHeight": "1.6",
                                "margin": "0 0 20px",
                            },
                        ),
                        html.Ul(
                            [
                                html.Li("Natural language queries"),
                                html.Li("Automated tool calling"),
                                html.Li("Business recommendations"),
                                html.Li("Runs 100% locally (no cloud)"),
                            ],
                            style={
                                "color": "#6C757D",
                                "fontSize": "14px",
                                "lineHeight": "1.8",
                                "paddingLeft": "20px",
                                "margin": "0 0 24px",
                            },
                        ),
                        html.A(
                            "Start Chat",
                            href=CHAT_URL,
                            target="_blank",
                            rel="noopener noreferrer",
                            style={
                                **BUTTON_BASE,
                                "background": COLORS["success"],
                                "color": "white",
                                "border": "none",
                            },
                        ) if CHAT_ENABLED else html.Button(
                            "Start Chat",
                            disabled=True,
                            style={
                                "width": "100%",
                                "padding": "14px",
                                "background": "#95a5a6",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "6px",
                                "fontSize": "16px",
                                "cursor": "not-allowed",
                                "marginTop": "16px",
                            },
                        ),
                        html.P(
                            "AI Chat runs locally only. See README for setup instructions.",
                            style={
                                "color": "#6C757D",
                                "fontSize": "12px",
                                "margin": "12px 0 0",
                                "textAlign": "center",
                            },
                        ),
                    ],
                    style=CARD_STYLE,
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "center",
                "alignItems": "stretch",
                "gap": "24px",
                "padding": "32px 24px",
                "flexWrap": "wrap",
            },
        ),
        # -- Footer --
        html.Div(
            [
                html.P(
                    "BizSentinel v0.1.0 · Built with Kedro, MLflow, Prefect, FastMCP",
                    style={"margin": "0 0 8px", "fontSize": "13px"},
                ),
                html.A(
                    "View on GitHub →",
                    href="https://github.com/HernanRochon664/biz-sentinel",
                    target="_blank",
                    rel="noopener noreferrer",
                    style={
                        "color": COLORS["primary"],
                        "textDecoration": "none",
                        "fontSize": "13px",
                        "fontWeight": "600",
                    },
                ),
            ],
            style={
                "textAlign": "center",
                "padding": "24px",
                "color": "#6C757D",
                "borderTop": f"1px solid {COLORS['border']}",
                "fontSize": "13px",
            },
        ),
    ],
    style={
        "background": COLORS["background"],
        "minHeight": "100vh",
        "fontFamily": FONT_FAMILY,
        "margin": "0",
    },
)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8055"))
    debug = os.getenv("ENVIRONMENT", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
