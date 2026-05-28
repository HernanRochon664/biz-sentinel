"""BizSentinel AI Chat — Dash web UI for Ollama-based Q&A with tool calling.

Runs standalone on port 8060. Requires Ollama running locally with gemma4:e4b.
Reuses the exact tool-calling logic from biz_sentinel.scripts.ollama_chat.
"""

import json
import os
import re

import dash
import ollama
from dash import Dash, Input, Output, State, callback, dcc, html

from biz_sentinel.scripts.ollama_chat import TOOLS, call_tool

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are BizSentinel, a business intelligence assistant.
You have access to tools that query real customer data.

STRICT RULES:
- To call a tool, output ONLY a JSON object: {"name": "tool_name", "arguments": {...}}
- After receiving a [TOOL RESULT], you MUST respond in plain conversational Spanish.
- NEVER output JSON after receiving a [TOOL RESULT].
- NEVER make up data. ONLY use data from tool results.
- Keep responses concise and business-focused.
- If you already have the data, answer directly without calling tools again.

Available tools:
- get_anomaly_summary(days): Returns anomaly statistics
- get_customer_risk(customer_hash): Returns risk profile for a customer
- get_segment_profile(segment_label): Returns segment statistics
- explain_alert(alert_id): Returns alert explanation
"""

# ---------------------------------------------------------------------------
# Color scheme (matches app.py / landing.py)
# ---------------------------------------------------------------------------

COLORS = {
    "primary": "#2C3E50",
    "danger": "#E74C3C",
    "warning": "#F39C12",
    "success": "#27AE60",
    "background": "#F8F9FA",
    "card": "#FFFFFF",
    "border": "#DEE2E6",
}

FONT_FAMILY = "system-ui, -apple-system, sans-serif"
MODEL = "gemma4:e4b"

INITIAL_MESSAGE = (
    "Hola, soy el asistente de BizSentinel. Puedo ayudarte a analizar "
    "anomalías, perfiles de clientes y segmentos. ¿Qué querés saber?"
)

# ---------------------------------------------------------------------------
# Bubble renderers
# ---------------------------------------------------------------------------


def _user_bubble(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "alignSelf": "flex-end",
            "background": COLORS["primary"],
            "color": "white",
            "padding": "10px 16px",
            "borderRadius": "16px 16px 4px 16px",
            "maxWidth": "75%",
            "whiteSpace": "pre-wrap",
            "fontSize": "14px",
            "lineHeight": "1.5",
        },
    )


def _assistant_bubble(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "alignSelf": "flex-start",
            "background": "#EBF5FB",
            "color": COLORS["primary"],
            "padding": "10px 16px",
            "borderRadius": "16px 16px 16px 4px",
            "maxWidth": "75%",
            "border": "1px solid #BBDEFB",
            "whiteSpace": "pre-wrap",
            "fontSize": "14px",
            "lineHeight": "1.5",
        },
    )


def _tool_call_bubble(name: str) -> html.Div:
    return html.Div(
        f"[Calling tool: {name}]",
        style={
            "alignSelf": "flex-start",
            "color": "#6C757D",
            "fontSize": "12px",
            "fontStyle": "italic",
            "padding": "4px 8px",
        },
    )


def _system_bubble(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "alignSelf": "center",
            "background": "#FFF3CD",
            "color": "#856404",
            "padding": "8px 16px",
            "borderRadius": "8px",
            "maxWidth": "80%",
            "fontSize": "13px",
            "lineHeight": "1.4",
            "textAlign": "center",
        },
    )


def render_messages(messages: list[dict]) -> list[html.Div]:
    """Convert the message store into a list of Dash bubble components."""
    children: list[html.Div] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue

        if role == "user":
            children.append(_user_bubble(content))
            continue

        if role == "assistant":
            has_native_tc = bool(msg.get("tool_calls"))

            # Check for JSON-format tool call in content (qwen fallback path)
            is_json_tc = False
            json_tc_name: str | None = None
            if not has_native_tc and content and content.strip().startswith("{"):
                try:
                    cleaned = re.sub(r"```json\n?|\n?```", "", content).strip()
                    parsed = json.loads(cleaned)
                    if "name" in parsed and "arguments" in parsed:
                        is_json_tc = True
                        json_tc_name = parsed["name"]
                except (json.JSONDecodeError, KeyError):
                    pass

            if has_native_tc:
                for tc in msg["tool_calls"]:
                    children.append(_tool_call_bubble(tc["function"]["name"]))
                if content:
                    children.append(_assistant_bubble(content))
            elif is_json_tc:
                children.append(_tool_call_bubble(json_tc_name))
            else:
                children.append(_assistant_bubble(content))

            continue

        if role == "tool":
            continue

    return children


# ---------------------------------------------------------------------------
# Ollama call + tool execution (same logic as ollama_chat.py)
# ---------------------------------------------------------------------------


def _sanitize_final_response(content: str) -> str:
    """Strip pure-JSON responses that leak from the model as a safety net."""
    stripped = content.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, dict) and ("name" in parsed or "arguments" in parsed):
                return "Entendido. ¿Hay algo más en lo que pueda ayudarte?"
        except json.JSONDecodeError:
            pass
    return content


def _process_ollama_response(response, messages: list[dict]) -> list[dict]:
    """Process an Ollama response, execute any tool calls, return updated messages.
    Mutates *messages* in place."""
    msg = response.message

    # --- Path 1: native tool_calls (works with llama3.1, etc.) ---
    if msg.tool_calls:
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        )
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = (
                tc.function.arguments
                if isinstance(tc.function.arguments, dict)
                else json.loads(tc.function.arguments)
            )
            result = call_tool(fn_name, fn_args)
            messages.append(
                {
                    'role': 'user',
                    'content': (
                        f"[TOOL RESULT from {fn_name}]:\n"
                        f"{json.dumps(result, indent=2, ensure_ascii=False)}\n\n"
                        "Now respond in plain Spanish based on this data."
                    ),
                }
            )
        messages.append(
            {
                'role': 'user',
                'content': (
                    'Responde en español natural y conversacional. '
                    'No uses JSON.'
                ),
            }
        )
        final = ollama.chat(model=MODEL, messages=messages)
        messages.pop()
        messages.append(
            {"role": "assistant", "content": _sanitize_final_response(final.message.content)}
        )

    # --- Path 2: text-based JSON tool call (qwen2.5-coder fallback) ---
    elif msg.content and msg.content.strip().startswith("{"):
        try:
            cleaned = re.sub(r"```json\n?|\n?```", "", msg.content).strip()
            parsed = json.loads(cleaned)
            if "name" in parsed and "arguments" in parsed:
                fn_name = parsed["name"]
                fn_args = parsed["arguments"]
                messages.append({"role": "assistant", "content": msg.content})
                result = call_tool(fn_name, fn_args)
                messages.append({"role": "assistant", "content": "Consultando datos..."})
                messages.append(
                    {
                        'role': 'user',
                        'content': (
                            f"[TOOL RESULT from {fn_name}]:\n"
                            f"{json.dumps(result, indent=2, ensure_ascii=False)}\n\n"
                            "Now respond in plain Spanish based on this data."
                        ),
                    }
                )
                messages.append(
                    {
                        'role': 'user',
                        'content': (
                            'Responde en español natural y conversacional. '
                            'No uses JSON.'
                        ),
                    }
                )
                final = ollama.chat(model=MODEL, messages=messages)
                messages.pop()
                safe = _sanitize_final_response(final.message.content)
                messages.append({"role": "assistant", "content": safe})
            else:
                messages.append({"role": "assistant", "content": msg.content})
        except (json.JSONDecodeError, KeyError):
            messages.append({"role": "assistant", "content": msg.content})

    # --- Path 3: plain text response ---
    else:
        messages.append({"role": "assistant", "content": msg.content})

    return messages


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

app = Dash(__name__, title="BizSentinel AI Chat")

_INITIAL_STORE = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "assistant", "content": INITIAL_MESSAGE},
]

app.layout = html.Div(
    [
        dcc.Store(id="messages-store", data=_INITIAL_STORE),  # type: ignore[reportPrivateImportUsage]
        html.Meta(name="viewport", content="width=device-width, initial-scale=1"),
        # -- Header --
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
                html.H1(
                    "BizSentinel AI Assistant",
                    style={
                        "color": COLORS["primary"],
                        "fontSize": "20px",
                        "fontWeight": "700",
                        "margin": "0",
                    },
                ),
                html.Button(
                    "Restart",
                    id="restart-btn",
                    n_clicks=0,
                    style={
                        "marginLeft": "auto",
                        "padding": "6px 14px",
                        "background": "transparent",
                        "color": "#6C757D",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "4px",
                        "fontSize": "13px",
                        "cursor": "pointer",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "padding": "16px 32px",
                "background": COLORS["card"],
                "borderBottom": f"1px solid {COLORS['border']}",
            },
        ),
        # -- Chat area (inside Loading so spinner shows during long callbacks) --
        dcc.Loading(  # type: ignore[reportPrivateImportUsage]
            id="loading",
            type="circle",
            color=COLORS["primary"],
            children=html.Div(
                id="chat-messages",
                children=render_messages(_INITIAL_STORE),
                style={
                    "flex": "1",
                    "overflowY": "auto",
                    "padding": "24px 32px",
                    "display": "flex",
                    "flexDirection": "column",
                    "gap": "8px",
                },
            ),
        ),
        # -- Input area --
        html.Div(
            [
                dcc.Input(  # type: ignore[reportPrivateImportUsage]
                    id="chat-input",
                    type="text",
                    placeholder="Escribí tu pregunta…",
                    n_submit=0,
                    style={
                        "flex": "1",
                        "padding": "10px 16px",
                        "border": f"1px solid {COLORS['border']}",
                        "borderRadius": "6px",
                        "fontSize": "14px",
                        "outline": "none",
                    },
                ),
                html.Button(
                    "Send",
                    id="send-btn",
                    n_clicks=0,
                    style={
                        "padding": "10px 20px",
                        "background": COLORS["primary"],
                        "color": "white",
                        "border": "none",
                        "borderRadius": "6px",
                        "fontSize": "14px",
                        "fontWeight": "600",
                        "cursor": "pointer",
                        "marginLeft": "8px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "padding": "16px 32px",
                "background": COLORS["card"],
                "borderTop": f"1px solid {COLORS['border']}",
            },
        ),
        # -- Note --
        html.P(
            "El asistente puede tardar hasta 2 minutos en responder. Por favor esperá.",
            style={
                "textAlign": "center",
                "color": "#6C757D",
                "fontSize": "12px",
                "margin": "0",
                "padding": "8px 32px 12px",
                "background": COLORS["card"],
            },
        ),
    ],
    style={
        "display": "flex",
        "flexDirection": "column",
        "height": "100vh",
        "background": COLORS["background"],
        "fontFamily": FONT_FAMILY,
        "margin": "0",
    },
)

# ---------------------------------------------------------------------------
# Callback — send message, call Ollama, handle tool calls, update UI
# ---------------------------------------------------------------------------


@callback(
    Output("messages-store", "data"),
    Output("chat-messages", "children"),
    Input("restart-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_restart(n_clicks: int):
    return _INITIAL_STORE, render_messages(_INITIAL_STORE)


@callback(
    Output("messages-store", "data", allow_duplicate=True),
    Output("chat-messages", "children", allow_duplicate=True),
    Output("chat-input", "value"),
    Input("send-btn", "n_clicks"),
    Input("chat-input", "n_submit"),
    State("messages-store", "data"),
    State("chat-input", "value"),
    prevent_initial_call=True,
)
def handle_send(n_clicks: int, n_submit: int, messages: list[dict], user_input: str | None):
    text = (user_input or "").strip()
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if button_id not in ("send-btn", "chat-input") or not text:
        raise dash.exceptions.PreventUpdate
    messages = list(messages)
    messages.append({"role": "user", "content": text})

    response = ollama.chat(model=MODEL, messages=messages, tools=TOOLS)
    messages = _process_ollama_response(response, messages)
    children = render_messages(messages)

    return messages, children, ""


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8060"))
    debug = os.getenv("ENVIRONMENT", "production") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
