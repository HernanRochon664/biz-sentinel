"""BizSentinel Ollama Chat — interactive LLM assistant with tool calling.

Usage:
    uv run python -m biz_sentinel.scripts.ollama_chat

Requires Ollama running locally with gemma4:e4b (or model in .env OLLAMA_MODEL).
"""

import json
import os
import re

import ollama

from biz_sentinel.serving.mcp.server import (
    explain_alert,
    get_anomaly_summary,
    get_customer_risk,
    get_segment_profile,
)

# ---------------------------------------------------------------------------
# Tool definitions (declared for Ollama's native tool-calling API)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_anomaly_summary",
            "description": "Get summary statistics of recent customer anomalies",
            "parameters": {
                "type": "object",
                "properties": {
                    "days": {
                        "type": "integer",
                        "description": "Number of days to consider",
                        "default": 7,
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_risk",
            "description": "Get full risk profile for a specific customer hash",
            "parameters": {
                "type": "object",
                "required": ["customer_hash"],
                "properties": {
                    "customer_hash": {
                        "type": "string",
                        "description": "Pseudonymized customer identifier",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_segment_profile",
            "description": "Get profile and description of a customer segment",
            "parameters": {
                "type": "object",
                "required": ["segment_label"],
                "properties": {
                    "segment_label": {
                        "type": "string",
                        "description": "champions|loyal|at_risk|new_customers|hibernating|lost",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_alert",
            "description": "Get natural language explanation of a specific alert",
            "parameters": {
                "type": "object",
                "required": ["alert_id"],
                "properties": {
                    "alert_id": {
                        "type": "integer",
                        "description": "Alert ID number",
                    }
                },
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool router
# ---------------------------------------------------------------------------

_TOOL_DISPATCH = {
    "get_anomaly_summary": get_anomaly_summary,
    "get_customer_risk": get_customer_risk,
    "get_segment_profile": get_segment_profile,
    "explain_alert": explain_alert,
}


def call_tool(name: str, args: dict):
    fn = _TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    return fn(**args)


# ---------------------------------------------------------------------------
# System prompt (explicit about JSON format for models that don't support
# native tool-calling, e.g. gemma4:e4b)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a business intelligence assistant for BizSentinel.\n"
    "RULES:\n"
    "1. When you need data, respond ONLY with JSON: "
    '{"name": "tool_name", "arguments": {...}}\n'
    "2. When you receive a tool result (role: tool), "
    "respond in natural language summarizing the data. "
    "DO NOT call another tool.\n"
    "3. Never call the same tool twice in one conversation turn.\n"
    "4. Your FINAL answer must ALWAYS be natural language. "
    "NEVER output JSON or code blocks in your final answer.\n"
    "Current language: respond in the same language the user writes in.\n"
    "Available tools: get_anomaly_summary, get_customer_risk, "
    "get_segment_profile, explain_alert."
)

# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------


def main():
    model = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"BizSentinel Chat — {model}")
    print("Escribí tu pregunta (o exit para salir)")
    print("-" * 50)

    while True:
        user_input = input("\nVos: ").strip()
        if user_input.lower() in ("exit", "quit", "salir"):
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = ollama.chat(model=model, messages=messages, tools=TOOLS)

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
                print(f"  [tool: {fn_name}({fn_args})]")

                result = call_tool(fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(result),
                        "name": fn_name,
                    }
                )

            final = ollama.chat(model=model, messages=messages)
            print(f"\nBizSentinel: {final.message.content}")
            messages.append({"role": "assistant", "content": final.message.content})

        # --- Path 2: text-based JSON tool call (qwen2.5-coder:* fallback) ---
        elif msg.content and msg.content.strip().startswith("{"):
            try:
                content = re.sub(r"```json\n?|\n?```", "", msg.content).strip()
                parsed = json.loads(content)
                if "name" in parsed and "arguments" in parsed:
                    fn_name = parsed["name"]
                    fn_args = parsed["arguments"]
                    print(f"  [tool: {fn_name}({fn_args})]")
                    result = call_tool(fn_name, fn_args)
                    messages.append(
                        {
                            "role": "tool",
                            "content": json.dumps(result),
                            "name": fn_name,
                        }
                    )
                    final = ollama.chat(model=model, messages=messages)
                    print(f"\nBizSentinel: {final.message.content}")
                    messages.append(
                        {"role": "assistant", "content": final.message.content}
                    )
                else:
                    print(f"\nBizSentinel: {msg.content}")
                    messages.append(
                        {"role": "assistant", "content": msg.content}
                    )
            except (json.JSONDecodeError, KeyError):
                print(f"\nBizSentinel: {msg.content}")
                messages.append({"role": "assistant", "content": msg.content})

        # --- Path 3: plain text response ---
        else:
            print(f"\nBizSentinel: {msg.content}")
            messages.append({"role": "assistant", "content": msg.content})


if __name__ == "__main__":
    main()
