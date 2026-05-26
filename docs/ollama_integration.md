# BizSentinel MCP + Ollama Integration

## Overview

The Model Context Protocol (MCP) is an open standard that lets LLMs securely invoke tools defined by external servers. BizSentinel exposes its anomaly detection, risk scoring, and customer segmentation results through an MCP server, allowing any MCP-compatible client — including Claude Desktop, custom scripts, and IDE plugins — to query business intelligence data via natural language. This turns BizSentinel from a batch-processing pipeline into an interactive AI assistant for SME stakeholders.

Ollama is the recommended runtime for running LLMs with BizSentinel's MCP server because it is fully open source, runs entirely on your local machine, and requires no API keys or cloud accounts. All data stays on-premise, aligning with BizSentinel's privacy-preserving design philosophy. Combined with Ollama's broad model support and low-friction setup, you get a production-ready local AI assistant without compromising sensitive business data.

## Prerequisites

- **Ollama** installed — download from [https://ollama.com](https://ollama.com)
- **BizSentinel** running (either via `docker compose up` or locally with `uv run`)
- A compatible LLM model pulled locally

### Recommended Models

| Model | RAM/VRAM | Speed (CPU) | Speed (GPU) | Tool Use |
|-------|----------|-------------|-------------|----------|
| `qwen2.5-coder:7b` | 8GB RAM | 30s–2min | 5–15s | ✓ Stable |
| `qwen2.5-coder:3b` | 4GB RAM | 15–30s | 3–8s | ⚠ Unstable |
| `gemma4:e4b` | 4GB RAM | 15–30s | 3–8s | ⚠ Parser issues in some clients |

Pull the recommended model:

```bash
ollama pull llama3.1:8b
```

## Starting the MCP Server

### a) Standalone (local Python)

```bash
uv run python -m biz_sentinel.serving.mcp.server stdio
```

### b) Via Docker

```bash
docker compose up mcp
```

### c) With MCP Inspector (development / testing)

```bash
npx @modelcontextprotocol/inspector \
  uv run python -m biz_sentinel.serving.mcp.server stdio
```

The Inspector opens a web UI where you can list available tools, test each one manually, and inspect raw request/response payloads — useful during development.

> **Note:** For the web chat interface (`localhost:8060`), the MCP server does not need to be started separately. The `chat.py` app calls tools directly via Python imports. The stdio MCP server is only needed for external MCP clients (Claude Desktop, MCP Inspector, etc.).

## Connecting to Claude Desktop (for Developers)

Add the following entry to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "biz-sentinel": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "biz_sentinel.serving.mcp.server",
        "stdio"
      ],
      "cwd": "/path/to/biz-sentinel"
    }
  }
}
```

Restart Claude Desktop. A hammer icon appears — click it to see BizSentinel's available tools.

## Example Conversations

Below are realistic interactions. The exact phrasing of the LLM's response will vary by model and temperature, but the tool call and the kind of answer should be similar.

---

### Example 1: Weekly anomaly summary

**User:** "How many anomalies were detected this week?"

**Tool call:**
```
get_anomaly_summary(days=7)
```

**Expected response:**
> This week BizSentinel detected **12 anomalies** across your transaction data. The majority (8) are revenue-related anomalies — unexpected drops in daily revenue. The remaining 4 are operational anomalies flagged in order fulfillment. No critical (severity 5) alerts were triggered, but you may want to review the 2 high-severity revenue drops on Tuesday.

---

### Example 2: Individual customer risk

**User:** "Is customer abc123def456 at risk?"

**Tool call:**
```
get_customer_risk(customer_id="abc123def456")
```

**Expected response:**
> Customer `abc123def456` has a **churn risk score of 0.82** (high). Their transaction frequency has dropped 40 % over the last 30 days and their average order value is declining. I recommend a retention offer — consider a personalized discount or a re-engagement email campaign. Their segment profile is "slipping" (previously high-value, now disengaging).

---

### Example 3: Alert explanation

**User:** "Explain alert number 42"

**Tool call:**
```
explain_alert(alert_id=42)
```

**Expected response:**
> Alert #42 was triggered **2025-03-12 14:32 UTC** — an anomalous **72 % drop** in hourly transaction volume compared to the same window over the prior 14 days. The likely cause was a 45-minute outage in the payment gateway (confirmed by the infrastructure team). No action is needed now; the system auto-recovered. If you see a similar pattern without a known outage, check the payment provider status page.

---

### Example 4: Segment-based action

**User:** "What should I do with my at-risk customers?"

**Tool call:**
```
get_segment_profile(segment="at_risk")
```

**Expected response:**
> Your **at-risk segment** contains 143 customers (≈ 9 % of your active base). Key characteristics:
> - Avg. days since last purchase: 45
> - Avg. churn score: 0.74
> - Most common tier: mid-value (68 %)
>
> Recommended actions:
> 1. **Send a win-back email** with a 15 % discount code — this recovered 22 % of at-risk customers in last quarter's campaign.
> 2. **Increase survey frequency** — ask why they're disengaging.
> 3. **Monitor weekly** — move them to the "churned" segment if no activity for 60 more days.

> **Note:** Small models (3b) may return tool calls as plain text JSON instead of structured `tool_calls`. BizSentinel's chat interface includes automatic fallback parsing to handle this correctly.

## Known Limitations

- **qwen2.5-coder:7b** on CPU takes 30s–2min per response — acceptable for interactive use but not for real-time.
- Models smaller than **7b** have **unreliable tool use**; expect occasional failures or malformed responses.
- The **web chat interface** (localhost:8060) handles slow responses with a loading indicator — use it for best UX on CPU.
- For **production use**, a GPU with **12GB+ VRAM** is recommended (e.g., NVIDIA T4 or better).
- **gemma4:e4b** tool calling may fail via OpenAI-compatible API — this is a known issue fixed in Ollama 0.20.2+.

## Privacy Notes

- MCP tools return **pseudonymized hashes only** — no real customer PII (names, emails, phone numbers) is exposed to the LLM.
- Raw transaction data (amounts, timestamps, product IDs) is never sent to the model.
- Because Ollama runs locally, **no data leaves your machine** — all inference happens on-premise.
- This setup is suitable for demos, development, and internal use. For production deployments, review your LLM provider's data-handling policies and consider additional access controls on the MCP server itself.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `No scores found` | Training / scoring pipeline never ran | Execute `python -m biz_sentinel.pipeline.run` or trigger via Kedro |
| `Connection refused on port 11434` | Ollama server not running | Start Ollama with `ollama serve` (or restart the Ollama app) |
| Model responses are slow | Model too large for available hardware | Use a smaller model (`phi4-mini` or `llama3.1:8b`), or enable GPU acceleration (`OLLAMA_GPU=1`) |
| Tools not showing in Claude Desktop | MCP server failed to start or path is wrong | Check `claude_desktop_config.json` — verify `cwd` is absolute and `args` are correct |
