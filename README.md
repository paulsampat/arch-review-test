# arch-review-test

A small FastAPI service that wraps the
[Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk) to run coding
tasks against this repository.

`app/agent.py` is the module that owns all model interaction - it
configures the Agent SDK's built-in agent loop (tools, permissions, system
prompt, working directory) and exposes `CodingAgent.run()`. The SDK itself
handles the model calls, tool execution, and context management; this
module just configures it and shapes the output for the API layer.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-api-key
```

The SDK reads the key from the process environment - it does not load
`.env` files automatically.

## Run the API

```bash
uvicorn app.main:app --reload
```

## Use it

```bash
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Add a health check endpoint to app/main.py"}'
```
