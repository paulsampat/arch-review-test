"""Manual coding agent built directly on the Claude API (Messages endpoint).

Compare this to `app/agent.py`. There, the Claude Agent SDK spawns a
bundled Claude Code binary that owns the whole loop - model calls, tool
parsing, tool execution (Read/Edit/Bash are built in), and looping - and
`CodingAgent` just configures it and reads its output stream.

Here, there is no bundled binary and no built-in tools. Every piece of
the harness is written out below:
  - the tool schemas Claude sees (TOOLS)
  - the functions that actually execute a tool call (execute_tool and its
    helpers) - there is no free Read/Edit/Bash, you implement each one
  - the `while` loop that calls the model, inspects `stop_reason`, and
    either stops or executes tool calls and feeds `tool_result` blocks
    back for another turn (ManualCodingAgent.run)

This file is not wired into app/main.py - it exists to show the two
approaches side by side. It needs `anthropic` installed and
ANTHROPIC_API_KEY set to actually run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent working inside this repository. Make focused, "
    "correct changes and explain what you did."
)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TURNS = 20

# --- Tool schemas: what the model is told it can call --------------------
# The Agent SDK gets Read/Edit/Write/Glob/Grep/Bash for free. Here, each
# one is a schema you write plus a Python function you write below.

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a UTF-8 text file, path relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite (or create) a UTF-8 text file, path relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_files",
        "description": "List files matching a glob pattern, relative to the repo root.",
        "input_schema": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "run_bash",
        "description": "Run a shell command in the repo root; returns exit code, stdout, stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


# --- Tool implementations: execution, sandboxing, errors - all yours -----

def _read_file(cwd: Path, path: str) -> str:
    return (cwd / path).read_text()


def _write_file(cwd: Path, path: str, content: str) -> str:
    target = cwd / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"Wrote {len(content)} bytes to {path}"


def _list_files(cwd: Path, pattern: str) -> str:
    matches = [str(p.relative_to(cwd)) for p in cwd.glob(pattern)]
    return "\n".join(matches) if matches else "(no matches)"


def _run_bash(cwd: Path, command: str) -> str:
    proc = subprocess.run(
        command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=60
    )
    return f"exit={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def execute_tool(cwd: Path, name: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Dispatch one tool call. Returns (result_text, is_error)."""
    try:
        if name == "read_file":
            return _read_file(cwd, tool_input["path"]), False
        if name == "write_file":
            return _write_file(cwd, tool_input["path"], tool_input["content"]), False
        if name == "list_files":
            return _list_files(cwd, tool_input["pattern"]), False
        if name == "run_bash":
            return _run_bash(cwd, tool_input["command"]), False
        return f"Unknown tool: {name}", True
    except Exception as exc:  # noqa: BLE001 - surface any failure to the model
        return f"Error running {name}: {exc}", True


# --- The harness: the loop the Agent SDK would otherwise own -------------

@dataclass
class AgentStep:
    kind: str  # "text" | "tool_use" | "tool_result"
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_result: str | None = None


@dataclass
class AgentResult:
    steps: list[AgentStep] = field(default_factory=list)
    success: bool = False

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.steps if s.kind == "text" and s.text)


class ManualCodingAgent:
    """Same job as `CodingAgent` in app/agent.py, but every part of the
    loop is code written here: the model call, the stop-reason check,
    the tool dispatch, and the tool_result assembly for the next turn.
    """

    def __init__(
        self,
        cwd: str | Path | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self._cwd = Path(cwd or ".").resolve()
        self._system_prompt = system_prompt
        self._model = model
        self._max_turns = max_turns

    def run(self, prompt: str) -> AgentResult:
        result = AgentResult()
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]

        for _turn in range(self._max_turns):
            response = self._client.messages.create(
                model=self._model,
                max_tokens=16000,
                system=self._system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            for block in response.content:
                if block.type == "text":
                    result.steps.append(AgentStep(kind="text", text=block.text))

            # Append the assistant turn (text + any tool_use blocks) before
            # you can send tool results - the API requires that ordering.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                result.success = response.stop_reason == "end_turn"
                break

            # No built-in tools: you dispatch every call yourself.
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result.steps.append(
                    AgentStep(kind="tool_use", tool_name=block.name, tool_input=block.input)
                )
                text, is_error = execute_tool(self._cwd, block.name, block.input)
                result.steps.append(AgentStep(kind="tool_result", tool_result=text))
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": text,
                        "is_error": is_error,
                    }
                )

            # All of this turn's tool results go back in a single user
            # message - splitting them across messages breaks parallel
            # tool use.
            messages.append({"role": "user", "content": tool_results})
        else:
            result.success = False  # hit max_turns without finishing

        return result


if __name__ == "__main__":
    agent = ManualCodingAgent()
    outcome = agent.run("List the files in this repository.")
    print(outcome.text)
