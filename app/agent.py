"""Coding agent built on the Claude Agent SDK.

This module owns every interaction with the model: it configures the
agent loop (tools, permissions, system prompt, working directory) and
exposes a small API for callers like the FastAPI app in `app/main.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a coding agent working inside this repository. Make focused, "
    "correct changes and explain what you did."
)

DEFAULT_ALLOWED_TOOLS = ["Read", "Edit", "Write", "Glob", "Grep", "Bash"]


@dataclass
class AgentStep:
    """One observable event from a run: Claude talking, or calling a tool."""

    kind: str  # "text" | "tool_use"
    text: str | None = None
    tool_name: str | None = None
    tool_input: dict | None = None


@dataclass
class AgentResult:
    """Everything a run produced, plus its final status."""

    steps: list[AgentStep] = field(default_factory=list)
    success: bool = False
    subtype: str | None = None

    @property
    def text(self) -> str:
        """Concatenate all of Claude's text output, in order."""
        return "\n".join(
            step.text for step in self.steps if step.kind == "text" and step.text
        )


class CodingAgent:
    """Thin wrapper around the Claude Agent SDK's built-in agent loop.

    One instance = one configuration (tools, permissions, system prompt,
    working directory). Call `run()` per task; each call is a fresh session.
    The SDK handles the model calls, tool execution, and context management
    internally - this class just configures it and shapes the output.
    """

    def __init__(
        self,
        cwd: str | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        allowed_tools: list[str] | None = None,
        permission_mode: str = "acceptEdits",
        model: str | None = None,
        max_turns: int | None = None,
    ) -> None:
        self._options = ClaudeAgentOptions(
            cwd=cwd,
            system_prompt=system_prompt,
            allowed_tools=(
                list(allowed_tools)
                if allowed_tools is not None
                else list(DEFAULT_ALLOWED_TOOLS)
            ),
            permission_mode=permission_mode,
            model=model,
            max_turns=max_turns,
        )

    async def run(self, prompt: str) -> AgentResult:
        """Run one task to completion and return everything that happened."""
        result = AgentResult()

        async for message in query(prompt=prompt, options=self._options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        result.steps.append(AgentStep(kind="text", text=block.text))
                    elif isinstance(block, ToolUseBlock):
                        result.steps.append(
                            AgentStep(
                                kind="tool_use",
                                tool_name=block.name,
                                tool_input=block.input,
                            )
                        )
            elif isinstance(message, ResultMessage):
                result.subtype = message.subtype
                result.success = message.subtype == "success"

        return result
