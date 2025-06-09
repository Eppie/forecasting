"""Minimal stub implementation of the ollama client for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    role: str
    content: str | None = None
    tool_calls: list[Message.ToolCall] | None = None

    @dataclass
    class ToolCall:
        @dataclass
        class Function:
            name: str
            arguments: dict[str, Any]

        function: Function


@dataclass
class ChatResponse:
    message: Message


def chat(*args: Any, **kwargs: Any) -> ChatResponse:
    raise NotImplementedError
