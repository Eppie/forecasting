from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pytest_mock import MockerFixture  # type: ignore

import ollama

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.ollama_utils import execute_tool_calls, generate_search_queries


def test_generate_search_queries(mocker: MockerFixture) -> None:
    expected = ["query one", "query two"]
    fake_resp = ollama.ChatResponse(message=ollama.Message(role="assistant", content=json.dumps(expected)))
    mocker.patch("src.ollama_utils.ollama.chat", return_value=fake_resp)

    result = generate_search_queries("Will AI end the world?", n=2)
    assert result == expected


def test_execute_tool_calls() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    call = ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name="add", arguments={"a": 1, "b": 2}))
    messages = execute_tool_calls([call], {"add": add})
    assert len(messages) == 1
    assert messages[0].role == "tool"
    content = messages[0].content
    assert content is not None
    result = json.loads(content)
    expected_sum = 3
    assert result == expected_sum


def test_execute_tool_calls_unknown() -> None:
    call = ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name="missing", arguments={}))
    with pytest.raises(ValueError):
        execute_tool_calls([call], {})
