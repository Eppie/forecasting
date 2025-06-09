import json
from typing import cast

import ollama
from pydantic import BaseModel

from src.tools.web_search import web_search

tools = [web_search]


class TopLinks(BaseModel):
    sources: list[str]


def run_chat(user_prompt: str, model: str = "llama3.3") -> TopLinks:
    # 1️⃣ Let the model decide whether to call brave_search
    first = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        tools=tools,
        options={"temperature": 0},  # deterministic
    )

    # 2️⃣ Execute any tool calls it emitted
    tool_outputs = []
    for tc in first.message.tool_calls or []:
        if tc.function.name == "brave_search":
            out = web_search(**tc.function.arguments)
            tool_outputs.append(
                {
                    "role": "tool",
                    "content": json.dumps(out),  # must be a string
                }
            )

    # 3️⃣ Ask the model to summarise the hits as structured JSON
    second = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": user_prompt},
            first.message.model_dump(),
            *tool_outputs,
        ],
        format=TopLinks.model_json_schema(),  # constrain reply
        options={"temperature": 0},
    )

    content = second.message.content
    if content is None:
        raise ValueError("Model returned empty content")
    return cast(TopLinks, TopLinks.model_validate_json(content))


if __name__ == "__main__":
    run_chat("What is the news of today?")
