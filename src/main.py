import json
from typing import cast

import ollama
from pydantic import BaseModel

from src.tools.web_search import web_search

tools = [web_search]


def run_chat(user_prompt: str, model: str = "llama3.3") -> list:
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
        if tc.function.name == "web_search":
            out = web_search(**tc.function.arguments)
            tool_outputs.append(
                {
                    "role": "tool",
                    "content": json.dumps(out),  # must be a string
                }
            )
    return tool_outputs


if __name__ == "__main__":
    result = run_chat("What is the news of today? Search for at most one thing.")
    print(result)
