# chat_loop.py
import json, ollama

from src.tools.web_search import brave_search

tools = [brave_search]

from pydantic import BaseModel


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
            out = brave_search(**tc.function.arguments)
            tool_outputs.append(
                {
                    "role": "tool",
                    "content": json.dumps(out)  # must be a string
                }
            )

    # 3️⃣ Ask the model to summarise the hits as structured JSON
    second = ollama.chat(
        model=model,
        messages=[
            *first.messages,  # original conversation so far
            *tool_outputs  # the search results
        ],
        format=TopLinks.model_json_schema(),  # constrain reply
        options={"temperature": 0},
    )

    return TopLinks.model_validate_json(second.message.content)


if __name__ == "__main__":
    run_chat("What is the news of today?")
