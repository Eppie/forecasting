from __future__ import annotations

import logging
from typing import Any, cast

import ollama
import requests

logger = logging.getLogger(__name__)


class LLMRouter:
    """Route chat prompts to local LLM back-ends.

    Parameters
    ----------
    model :
        Model identifier understood by all back-ends.
    llama_cpp_url :
        Base URL (include `/v1`) of the llama.cpp server.
    ollama_url :
        Full URL of the Ollama `/api/chat` endpoint.
    lmstudio_url :
        Base URL (include `/v1`) of the LM Studio server.
    """

    _PATH_LLAMACPP = "/chat/completions"
    _TIMEOUT = 30

    def __init__(
        self,
        model: str = "llama3",
        llama_cpp_url: str = "http://localhost:8080/v1",
        ollama_url: str = "http://localhost:11434/api/chat",
        lmstudio_url: str = "http://localhost:1234/v1",
    ) -> None:
        self.model = model
        self.llama_cpp_url = llama_cpp_url.rstrip("/")
        self.ollama_url = ollama_url.rstrip("/")
        self.lmstudio_url = lmstudio_url.rstrip("/")

        self._handlers: dict[str, Any] = {
            "llama_cpp": self._chat_llama_cpp,
            "ollama": self._chat_ollama,
            "lmstudio": self._chat_lmstudio,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        backend: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Return a chat completion.

        If *backend* is given, only that engine is tried.
        Otherwise, the order is llama.cpp → Ollama → LM Studio.

        Raises
        ------
        RuntimeError
            When every selected engine is unreachable.
        """
        order: list[str] = [backend] if backend else ["llama_cpp", "ollama", "lmstudio"]
        for name in order:
            logger.debug("Trying %s", name)
            try:
                return cast(dict[str, Any], self._handlers[name](messages, **kwargs))
            except requests.exceptions.ConnectionError as e:
                logger.warning("Error connecting to %s\n Reason: %s", name, e)
                if backend:
                    raise RuntimeError(f"{name} back-end is unreachable") from None
                continue
        raise RuntimeError("All back-ends are unreachable")

    @staticmethod
    def _normalize(provider: str, raw: Any) -> dict[str, Any]:
        """Convert provider‑specific responses into an OpenAI‑style payload.

        The unified format is:
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "<assistant content>"
                    }
                }
            ]
        }
        """
        if provider == "ollama":
            # Ollama's SDK returns an object/dict where the assistant reply
            # is at raw['message']['content']
            content = raw["message"]["content"]
            role = raw["message"].get("role", "assistant")
            return {"choices": [{"message": {"role": role, "content": content}}]}
        return cast(dict[str, Any], raw)

    def _chat_llama_cpp(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages} | kwargs
        resp = requests.post(
            f"{self.llama_cpp_url}{self._PATH_LLAMACPP}",
            json=payload,
            timeout=self._TIMEOUT,
        )
        resp.raise_for_status()
        return self._normalize("openai", cast(Any, resp.json()))

    def _chat_ollama(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Chat using the local Ollama server via the official SDK."""
        payload: dict[str, Any] = {"model": self.model, "messages": messages} | kwargs
        payload.setdefault("stream", False)
        response = ollama.chat(**payload)
        return self._normalize("ollama", response)

    def _chat_lmstudio(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        payload = {"model": self.model, "messages": messages} | kwargs
        resp = requests.post(
            f"{self.lmstudio_url}{self._PATH_LLAMACPP}",
            json=payload,
            timeout=self._TIMEOUT,
        )
        resp.raise_for_status()
        return self._normalize("openai", cast(Any, resp.json()))


if __name__ == "__main__":
    router = LLMRouter(model="qwen3:32b")
    conversation = [{"role": "user", "content": "Explain quantum entanglement in two sentences."}]

    # Explicitly target Ollama (no fallback)
    try:
        logger.info("—Ollama—")
        logger.info(router.chat(conversation, backend="ollama")["choices"][0]["message"]["content"])
    except RuntimeError as err:
        logger.error("Ollama unreachable: %s", err)

    # Let the router fall back automatically
    logger.info("\n—Automatic fallback—")
    logger.info(router.chat(conversation)["choices"][0]["message"]["content"])
