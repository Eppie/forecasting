import re
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any

from llama_cpp import Llama
from pydantic import BaseModel

from forecasting.config import settings


class LLMOutput(BaseModel):
    raw_text: str
    probability_str: str | None = None  # e.g., "75%"
    probability_float: float | None = None  # e.g., 0.75
    rationale: str

    @classmethod
    def from_raw_text(cls, text: str) -> "LLMOutput":
        prob_str, prob_float = None, None
        # Try to extract "Final Probability: X%"
        match = re.search(r"Final Probability:\s*(\d{1,3})\s*%", text, re.IGNORECASE)
        if match:
            prob_str = match.group(0)  # Full match "Final Probability: X%"
            prob_float = float(match.group(1)) / 100.0

        # A simpler regex if the above is too strict or complex
        # match_simple = re.search(r"(\d{1,3})\s*%", text)
        # if match_simple:
        #     prob_float = float(match_simple.group(1)) / 100.0
        #     prob_str = f"{match_simple.group(1)}%"

        # Rationale is everything for now, can be refined
        rationale = text
        return cls(
            raw_text=text,
            probability_str=prob_str,
            probability_float=prob_float,
            rationale=rationale,
        )


class LlmEngineABC(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> LLMOutput:
        pass

    @abstractmethod
    def stream_generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3
    ) -> Generator[str]:
        pass


class LlamaCppEngine(LlmEngineABC):
    def __init__(
        self,
        model_path: str = settings.LLM_MODEL_PATH,
        n_gpu_layers: int = -1,
        n_ctx: int = 4096,
    ):  # -1 for all layers to GPU
        # print(f"Initializing LlamaCppEngine with model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,  # Context window
            verbose=False,  # Set to True for Llama.cpp's own logs
        )
        # print("LlamaCppEngine initialized.")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> LLMOutput:
        # Note: 'stream' param here is conceptual, llama_cpp Llama.__call__ has its own stream arg
        # This method returns the full response. For actual streaming, use stream_generate.
        if (
            stream
        ):  # This generate() method isn't for streaming output to console, but for getting full text
            full_text = ""
            for token_chunk in self.stream_generate(prompt, max_tokens, temperature):
                full_text += token_chunk
            return LLMOutput.from_raw_text(full_text)

        response: Any = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|eot_id|>"],  # Common stop tokens
            echo=False,  # Don't echo the prompt
            stream=False,
        )
        generated_text = response["choices"][0]["text"].strip()
        return LLMOutput.from_raw_text(generated_text)

    def stream_generate(
        self, prompt: str, max_tokens: int = 1024, temperature: float = 0.3
    ) -> Generator[str]:
        stream_response: Any = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["</s>", "<|eot_id|>"],
            echo=False,
            stream=True,
        )
        for chunk in stream_response:
            delta = chunk["choices"][0].get("text")
            if delta:
                yield delta
