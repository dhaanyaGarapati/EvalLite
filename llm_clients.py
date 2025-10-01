
from __future__ import annotations
import os
from typing import Dict

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

class LLMClients:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self._openai = OpenAI(api_key=self.openai_key) if (OpenAI and self.openai_key) else None
        self._anthropic = anthropic.Anthropic(api_key=self.anthropic_key) if (anthropic and self.anthropic_key) else None

    def available(self) -> Dict[str, bool]:
        return {
            "openai": self._openai is not None,
            "anthropic": self._anthropic is not None
        }

    def generate_openai(self, prompt: str, model: str = "gpt-4o-mini", max_tokens: int = 400, temperature: float = 0.2) -> str:
        if not self._openai:
            raise RuntimeError("OpenAI client not initialized; set OPENAI_API_KEY")
        resp = self._openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def generate_anthropic(self, prompt: str, model: str = "claude-3-haiku-20240307", max_tokens: int = 400, temperature: float = 0.2) -> str:
        if not self._anthropic:
            raise RuntimeError("Anthropic client not initialized; set ANTHROPIC_API_KEY")
        msg = self._anthropic.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        parts = []
        for blk in msg.content:
            if getattr(blk, "type", None) == "text":
                parts.append(blk.text)
        return "\n".join(parts).strip() if parts else ""
