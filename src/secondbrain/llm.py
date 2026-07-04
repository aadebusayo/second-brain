"""
LLM client wrappers for secondbrain.

Provides a unified interface for Anthropic, OpenAI, and DeepSeek
backends so entity extraction and consolidation work with any
supported provider.
"""

from __future__ import annotations

from typing import Any


class DeepSeekClient:
    """
    DeepSeek LLM client via the OpenAI-compatible API.

    Uses deepseek-chat (V4 Flash) for fast, cost-effective extraction
    and summarisation.
    """

    BASE_URL = "https://api.deepseek.com/v1"
    MODEL = "deepseek-chat"  # standard model for accurate entity extraction

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model or self.MODEL
        self.base_url = base_url or self.BASE_URL

    @property
    def chat(self):
        """OpenAI-compatible chat completions interface."""
        from openai import OpenAI

        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        ).chat

    def messages(self):
        """Alias for anthropic-style 'messages' interface (not supported).
        Raises an explicit error so calling code can fall back."""
        raise NotImplementedError(
            "DeepSeek uses the OpenAI-compatible chat.completions API. "
            "Use client.chat.completions.create() instead."
        )


def create_llm_client(provider: str = "deepseek", api_key: str | None = None) -> Any:
    """
    Factory: create an LLM client for the given provider.

    Supported providers: deepseek, openai, anthropic.
    """
    import os

    provider = provider.lower()

    if provider == "deepseek":
        key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY required for DeepSeek")
        return DeepSeekClient(api_key=key)

    if provider == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OPENAI_API_KEY required for OpenAI")
        from openai import OpenAI

        return OpenAI(api_key=key)

    if provider == "anthropic":
        key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY required for Anthropic")
        import anthropic

        return anthropic.Anthropic(api_key=key)

    raise ValueError(f"Unknown LLM provider: {provider}")
