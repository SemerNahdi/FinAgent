# services/tools/groq_wrapper.py

import os
import aiohttp
import asyncio
import async_timeout
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GroqLLM:
    """
    Async/sync wrapper for the Groq API (chat-completion).

    Features:
    - Async and sync calls
    - Retry with delay
    - Timeout handling
    - Robust JSON parsing
    - Configurable endpoint
    """

    def __init__(self, api_key: str = None, endpoint: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not provided or missing in env variable 'GROQ_API_KEY'"
            )
        self.endpoint = endpoint or "https://api.groq.com/openai/v1/chat/completions"

    async def call_async(
        self,
        prompt: str,
        model: str = None,
        max_tokens: int = 256,
        retries: int = 3,
        timeout: int = 15,
    ) -> dict | str:
        """
        Send a prompt to Groq asynchronously and return parsed JSON or raw text.
        """
        # Resolve model from parameter, .env, or fallback
        model = model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }

        for attempt in range(1, retries + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with async_timeout.timeout(timeout):
                        logger.info(
                            f"[Attempt {attempt}] Sending prompt to Groq (model={model})"
                        )
                        async with session.post(
                            self.endpoint, headers=headers, json=payload
                        ) as resp:
                            if resp.status != 200:
                                text = await resp.text()
                                raise RuntimeError(
                                    f"Groq API error {resp.status}: {text}"
                                )
                            data = await resp.json()

                            # Extract message content
                            choices = data.get("choices", [])
                            if choices and "message" in choices[0]:
                                text = choices[0]["message"].get("content", "")
                            else:
                                text = ""

                            # Parse JSON if possible
                            try:
                                parsed = json.loads(text)
                                return parsed
                            except json.JSONDecodeError:
                                return text
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt == retries:
                    raise RuntimeError(
                        f"Groq call failed after {retries} attempts: {e}"
                    )
                await asyncio.sleep(1)

    def call(self, prompt: str, model: str = None, max_tokens: int = 256) -> dict | str:
        """
        Synchronous wrapper for call_async.
        """
        return asyncio.run(self.call_async(prompt, model=model, max_tokens=max_tokens))

    async def call_json_async(self, prompt: str, model: str = None) -> dict:
        """
        Async helper that ensures the response is always a dict with a 'plan' key.
        - List or string responses are normalized into {"plan": [...]}
        """
        result = await self.call_async(prompt, model=model)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list):
            return {"plan": result}
        elif isinstance(result, str):
            return {"plan": [result]}
        else:
            raise ValueError(f"Unexpected Groq response: {result}")

    def call_json(self, prompt: str, model: str = None) -> dict:
        """
        Synchronous helper that ensures the response is always a dict with a 'plan' key.
        """
        result = self.call(prompt, model=model)
        if isinstance(result, dict):
            return result
        elif isinstance(result, list):
            return {"plan": result}
        elif isinstance(result, str):
            return {"plan": [result]}
        else:
            raise ValueError(f"Unexpected Groq response: {result}")
