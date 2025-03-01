from logging import getLogger as get_logger
import sys

import aiohttp

from babylon.providers.provider import LLMProvider


class xAIProvider(LLMProvider):
    """
    Impl for xAI, inheriting from LLMProvider
    """
    def __init__(self, api_key, base_url="https://api.x.ai/v1"):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        super().__init__(api_key, base_url, headers)
        self.logger = get_logger(__name__)

    async def _build_url(self, model, messages, **kwargs):
        return f"{self.base_url}/chat/completions"

    async def _build_payload(self, model, messages, **kwargs):
        return {
            "model": model,
            "messages": messages,
            **kwargs
        }

    def _extract_text_from_response(self, raw_response):
        if raw_response and raw_response.get("choices"):
            return raw_response["choices"][0]["message"]["content"]
        return None

    def _extract_usage_from_response(self, raw_response):
        return raw_response.get("usage")

    def _extract_finish_reason_from_response(self, raw_response):
        if raw_response and raw_response.get("choices"):
            return raw_response["choices"][0].get("finish_reason")
        return None


# Tests

if 'pytest' in sys.modules:
    import pytest

    @pytest.mark.asyncio
    async def test_xai_chat_completion(xai_provider: LLMProvider):
        log = get_logger("test_xai_chat_completion")
        messages = [{"role": "user", "content": "Hello xAI, this is a test."}]
        model = "grok-2"

        try:
            response = await xai_provider.chat_completion(messages, model)
            assert response["provider"] == "xai"
            assert response["model"] == model
            assert response["text"] is not None
            assert isinstance(response["text"], str)
            log.info("xAI rcvd response[%s]", response)
            if "usage" in response:
                log.info("xAI rcvd usage[%s]", response["usage"])
            if "finish_reason" in response:
                log.info("xAI rcvd finish reason[%s]", response["finish_reason"])
        except aiohttp.ClientError as e:
            pytest.fail(f"xAI API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"xAI Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

    @pytest.mark.asyncio
    async def test_xai_tool(xai_provider: LLMProvider):
        log = get_logger(__name__)
        found_tool_call = False
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Get the current stock price for a given ticker symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                            }
                        },
                        "required": ["ticker"]
                    }
                }
            }
        ]
        messages = [{"role": "user", "content": "Hello GPT, this is a test. What's the stock price for NVDA?"}]
        model = "grok-2"

        try:
            response = await xai_provider.chat_completion(messages, model, tools=tools)
            log.info("xAI rcvd response[%s]", response)

            for choice in response["raw_response"]["choices"]:
                if "tool_calls" in choice["message"]:
                    found_tool_call = True

        except aiohttp.ClientError as e:
            pytest.fail(f"Anthropic API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"Anthropic Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

        assert found_tool_call
