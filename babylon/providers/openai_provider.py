# babylon/providers/openai_provider.py
import sys
import aiohttp
import os
from babylon.providers.provider import LLMProvider  # Import the refactored base class
from logging import getLogger as get_logger

class OpenAIProvider(LLMProvider):
    """
    Implementation for OpenAI API, inheriting from LLMProvider base class with logging.
    """
    def __init__(self, api_key, base_url="https://api.openai.com/v1"):
        headers = { # Define headers here and pass to super().__init__
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        super().__init__(api_key=api_key, base_url=base_url, headers=headers) # Call base class __init__
        self.logger = get_logger(__name__) # Get logger for this class


    async def _build_url(self, model, messages, **kwargs):
        """OpenAI-specific URL builder."""
        return f"{self.base_url}/chat/completions"

    async def _build_payload(self, model, messages, **kwargs):
        """OpenAI-specific payload builder."""
        return {
            "model": model,
            "messages": messages,
            **kwargs  # Include any extra parameters like temperature, max_tokens, etc.
        }

    def _extract_text_from_response(self, raw_response):
        """OpenAI-specific text extraction from response."""
        if raw_response and raw_response.get('choices'):
            return raw_response['choices'][0]['message']['content']
        return None

    def _extract_usage_from_response(self, raw_response):
        """OpenAI-specific usage extraction."""
        return raw_response.get('usage')

    def _extract_finish_reason_from_response(self, raw_response):
        """OpenAI-specific finish reason extraction."""
        if raw_response and raw_response.get('choices'):
            return raw_response['choices'][0].get('finish_reason')
        return None


# Tests (remain mostly the same, just class name change)
if 'pytest' in sys.modules:
    import pytest
    import asyncio

    @pytest.mark.asyncio  # Decorator for async test function
    async def test_openai_chat_completion(openai_provider: LLMProvider): # Fixture as direct argument
        log = get_logger("test_openai_chat_completion")
        messages = [{"role": "user", "content": "Hello OpenAI, this is a test."}]
        model = "gpt-4o-mini"

        try:
            response = await openai_provider.chat_completion(messages, model)
            assert response['provider'] == 'openai'
            assert response['model'] == model
            assert response['text'] is not None
            assert isinstance(response['text'], str)
            log.info(f"OpenAI Response Text (truncated): {response['text'][:50]}...") # print a snippet of response
            if 'usage' in response:
                log.info(f"OpenAI Usage: {response['usage']}")
            if 'finish_reason' in response:
                log.info(f"OpenAI Finish Reason: {response['finish_reason']}")

        except aiohttp.ClientError as e:
            pytest.fail(f"OpenAI API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"OpenAI Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

    @pytest.mark.asyncio
    async def test_openai_tool(openai_provider: LLMProvider):
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
        model = "gpt-4o-mini"

        try:
            response = await openai_provider.chat_completion(messages, model, tools=tools)

            for choice in response["raw_response"]["choices"]:
                if "tool_calls" in choice["message"]:
                    found_tool_call = True

        except aiohttp.ClientError as e:
            pytest.fail(f"Anthropic API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"Anthropic Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

        assert found_tool_call
