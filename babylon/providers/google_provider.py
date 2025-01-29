# babylon/providers/google_provider.py
import sys
import aiohttp
import json
import os
from babylon.providers.provider import LLMProvider  # Import the refactored base class
from logging import getLogger as get_logger

class GoogleProvider(LLMProvider):
    """
    Implementation for Google Gemini API, inheriting from LLMProvider base class with logging.
    """
    def __init__(self, api_key, base_url="https://generativelanguage.googleapis.com/v1beta"):
        headers = {"Content-Type": "application/json"}
        super().__init__(api_key=api_key, base_url=base_url, headers=headers)
        self.logger = get_logger(__name__) # Get logger

    async def _build_url(self, model, messages, **kwargs):
        """Google-specific URL builder."""
        return f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"

    async def _build_payload(self, model, messages, **kwargs):
        """Google-specific payload builder."""
        prompt_text = ""
        for msg in messages:
            if msg['role'] == 'user':
                prompt_text += msg['content'] + "\n"
        return {
            "contents": [{
                "parts": [{"text": prompt_text.strip()}]
            }],
            **kwargs
        }

    def _extract_text_from_response(self, raw_response):
        """Google-specific text extraction from response."""
        if raw_response and raw_response.get('candidates') and raw_response['candidates'][0].get('content') and raw_response['candidates'][0]['content'].get('parts'):
            text_parts = raw_response['candidates'][0]['content']['parts']
            return "".join([part['text'] for part in text_parts if part.get('text')])
        return None

    def _extract_usage_from_response(self, raw_response):
        """Google-specific usage extraction."""
        return raw_response.get('usageMetadata') # Check Google usage metadata

    def _extract_finish_reason_from_response(self, raw_response):
        """Google-specific finish reason extraction."""
        if raw_response and raw_response.get('candidates') and raw_response['candidates'][0].get('finishReason'):
            return raw_response['candidates'][0].get('finishReason') # Check Google finish reason
        return None


# Tests
if 'pytest' in sys.modules:
    import pytest
    import asyncio

    @pytest.mark.asyncio
    async def test_google_chat_completion(google_provider): # Fixture as argument
        log = get_logger("test_google_chat_completion")
        messages = [{"role": "user", "content": "Hello Google Gemini, this is a test."}]
        model = "gemini-1.5-flash"

        try:
            response = await google_provider.chat_completion(messages, model)
            assert response['provider'] == 'google'
            assert response['model'] == model
            assert response['text'] is not None
            assert isinstance(response['text'], str)
            log.info(f"Google Gemini Response Text (truncated): {response['text'][:50]}...")
            if 'usage' in response:
                log.info(f"Google Usage: {response['usage']}")
            if 'finish_reason' in response:
                log.info(f"Google Finish Reason: {response['finish_reason']}")

        except aiohttp.ClientError as e:
            pytest.fail(f"Google API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"Google Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

    @pytest.mark.asyncio
    async def test_google_tool(google_provider: LLMProvider):
        log = get_logger(__name__)
        found_tool_call = False
        tools = [
            {
                "function_declarations": [
                    {
                        "name": "get_stock_price",
                        "description": "Get the current stock price for a given ticker symbol.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "ticker": {
                                    "type": "string",
                                    "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                                }
                            }
                        }
                    }
                ]
            }
        ]
        messages = [{"role": "user", "content": "Hello Google Gemini, this is a test. What's the stock price for NVDA?"}]
        model = "gemini-pro"
        try:
            response = await google_provider.chat_completion(messages, model, tools=tools)

            for candidate in response["raw_response"]["candidates"]:
                for part in candidate["content"]["parts"]:
                    if "functionCall" in part:
                        found_tool_call = True

        except aiohttp.ClientError as e:
            pytest.fail(f"Anthropic API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"Anthropic Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

        assert found_tool_call
