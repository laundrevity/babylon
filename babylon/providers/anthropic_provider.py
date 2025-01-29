# babylon/providers/anthropic_provider.py
import sys
import aiohttp
import json
import os
from babylon.providers.provider import LLMProvider  # Import the refactored base class
from logging import getLogger as get_logger

class AnthropicProvider(LLMProvider):
    """
    Implementation for Anthropic API, inheriting from LLMProvider base class with logging.
    """
    def __init__(self, api_key, base_url="https://api.anthropic.com/v1"):
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        super().__init__(api_key=api_key, base_url=base_url, headers=headers)
        self.logger = get_logger(__name__) # Get logger

    async def _build_url(self, model, messages, **kwargs):
        """Anthropic-specific URL builder."""
        return f"{self.base_url}/messages"

    async def _build_payload(self, model, messages, **kwargs):
        """Anthropic-specific payload builder."""
        return {
            "model": model,
            "messages": messages,
            "max_tokens": kwargs.get('max_tokens', 1024),
            **kwargs
        }

    def _extract_text_from_response(self, raw_response):
        """Anthropic-specific text extraction from response."""
        if raw_response and raw_response.get('content'):
            text_content_blocks = [block for block in raw_response['content'] if block['type'] == 'text']
            return "".join([block['text'] for block in text_content_blocks])
        return None

    def _extract_usage_from_response(self, raw_response):
        """Anthropic-specific usage extraction."""
        return raw_response.get('usage') # Anthropic provides usage

    def _extract_finish_reason_from_response(self, raw_response):
        """Anthropic-specific finish reason extraction."""
        return raw_response.get('stop_reason') # Anthropic stop_reason


# Tests
if 'pytest' in sys.modules:
    import pytest
    import asyncio

    @pytest.mark.asyncio
    async def test_anthropic_chat_completion(anthropic_provider): # Fixture as argument
        log = get_logger("test_anthropic_chat_completion")
        messages = [{"role": "user", "content": "Hello Anthropic Claude, this is a test."}]
        model = "claude-3-opus-20240229"

        try:
            response = await anthropic_provider.chat_completion(messages, model)
            assert response['provider'] == 'anthropic'
            assert response['model'] == model
            assert response['text'] is not None
            assert isinstance(response['text'], str)
            log.info(f"Anthropic Claude Response Text (truncated): {response['text'][:50]}...")
            if 'usage' in response:
                log.info(f"Anthropic Usage: {response['usage']}")
            if 'finish_reason' in response:
                log.info(f"Anthropic Finish Reason: {response['finish_reason']}")

        except aiohttp.ClientError as e:
            pytest.fail(f"Anthropic API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"Anthropic Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")

    @pytest.mark.asyncio
    async def test_anthropic_chat_completion_tool(anthropic_provider):
        log = get_logger(__name__)
        tools = [
            {
                "name": "get_stock_price",
                "description": "Get the current stock price for a given ticker symbol.",
                "input_schema": 
                {
                    "type": "object",
                    "properties": 
                    {
                        "ticker": 
                        {
                            "type": "string",
                            "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
                        }
                    },
                "required": ["ticker"]
                }
            }
        ]
        messages = [{"role": "user", "content": "Hello Anthropic Claude, this is a test. Whats the stock price for NVDA?"}]
        model = "claude-3-opus-20240229"
        try:
            response = await anthropic_provider.chat_completion(messages, model, tools=tools)
            log.info("recv response[%s]", response)
        except aiohttp.ClientError as e:
            pytest.fail(f"Anthropic API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"Anthropic Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")
