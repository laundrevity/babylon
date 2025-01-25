# babylon/providers/deepseek_provider.py
import sys
import aiohttp
import json
import os
from babylon.providers.provider import LLMProvider  # Import the refactored base class
from logging import getLogger as get_logger

class DeepSeekProvider(LLMProvider):
    """
    Implementation for DeepSeek API, inheriting from LLMProvider base class with logging.
    """
    def __init__(self, api_key, base_url="https://api.deepseek.com/v1"): # or just https://api.deepseek.com
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        super().__init__(api_key=api_key, base_url=base_url, headers=headers)
        self.logger = get_logger(__name__) # Get logger

    async def _build_url(self, model, messages, **kwargs):
        """DeepSeek-specific URL builder."""
        return f"{self.base_url}/chat/completions"

    async def _build_payload(self, model, messages, **kwargs):
        """DeepSeek-specific payload builder."""
        return {
            "model": model,
            "messages": messages,
            **kwargs
        }

    def _extract_text_from_response(self, raw_response):
        """DeepSeek-specific text extraction from response."""
        if raw_response and raw_response.get('choices'):
            return raw_response['choices'][0]['message']['content'].strip()
        return None

    def _extract_usage_from_response(self, raw_response):
        """DeepSeek-specific usage extraction."""
        return raw_response.get('usage') # Check if Deepseek provides usage

    def _extract_finish_reason_from_response(self, raw_response):
        """DeepSeek-specific finish reason extraction."""
        if raw_response and raw_response.get('choices'):
            return raw_response['choices'][0].get('finish_reason') # Check if Deepseek provides finish_reason
        return None


# Tests
if 'pytest' in sys.modules:
    import pytest
    import asyncio

    @pytest.mark.asyncio
    async def test_deepseek_chat_completion(deepseek_provider): # Fixture as argument
        log = get_logger("test_deepseek_chat_completion")
        messages = [{"role": "user", "content": "Hello DeepSeek, this is a test."}]
        model = "deepseek-chat"

        try:
            response = await deepseek_provider.chat_completion(messages, model)
            assert response['provider'] == 'deepseek'
            assert response['model'] == model
            assert response['text'] is not None
            assert isinstance(response['text'], str)
            log.info(f"DeepSeek Response Text (truncated): {response['text'][:50]}...")
            if 'usage' in response:
                log.info(f"DeepSeek Usage: {response['usage']}")
            if 'finish_reason' in response:
                log.info(f"DeepSeek Finish Reason: {response['finish_reason']}")

        except aiohttp.ClientError as e:
            pytest.fail(f"DeepSeek API request failed: {e}")
        except AssertionError as e:
            pytest.fail(f"DeepSeek Response assertion failed: {e}. Raw response: {response.get('raw_response') if 'response' in locals() else 'No response received'}")
