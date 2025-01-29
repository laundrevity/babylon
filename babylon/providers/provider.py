# babylon/providers/provider.py
import abc
import aiohttp
import json
import logging
from logging import getLogger as get_logger

# Configure basic logging - you can customize this further
logging.basicConfig(level=logging.DEBUG,  # Set default logging level to DEBUG
                    format='%(asctime)s.%(msecs)03d T[%(thread)d] %(levelname)s %(filename)s:%(lineno)d %(name)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S') # Customize format to include milliseconds and thread


class LLMProvider(abc.ABC):
    """
    Abstract base class for LLM providers, with common request handling and logging.
    """
    def __init__(self, api_key, base_url, headers=None, timeout=None):
        """
        Initializes the provider with API key, base URL, and default headers.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout or 30
        self.logger = get_logger(__name__) # Get logger for this class

    @abc.abstractmethod
    async def _build_url(self, model, messages, **kwargs):
        """Abstract method to build the API request URL."""
        pass

    @abc.abstractmethod
    async def _build_payload(self, model, messages, **kwargs):
        """Abstract method to build the API request payload (JSON body)."""
        pass

    @abc.abstractmethod
    def _extract_text_from_response(self, raw_response):
        """Abstract method to extract text from raw API response."""
        pass

    def _extract_usage_from_response(self, raw_response):
        """Optional method to extract usage info."""
        return None

    def _extract_finish_reason_from_response(self, raw_response):
        """Optional method to extract finish reason."""
        return None

    async def _make_request(self, url, headers, payload):
        """Handles common API request logic with logging."""
        logger = self.logger # Get logger instance

        logger.debug(f"Request URL: {url}") # Use logger.debug for URL
        logger.debug("Request Headers:") # Use logger.debug for headers
        logger.debug(headers)
        logger.debug("Request Payload:") # Use logger.debug for payload
        logger.debug(json.dumps(payload, indent=2))

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=self.timeout) as response:
                try:
                    response.raise_for_status()
                    response_json = await response.json()
                    logger.debug("Response JSON:") # Log successful JSON response
                    logger.debug(json.dumps(response_json, indent=2))
                    return response_json
                except aiohttp.ClientResponseError as e:
                    logger.error(f"ClientResponseError: {e}") # Use logger.error for ClientResponseError
                    logger.error("Response Headers on Error:") # Log headers on error
                    logger.error(response.headers)
                    raise e

    async def chat_completion(self, messages, model, **kwargs):
        """Provider-agnostic chat completion with logging."""
        url = await self._build_url(model, messages, **kwargs)
        payload = await self._build_payload(model, messages, **kwargs)
        raw_response = await self._make_request(url, self.headers, payload)

        text = self._extract_text_from_response(raw_response)
        usage = self._extract_usage_from_response(raw_response)
        finish_reason = self._extract_finish_reason_from_response(raw_response)

        return {
            'text': text if text else None,
            'provider': self.__class__.__name__.replace('Provider', '').lower(),
            'model': model,
            'raw_response': raw_response,
            'usage': usage,
            'finish_reason': finish_reason
        }
