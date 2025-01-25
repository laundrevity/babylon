# conftest.py
import pytest
import os
from babylon.providers.openai_provider import OpenAIProvider
from babylon.providers.deepseek_provider import DeepSeekProvider
from babylon.providers.google_provider import GoogleProvider
from babylon.providers.anthropic_provider import AnthropicProvider

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

@pytest.fixture(scope="session")
def openai_provider():
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return OpenAIProvider(api_key=OPENAI_API_KEY)

@pytest.fixture(scope="session")
def deepseek_provider():
    if not DEEPSEEK_API_KEY:
        pytest.skip("DEEPSEEK_API_KEY environment variable not set")
    return DeepSeekProvider(api_key=DEEPSEEK_API_KEY)

@pytest.fixture(scope="session")
def google_provider():
    if not GOOGLE_API_KEY:
        pytest.skip("GOOGLE_API_KEY environment variable not set")
    return GoogleProvider(api_key=GOOGLE_API_KEY)

@pytest.fixture(scope="session")
def anthropic_provider():
    if not ANTHROPIC_API_KEY:
        pytest.skip("ANTHROPIC_API_KEY environment variable not set")
    return AnthropicProvider(api_key=ANTHROPIC_API_KEY)
