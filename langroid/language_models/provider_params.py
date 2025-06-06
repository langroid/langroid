"""
Provider-specific parameter configurations for various LLM providers.
"""

from typing import Any, Dict, Optional

from langroid.pydantic_v1 import BaseSettings

# Constants
LANGDB_BASE_URL = "https://api.us-east-1.langdb.ai"
PORTKEY_BASE_URL = "https://api.portkey.ai"
DUMMY_API_KEY = "xxx"


class LangDBParams(BaseSettings):
    """
    Parameters specific to LangDB integration.
    """

    api_key: str = DUMMY_API_KEY
    project_id: str = ""
    label: Optional[str] = None
    run_id: Optional[str] = None
    thread_id: Optional[str] = None
    base_url: str = LANGDB_BASE_URL

    class Config:
        # allow setting of fields via env vars,
        # e.g. LANGDB_PROJECT_ID=1234
        env_prefix = "LANGDB_"


class PortkeyParams(BaseSettings):
    """
    Parameters specific to Portkey integration.

    Portkey is an AI gateway that provides a unified API for multiple LLM providers,
    with features like automatic retries, fallbacks, load balancing, and observability.

    Example usage:
        # Use Portkey with Anthropic
        config = OpenAIGPTConfig(
            chat_model="portkey/anthropic/claude-3-sonnet-20240229",
            portkey_params=PortkeyParams(
                api_key="your-portkey-api-key",
                provider="anthropic"
            )
        )
    """

    api_key: str = DUMMY_API_KEY  # Portkey API key
    provider: str = ""  # Required: e.g., "openai", "anthropic", "cohere", etc.
    virtual_key: Optional[str] = None  # Optional: virtual key for the provider
    trace_id: Optional[str] = None  # Optional: trace ID for request tracking
    metadata: Optional[Dict[str, Any]] = None  # Optional: metadata for logging
    retry: Optional[Dict[str, Any]] = None  # Optional: retry configuration
    cache: Optional[Dict[str, Any]] = None  # Optional: cache configuration
    cache_force_refresh: Optional[bool] = None  # Optional: force cache refresh
    user: Optional[str] = None  # Optional: user identifier
    organization: Optional[str] = None  # Optional: organization identifier
    custom_headers: Optional[Dict[str, str]] = None  # Optional: additional headers
    base_url: str = PORTKEY_BASE_URL

    class Config:
        # allow setting of fields via env vars,
        # e.g. PORTKEY_API_KEY=xxx, PORTKEY_PROVIDER=anthropic
        env_prefix = "PORTKEY_"

    def get_headers(self) -> Dict[str, str]:
        """Generate Portkey-specific headers from parameters."""
        import json
        import os

        headers = {}

        # API key - from params or environment
        if self.api_key and self.api_key != DUMMY_API_KEY:
            headers["x-portkey-api-key"] = self.api_key
        else:
            portkey_key = os.getenv("PORTKEY_API_KEY", "")
            if portkey_key:
                headers["x-portkey-api-key"] = portkey_key

        # Provider
        if self.provider:
            headers["x-portkey-provider"] = self.provider

        # Virtual key
        if self.virtual_key:
            headers["x-portkey-virtual-key"] = self.virtual_key

        # Trace ID
        if self.trace_id:
            headers["x-portkey-trace-id"] = self.trace_id

        # Metadata
        if self.metadata:
            headers["x-portkey-metadata"] = json.dumps(self.metadata)

        # Retry configuration
        if self.retry:
            headers["x-portkey-retry"] = json.dumps(self.retry)

        # Cache configuration
        if self.cache:
            headers["x-portkey-cache"] = json.dumps(self.cache)

        # Cache force refresh
        if self.cache_force_refresh is not None:
            headers["x-portkey-cache-force-refresh"] = str(
                self.cache_force_refresh
            ).lower()

        # User identifier
        if self.user:
            headers["x-portkey-user"] = self.user

        # Organization identifier
        if self.organization:
            headers["x-portkey-organization"] = self.organization

        # Add any custom headers
        if self.custom_headers:
            headers.update(self.custom_headers)

        return headers

    def parse_model_string(self, model_string: str) -> tuple[str, str]:
        """
        Parse a model string like "portkey/anthropic/claude-3-sonnet"
        and extract provider and model name.

        Returns:
            tuple: (provider, model_name)
        """
        parts = model_string.split("/", 2)
        if len(parts) >= 3 and parts[0] == "portkey":
            _, provider, model = parts
            return provider, model
        else:
            # Fallback: just remove "portkey/" prefix and return empty provider
            model = model_string.replace("portkey/", "")
            return "", model

    def get_provider_api_key(
        self, provider: str, default_key: str = DUMMY_API_KEY
    ) -> str:
        """
        Get the API key for the provider from environment variables.

        Args:
            provider: The provider name (e.g., "anthropic", "openai")
            default_key: Default key to return if not found

        Returns:
            The API key for the provider
        """
        import os

        # Common environment variable patterns for different providers
        env_patterns = [
            f"{provider.upper()}_API_KEY",
            f"{provider.upper()}_KEY",
        ]

        for pattern in env_patterns:
            key = os.getenv(pattern, "")
            if key:
                return key

        return default_key
