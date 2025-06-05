#!/usr/bin/env python3

"""
Basic Portkey example showing how to use different AI providers through Portkey's gateway.

This example demonstrates:
- Basic Portkey configuration
- Switching between different AI providers
- Automatic API key resolution

Run with: python portkey_basic_chat.py
"""

import os
from typing import Optional

import langroid.language_models as lm
from langroid.language_models.provider_params import PortkeyParams


def check_env_var(var_name: str) -> Optional[str]:
    """Check if environment variable is set and return its value."""
    value = os.getenv(var_name)
    if not value:
        print(f"⚠️  Warning: {var_name} not set in environment")
        return None
    return value


def create_portkey_llm(provider: str, model: str, portkey_api_key: str) -> lm.OpenAIGPT:
    """Create a Portkey-enabled LLM configuration."""
    config = lm.OpenAIGPTConfig(
        chat_model=f"portkey/{provider}/{model}",
        portkey_params=PortkeyParams(
            api_key=portkey_api_key,
        ),
        max_output_tokens=150,
        temperature=0.7,
    )
    return lm.OpenAIGPT(config)


def test_provider(llm: lm.OpenAIGPT, provider_name: str):
    """Test a specific provider with a simple question."""
    print(f"\n🔮 Testing {provider_name}...")
    print("=" * 50)

    question = "What is the capital of France? Answer in one sentence."

    try:
        response = llm.chat(question)
        print(f"✅ {provider_name} Response:")
        print(f"   {response.message}")

        if response.usage:
            print(f"   📊 Tokens: {response.usage.total_tokens}")

    except Exception as e:
        print(f"❌ {provider_name} Error: {str(e)}")


def main():
    """Main function demonstrating Portkey basic usage."""
    print("🚀 Portkey Basic Chat Example")
    print("=" * 40)

    # Check for required environment variables
    portkey_api_key = check_env_var("PORTKEY_API_KEY")
    if not portkey_api_key:
        print("❌ PORTKEY_API_KEY is required. Please set it in your environment.")
        return

    print("✅ Portkey API key found")

    # Test different providers through Portkey
    providers_to_test = []

    # Check which provider keys are available
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append(("OpenAI", "openai", "gpt-4o-mini"))
        print("✅ OpenAI API key found")

    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append(("Anthropic", "anthropic", "claude-3-haiku-20240307"))
        print("✅ Anthropic API key found")

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers_to_test.append(("Google Gemini", "google", "gemini-2.0-flash-lite"))
        print("✅ Google/Gemini API key found")

    if not providers_to_test:
        print("❌ No provider API keys found. Please set at least one of:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY or GEMINI_API_KEY")
        return

    print(f"\n🎯 Testing {len(providers_to_test)} provider(s) through Portkey...")

    # Test each available provider
    for provider_display_name, provider, model in providers_to_test:
        try:
            llm = create_portkey_llm(provider, model, portkey_api_key)
            test_provider(llm, provider_display_name)
        except Exception as e:
            print(f"❌ Failed to create {provider_display_name} LLM: {str(e)}")

    print("\n🎉 Portkey basic chat example completed!")
    print("\n💡 Next steps:")
    print("   - Try the advanced features example: portkey_advanced_features.py")
    print("   - View your requests in the Portkey dashboard: https://app.portkey.ai")


if __name__ == "__main__":
    main()
