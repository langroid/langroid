#!/usr/bin/env python3

"""
Multi-provider Portkey example showing how to compare responses across different AI providers.

This example demonstrates:
- Using multiple providers through Portkey
- Comparing response quality and characteristics
- Provider-specific configurations
- Fallback strategies

Run with: python portkey_multi_provider.py
"""

import os
import time
from typing import List, Optional, Tuple

import langroid.language_models as lm
from langroid.language_models.provider_params import PortkeyParams


def check_env_var(var_name: str) -> Optional[str]:
    """Check if environment variable is set and return its value."""
    value = os.getenv(var_name)
    if not value:
        print(f"⚠️  Warning: {var_name} not set in environment")
        return None
    return value


def create_provider_llm(
    provider: str, model: str, portkey_api_key: str, temperature: float = 0.7
) -> Tuple[lm.OpenAIGPT, str]:
    """Create a Portkey-enabled LLM for a specific provider."""
    config = lm.OpenAIGPTConfig(
        chat_model=f"portkey/{provider}/{model}",
        portkey_params=PortkeyParams(
            api_key=portkey_api_key,
            metadata={
                "provider": provider,
                "model": model,
                "demo": "multi-provider-comparison",
            },
            user="multi-provider-demo",
        ),
        max_output_tokens=200,
        temperature=temperature,
    )

    display_name = f"{provider.title()} ({model})"
    return lm.OpenAIGPT(config), display_name


def test_providers_on_question(
    providers: List[Tuple[lm.OpenAIGPT, str]], question: str
):
    """Test all providers on the same question and compare responses."""
    print(f"\n❓ Question: {question}")
    print("=" * 80)

    responses = []

    for llm, display_name in providers:
        print(f"\n🤖 {display_name}:")
        print("-" * 40)

        try:
            start_time = time.time()
            response = llm.chat(question)
            end_time = time.time()

            print(f"📝 Response: {response.message}")
            print(f"⏱️  Time: {end_time - start_time:.2f}s")

            if response.usage:
                print(f"📊 Tokens: {response.usage.total_tokens}")

            responses.append((display_name, response.message, response.usage))

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            responses.append((display_name, f"Error: {str(e)}", None))

    return responses


def demonstrate_creative_tasks(providers: List[Tuple[lm.OpenAIGPT, str]]):
    """Test providers on creative tasks to see different capabilities."""
    print("\n🎨 Creative Tasks Comparison")
    print("=" * 50)

    creative_questions = [
        "Write a haiku about artificial intelligence.",
        "Explain quantum computing using a food analogy.",
        "Create a short story opening with exactly 50 words.",
    ]

    for question in creative_questions:
        test_providers_on_question(providers, question)


def demonstrate_analytical_tasks(providers: List[Tuple[lm.OpenAIGPT, str]]):
    """Test providers on analytical tasks."""
    print("\n🧮 Analytical Tasks Comparison")
    print("=" * 50)

    analytical_questions = [
        "What are the pros and cons of renewable energy?",
        "Explain the causes of inflation in simple terms.",
        "Compare machine learning and traditional programming.",
    ]

    for question in analytical_questions:
        test_providers_on_question(providers, question)


def demonstrate_fallback_strategy(portkey_api_key: str):
    """Demonstrate a simple fallback strategy across providers."""
    print("\n🔄 Fallback Strategy Demo")
    print("=" * 50)

    # Define providers in order of preference
    fallback_providers = [
        ("openai", "gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic", "claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
        ("google", "gemini-2.0-flash-lite", "GOOGLE_API_KEY"),
    ]

    question = "What is the meaning of life in one sentence?"

    for provider, model, env_var in fallback_providers:
        if os.getenv(env_var):
            print(f"\n🎯 Trying {provider.title()}...")
            try:
                llm, display_name = create_provider_llm(
                    provider, model, portkey_api_key, temperature=0.5
                )
                response = llm.chat(question)
                print(f"✅ Success with {display_name}")
                print(f"📝 Response: {response.message}")
                return  # Success, stop trying

            except Exception as e:
                print(f"❌ {display_name} failed: {str(e)}")
                print("🔄 Trying next provider...")
        else:
            print(f"⏭️  Skipping {provider.title()} (API key not available)")

    print("❌ All providers failed!")


def main():
    """Main function demonstrating multi-provider usage."""
    print("🚀 Portkey Multi-Provider Example")
    print("=" * 45)

    # Check for required environment variables
    portkey_api_key = check_env_var("PORTKEY_API_KEY")
    if not portkey_api_key:
        print("❌ PORTKEY_API_KEY is required. Please set it in your environment.")
        return

    print("✅ Portkey API key found")

    # Collect available providers
    providers = []

    if os.getenv("OPENAI_API_KEY"):
        try:
            llm, name = create_provider_llm("openai", "gpt-4o-mini", portkey_api_key)
            providers.append((llm, name))
            print("✅ OpenAI provider ready")
        except Exception as e:
            print(f"⚠️  OpenAI setup failed: {e}")

    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            llm, name = create_provider_llm(
                "anthropic", "claude-3-haiku-20240307", portkey_api_key
            )
            providers.append((llm, name))
            print("✅ Anthropic provider ready")
        except Exception as e:
            print(f"⚠️  Anthropic setup failed: {e}")

    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        try:
            llm, name = create_provider_llm(
                "google", "gemini-2.0-flash-lite", portkey_api_key
            )
            providers.append((llm, name))
            print("✅ Google/Gemini provider ready")
        except Exception as e:
            print(f"⚠️  Google/Gemini setup failed: {e}")

    if len(providers) < 2:
        print("\n⚠️  This example works best with at least 2 providers.")
        print("   Please set API keys for multiple providers:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY or GEMINI_API_KEY")

        if len(providers) == 0:
            print("❌ No providers available. Exiting.")
            return

    print(f"\n🎯 Ready to compare {len(providers)} provider(s)")

    # Run comparisons
    if len(providers) >= 2:
        demonstrate_creative_tasks(providers)
        demonstrate_analytical_tasks(providers)

    # Always demonstrate fallback (works with 1+ providers)
    demonstrate_fallback_strategy(portkey_api_key)

    print("\n🎉 Multi-provider example completed!")
    print("\n💡 Analysis tips:")
    print("   - Different providers may excel at different types of tasks")
    print("   - Response styles and lengths can vary significantly")
    print("   - Use Portkey dashboard to analyze performance metrics")
    print("   - Consider cost, speed, and quality when choosing providers")
    print("\n🔍 View detailed comparisons at: https://app.portkey.ai")


if __name__ == "__main__":
    main()
