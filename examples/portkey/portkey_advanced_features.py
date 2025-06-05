#!/usr/bin/env python3

"""
Advanced Portkey example showing observability, caching, retries, and metadata.

This example demonstrates:
- Advanced Portkey configuration with all features
- Request tracing and metadata
- Caching and retry strategies
- Custom headers for observability

Run with: python portkey_advanced_features.py
"""

import os
import uuid
from typing import Optional

import langroid as lr
import langroid.language_models as lm
from langroid.language_models.provider_params import PortkeyParams


def check_env_var(var_name: str) -> Optional[str]:
    """Check if environment variable is set and return its value."""
    value = os.getenv(var_name)
    if not value:
        print(f"⚠️  Warning: {var_name} not set in environment")
        return None
    return value


def create_advanced_portkey_llm(portkey_api_key: str, user_id: str) -> lm.OpenAIGPT:
    """Create an advanced Portkey-enabled LLM with all features."""

    # Generate unique trace ID for this session
    trace_id = f"trace-{uuid.uuid4().hex[:8]}"

    config = lm.OpenAIGPTConfig(
        chat_model="portkey/openai/gpt-4o-mini",
        portkey_params=PortkeyParams(
            api_key=portkey_api_key,
            # Observability features
            trace_id=trace_id,
            metadata={
                "user_id": user_id,
                "app": "langroid-advanced-example",
                "version": "1.0",
                "environment": "demo",
            },
            # Retry configuration
            retry={"max_retries": 3, "backoff": "exponential", "jitter": True},
            # Caching configuration
            cache={
                "enabled": True,
                "ttl": 3600,  # 1 hour
                "namespace": "langroid-demo",
            },
            cache_force_refresh=False,
            # User tracking
            user=user_id,
            organization="langroid-demo-org",
            # Custom headers for additional tracking
            custom_headers={
                "x-session-id": f"session-{uuid.uuid4().hex[:8]}",
                "x-demo-type": "advanced-features",
                "x-langroid-version": (
                    lr.__version__ if hasattr(lr, "__version__") else "unknown"
                ),
            },
        ),
        max_output_tokens=200,
        temperature=0.3,
    )

    return lm.OpenAIGPT(config)


def demonstrate_caching(llm: lm.OpenAIGPT):
    """Demonstrate Portkey's caching capabilities."""
    print("\n🧠 Testing Caching Capabilities")
    print("=" * 50)

    question = "What are the three laws of robotics by Isaac Asimov?"

    print("🔄 First request (should hit the API)...")
    response1 = llm.chat(question)
    print(f"✅ Response: {response1.message[:100]}...")
    print(f"📊 Cached: {response1.cached}")
    if response1.usage:
        print(f"📊 Tokens: {response1.usage.total_tokens}")

    print("\n🔄 Second identical request (should hit cache)...")
    response2 = llm.chat(question)
    print(f"✅ Response: {response2.message[:100]}...")
    print(f"📊 Cached: {response2.cached}")
    if response2.usage:
        print(f"📊 Tokens: {response2.usage.total_tokens}")


def demonstrate_metadata_tracking(llm: lm.OpenAIGPT, user_id: str):
    """Demonstrate request tracking with metadata."""
    print("\n📊 Testing Metadata and Tracking")
    print("=" * 50)

    questions = [
        "What is machine learning?",
        "Explain neural networks briefly.",
        "What is the difference between AI and ML?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n🔍 Question {i}: {question}")

        # Create a new LLM instance with updated metadata for each question
        trace_id = f"trace-q{i}-{uuid.uuid4().hex[:6]}"

        config = lm.OpenAIGPTConfig(
            chat_model="portkey/openai/gpt-4o-mini",
            portkey_params=PortkeyParams(
                api_key=llm.config.portkey_params.api_key,
                trace_id=trace_id,
                metadata={
                    "user_id": user_id,
                    "question_number": i,
                    "question_category": "AI/ML basics",
                    "timestamp": str(uuid.uuid4()),  # Unique per request
                },
                user=user_id,
                custom_headers={
                    "x-question-id": f"q-{i}",
                    "x-session-type": "educational",
                },
            ),
            max_output_tokens=150,
            temperature=0.3,
        )

        question_llm = lm.OpenAIGPT(config)
        response = question_llm.chat(question)

        print(f"✅ Response: {response.message[:80]}...")
        print(f"🏷️  Trace ID: {trace_id}")


def demonstrate_error_handling():
    """Demonstrate error handling and retry behavior."""
    print("\n⚠️  Testing Error Handling")
    print("=" * 50)

    try:
        # Create config with invalid model to test error handling
        config = lm.OpenAIGPTConfig(
            chat_model="portkey/openai/invalid-model-name",
            portkey_params=PortkeyParams(
                api_key=os.getenv("PORTKEY_API_KEY", ""),
                retry={"max_retries": 2, "backoff": "linear"},
                metadata={"test_type": "error_handling"},
            ),
        )

        error_llm = lm.OpenAIGPT(config)
        response = error_llm.chat("This should fail")
        print(f"Unexpected success: {response.message}")

    except Exception as e:
        print(f"✅ Expected error caught: {type(e).__name__}")
        print(f"   Error details: {str(e)[:100]}...")


def main():
    """Main function demonstrating advanced Portkey features."""
    print("🚀 Portkey Advanced Features Example")
    print("=" * 45)

    # Check for required environment variables
    portkey_api_key = check_env_var("PORTKEY_API_KEY")
    if not portkey_api_key:
        print("❌ PORTKEY_API_KEY is required. Please set it in your environment.")
        return

    openai_api_key = check_env_var("OPENAI_API_KEY")
    if not openai_api_key:
        print("❌ OPENAI_API_KEY is required for this example.")
        return

    print("✅ All required API keys found")

    # Generate a unique user ID for this session
    user_id = f"user-{uuid.uuid4().hex[:8]}"
    print(f"🆔 Demo User ID: {user_id}")

    # Create advanced LLM configuration
    try:
        llm = create_advanced_portkey_llm(portkey_api_key, user_id)
        print("✅ Advanced Portkey LLM created successfully")

        # Demonstrate different features
        demonstrate_caching(llm)
        demonstrate_metadata_tracking(llm, user_id)
        demonstrate_error_handling()

    except Exception as e:
        print(f"❌ Failed to create advanced LLM: {str(e)}")
        return

    print("\n🎉 Advanced Portkey features example completed!")
    print("\n💡 Next steps:")
    print(
        "   - View detailed request logs in Portkey dashboard: https://app.portkey.ai"
    )
    print(
        "   - Filter by trace IDs, user IDs, or metadata to analyze specific requests"
    )
    print("   - Try the multi-provider example: portkey_multi_provider.py")
    print(f"\n🔍 Your demo user ID: {user_id}")
    print("   Use this to filter requests in the Portkey dashboard")


if __name__ == "__main__":
    main()
