#!/usr/bin/env python3
"""
Example of using OpenAI Responses API with Langroid.

This example demonstrates:
1. Basic chat completion
2. Streaming responses
3. Tool/function calling
4. Structured JSON output
5. Vision capabilities (if using a vision model)
6. Reasoning models (o1 models)

Run with:
    python examples/basic/openai_responses_example.py
"""

import json
import os
from typing import List

from pydantic import BaseModel, Field

from langroid.language_models import LanguageModel
from langroid.language_models.base import (
    LLMFunctionSpec,
    LLMMessage,
    OpenAIJsonSchemaSpec,
    OpenAIToolCall,
    OpenAIToolSpec,
    Role,
)
from langroid.language_models.openai_responses import OpenAIResponsesConfig
from langroid.parsing.file_attachment import FileAttachment


def example_basic_chat():
    """Basic chat completion example."""
    print("\n=== Basic Chat Example ===")

    config = OpenAIResponsesConfig(
        chat_model="gpt-4o-mini",
        stream=False,
        temperature=0.7,
    )
    llm = LanguageModel.create(config)

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(role=Role.USER, content="What is the capital of France?"),
    ]

    response = llm.chat(messages, max_tokens=50)
    print(f"Response: {response.message}")
    print(f"Tokens used: {response.usage.total_tokens}")
    print(f"Cost: ${response.usage.cost:.4f}")


def example_streaming():
    """Streaming response example."""
    print("\n=== Streaming Example ===")

    # Custom streamer to show tokens as they arrive
    def print_token(token: str):
        print(token, end="", flush=True)

    config = OpenAIResponsesConfig(
        chat_model="gpt-4o-mini",
        stream=True,
        temperature=0.5,
        streamer=print_token,
    )
    llm = LanguageModel.create(config)

    messages = [
        LLMMessage(role=Role.USER, content="Count from 1 to 5 slowly."),
    ]

    print("Streaming: ", end="")
    response = llm.chat(messages, max_tokens=100)
    print()  # New line after streaming
    print(f"\nTotal tokens: {response.usage.total_tokens}")


def example_tool_calling():
    """Tool/function calling example."""
    print("\n=== Tool Calling Example ===")

    # Define a weather tool
    weather_tool = OpenAIToolSpec(
        type="function",
        function={
            "name": "get_weather",
            "description": "Get the weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    )

    config = OpenAIResponsesConfig(
        chat_model="gpt-4o-mini",
        stream=False,
    )
    llm = LanguageModel.create(config)

    messages = [
        LLMMessage(role=Role.USER, content="What's the weather in Paris?"),
    ]

    response = llm.chat(messages, tools=[weather_tool], max_tokens=100)

    if response.oai_tool_calls:
        for tool_call in response.oai_tool_calls:
            print(f"Tool called: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")
    else:
        print(f"Response: {response.message}")


def example_structured_output():
    """Structured JSON output example."""
    print("\n=== Structured Output Example ===")

    # Define a Pydantic model for the output structure
    class PersonInfo(BaseModel):
        name: str = Field(description="Person's full name")
        age: int = Field(description="Person's age in years")
        occupation: str = Field(description="Person's job or profession")
        city: str = Field(description="City where the person lives")

    # Create JSON schema specification
    schema_spec = OpenAIJsonSchemaSpec(
        strict=True,
        function=LLMFunctionSpec(
            name="person_info",
            description="Information about a person",
            parameters=PersonInfo.model_json_schema(),
        ),
    )

    config = OpenAIResponsesConfig(
        chat_model="gpt-4o-mini",
        stream=False,
        temperature=0.3,
    )
    llm = LanguageModel.create(config)

    messages = [
        LLMMessage(
            role=Role.USER,
            content="Generate information for a software engineer named Alice, "
            "age 28, living in San Francisco.",
        ),
    ]

    response = llm.chat(messages, response_format=schema_spec, max_tokens=150)

    # Parse the JSON response
    person_data = json.loads(response.message)
    print(f"Structured output: {json.dumps(person_data, indent=2)}")


def example_vision():
    """Vision model example with image input."""
    print("\n=== Vision Example ===")

    # Create a simple red pixel image as data URI for testing
    red_pixel_data_uri = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    )

    config = OpenAIResponsesConfig(
        chat_model="gpt-4o",  # Vision-capable model
        stream=False,
    )
    llm = LanguageModel.create(config)

    # Create image attachment
    attachment = FileAttachment(
        content=b"",  # Required but not used for data URI
        url=red_pixel_data_uri,
    )

    messages = [
        LLMMessage(
            role=Role.USER,
            content="What color is this image?",
            files=[attachment],
        ),
    ]

    response = llm.chat(messages, max_tokens=50)
    print(f"Vision response: {response.message}")


def example_reasoning_model():
    """Reasoning model (o1) example."""
    print("\n=== Reasoning Model Example ===")

    config = OpenAIResponsesConfig(
        chat_model="o1-mini",  # Reasoning model
        reasoning_effort="medium",  # low, medium, or high
        stream=False,
    )
    llm = LanguageModel.create(config)

    # Note: o1 models don't support system messages
    messages = [
        LLMMessage(
            role=Role.USER,
            content="Solve this step by step: If a train travels 120 km in 2 hours, "
            "and then 180 km in the next 3 hours, what is its average speed?",
        ),
    ]

    response = llm.chat(messages, max_tokens=500)

    if response.reasoning:
        print(f"Reasoning process:\n{response.reasoning}\n")
    print(f"Final answer: {response.message}")


def example_fallback():
    """Example showing fallback from Responses API to Chat Completions."""
    print("\n=== Fallback Example ===")

    config = OpenAIResponsesConfig(
        chat_model="gpt-4o-mini",
        stream=False,
    )
    llm = LanguageModel.create(config)

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are helpful."),
        LLMMessage(role=Role.USER, content="Hello!"),
    ]

    # This will use Responses API if available, otherwise Chat Completions
    response = llm.chat(messages, max_tokens=20)
    print(f"Response (with automatic fallback): {response.message}")

    # Check if cached tokens were used
    if response.usage.cached_tokens > 0:
        print(f"Cached tokens used: {response.usage.cached_tokens}")


def main():
    """Run all examples."""
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    # Run examples
    try:
        example_basic_chat()
        example_streaming()
        example_tool_calling()
        example_structured_output()

        # Vision example requires vision-capable model
        try:
            example_vision()
        except Exception as e:
            print(f"\nVision example skipped: {e}")

        # Reasoning model example requires o1 model access
        try:
            example_reasoning_model()
        except Exception as e:
            print(f"\nReasoning model example skipped: {e}")

        example_fallback()

    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
