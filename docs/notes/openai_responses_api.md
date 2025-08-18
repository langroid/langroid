# OpenAI Responses API Support in Langroid

Langroid now supports OpenAI's new Responses API, providing a modern alternative
to the Chat Completions API with enhanced features and better performance.

## Overview

The Responses API is OpenAI's next-generation API that offers:
- Improved streaming capabilities
- Better support for reasoning models (o1 series)
- Enhanced caching for reduced costs
- Cleaner API design with explicit input/output structure

## Quick Start

```python
from langroid.language_models import LanguageModel
from langroid.language_models.openai_responses import OpenAIResponsesConfig
from langroid.language_models.base import LLMMessage, Role

# Create configuration
config = OpenAIResponsesConfig(
    chat_model="gpt-4o-mini",
    stream=True,
    temperature=0.7,
)

# Create language model instance
llm = LanguageModel.create(config)

# Send messages
messages = [
    LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
    LLMMessage(role=Role.USER, content="Hello! How are you?"),
]

response = llm.chat(messages, max_tokens=100)
print(response.message)
```

## Features

### 1. Basic Chat Completion

The Responses API seamlessly integrates with Langroid's existing chat interface:

```python
config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
llm = LanguageModel.create(config)

response = llm.chat("What is the capital of France?", max_tokens=50)
print(response.message)  # "The capital of France is Paris."
```

### 2. Streaming Support

Stream responses token by token for real-time output:

```python
def print_token(token: str):
    print(token, end="", flush=True)

config = OpenAIResponsesConfig(
    chat_model="gpt-4o-mini",
    stream=True,
    streamer=print_token,  # Custom callback for each token
)
llm = LanguageModel.create(config)

response = llm.chat("Tell me a story", max_tokens=200)
```

### 3. Tool/Function Calling

Support for OpenAI's function calling feature:

```python
from langroid.language_models.base import OpenAIToolSpec

# Define a tool
weather_tool = OpenAIToolSpec(
    type="function",
    function={
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    },
)

response = llm.chat(
    "What's the weather in Paris?",
    tools=[weather_tool],
    max_tokens=100,
)

if response.oai_tool_calls:
    for tool_call in response.oai_tool_calls:
        print(f"Tool: {tool_call.function.name}")
        print(f"Args: {tool_call.function.arguments}")
```

### 4. Structured Output (JSON Schema)

Generate responses that conform to a specific JSON schema:

```python
from pydantic import BaseModel, Field
from langroid.language_models.base import LLMFunctionSpec, OpenAIJsonSchemaSpec

class PersonInfo(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    city: str = Field(description="Person's city")

schema_spec = OpenAIJsonSchemaSpec(
    strict=True,
    function=LLMFunctionSpec(
        name="person_info",
        description="Person information",
        parameters=PersonInfo.model_json_schema(),
    ),
)

response = llm.chat(
    "Generate info for John, 30 years old, from Paris",
    response_format=schema_spec,
    max_tokens=100,
)

import json
data = json.loads(response.message)
print(data)  # {"name": "John", "age": 30, "city": "Paris"}
```

### 5. Vision/Multimodal Support

Process images alongside text (requires vision-capable models like gpt-4o):

```python
from langroid.parsing.file_attachment import FileAttachment

# Create image attachment
image_url = "https://example.com/image.jpg"
attachment = FileAttachment(content=b"", url=image_url)

messages = [
    LLMMessage(
        role=Role.USER,
        content="What's in this image?",
        files=[attachment],
    ),
]

response = llm.chat(messages, max_tokens=200)
print(response.message)
```

### 6. Reasoning Models (o1 Series)

Special support for OpenAI's o1 reasoning models:

```python
config = OpenAIResponsesConfig(
    chat_model="o1-mini",
    reasoning_effort="medium",  # low, medium, or high
)
llm = LanguageModel.create(config)

# Note: o1 models don't support system messages
messages = [
    LLMMessage(
        role=Role.USER,
        content="Solve this complex problem step by step...",
    ),
]

response = llm.chat(messages, max_tokens=500)

# Access reasoning process
if response.reasoning:
    print(f"Reasoning: {response.reasoning}")
print(f"Answer: {response.message}")
```

### 7. Usage Tracking and Cost Calculation

Automatic tracking of token usage and costs:

```python
response = llm.chat("Hello!", max_tokens=50)

print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
print(f"Cached tokens: {response.usage.cached_tokens}")
print(f"Total cost: ${response.usage.cost:.4f}")
```

### 8. Automatic Fallback

The implementation automatically falls back to the Chat Completions API
if the Responses API is unavailable:

```python
# This works transparently whether Responses API is available or not
config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
llm = LanguageModel.create(config)
response = llm.chat("Hello!", max_tokens=50)
```

## Configuration

The `OpenAIResponsesConfig` class extends `OpenAIGPTConfig` with additional
options:

```python
config = OpenAIResponsesConfig(
    # Standard OpenAI configuration
    chat_model="gpt-4o-mini",
    api_key="your-api-key",  # Or use OPENAI_API_KEY env var
    api_base="https://api.openai.com/v1",
    temperature=0.7,
    stream=False,
    
    # Responses API specific
    reasoning_effort="medium",  # For o1 models: low, medium, high
    
    # Retry configuration
    retry_params=RetryParams(
        max_retries=3,
        initial_delay=1.0,
        exponential_base=2.0,
        jitter=True,
    ),
)
```

## Error Handling

The implementation includes robust error handling with automatic retries:

- **Authentication errors**: Clear error messages for invalid API keys
- **Rate limiting**: Automatic retry with exponential backoff
- **Network errors**: Graceful handling of connection issues
- **Timeout handling**: Configurable timeouts with proper error reporting
- **Invalid models**: Helpful error messages for non-existent models

## Testing

Run the test suite to verify the implementation:

```bash
# Run all Responses API tests
pytest tests/language_models/test_openai_responses_*.py -v

# Run specific test categories
pytest tests/language_models/test_openai_responses_basic.py -v
pytest tests/language_models/test_openai_responses_streaming.py -v
pytest tests/language_models/test_openai_responses_tools.py -v
pytest tests/language_models/test_openai_responses_structured.py -v
pytest tests/language_models/test_openai_responses_reasoning.py -v
pytest tests/language_models/test_openai_responses_vision.py -v
pytest tests/language_models/test_openai_responses_usage.py -v
pytest tests/language_models/test_openai_responses_errors.py -v
pytest tests/language_models/test_openai_responses_integration.py -v
```

## Examples

See `examples/basic/openai_responses_example.py` for comprehensive examples
of all features.

## Migration from Chat Completions

Migrating from the existing OpenAI implementation is straightforward:

```python
# Old (Chat Completions)
from langroid.language_models.openai_gpt import OpenAIGPTConfig

config = OpenAIGPTConfig(chat_model="gpt-4o-mini")

# New (Responses API)
from langroid.language_models.openai_responses import OpenAIResponsesConfig

config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
```

The rest of your code remains the same - the Responses API implementation
maintains full compatibility with Langroid's `LanguageModel` interface.

## Limitations

- The Responses API is currently in beta and may have limited availability
- Some older models may not support all Responses API features
- Automatic fallback to Chat Completions API ensures compatibility

## See Also

- [OpenAI Responses API Documentation](https://platform.openai.com/docs/api-reference/responses)
- [Langroid Language Models Documentation](https://langroid.github.io/langroid/tutorials/language_models/)
- [Example Script](../examples/basic/openai_responses_example.py)