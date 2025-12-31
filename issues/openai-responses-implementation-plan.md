# Test-Driven Implementation Plan: OpenAI Responses API Support

This document outlines a stage-by-stage, test-driven implementation plan for adding OpenAI Responses API support to Langroid. Each stage builds incrementally on the previous, with tests written first to guide the implementation.

## Testing Philosophy

- **Real API calls**: Tests use the actual OpenAI API, not mocks (except where specifically justified)
- **Progressive functionality**: Each stage delivers working features that subsequent stages build upon
- **Fail-first approach**: Write tests that fail initially, then implement code to make them pass
- **Meaningful assertions**: Tests verify actual behavior, not just that methods exist

## Environment Setup

### Required Environment Variables
```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_RESPONSES_TEST_MODEL="gpt-4o-mini"  # Cost-effective model for testing
export OPENAI_RESPONSES_TEST_REASONING_MODEL="o1-mini"  # For reasoning tests
```

### Test Markers
```python
# pytest.ini configuration
[pytest]
markers =
    openai_responses: Tests for OpenAI Responses API
    slow: Tests that make real API calls
    stream: Tests for streaming functionality
    tools: Tests for tool/function calling
    vision: Tests requiring vision models
    reasoning: Tests for reasoning models
    cache: Tests for caching functionality
```

## Stage 1: Skeleton and Basic Infrastructure

### Objectives
- Create minimal module structure
- Set up configuration class
- Wire into LanguageModel.create
- Establish test structure

### Tests to Write First
```python
# tests/language_models/test_openai_responses_basic.py

import pytest
from langroid.language_models import LanguageModel
from langroid.language_models.openai_responses import (
    OpenAIResponsesConfig,
    OpenAIResponses,
)

@pytest.mark.openai_responses
class TestBasicInfrastructure:
    def test_config_creation(self):
        """Config can be instantiated with minimal params."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
        assert config.type == "openai_responses"
        assert config.chat_model == "gpt-4o-mini"
    
    def test_provider_creation(self):
        """Provider can be instantiated from config."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
        provider = OpenAIResponses(config)
        assert isinstance(provider, LanguageModel)
    
    def test_language_model_create(self):
        """LanguageModel.create routes to OpenAIResponses."""
        config = OpenAIResponsesConfig(chat_model="gpt-4o-mini")
        llm = LanguageModel.create(config)
        assert isinstance(llm, OpenAIResponses)
```

### Skeleton Code to Create
```python
# langroid/language_models/openai_responses.py

from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.language_models.base import LanguageModel, LLMResponse

class OpenAIResponsesConfig(OpenAIGPTConfig):
    type: str = "openai_responses"

class OpenAIResponses(LanguageModel):
    def __init__(self, config: OpenAIResponsesConfig = OpenAIResponsesConfig()):
        super().__init__(config)
        # Client setup will come in Stage 2
    
    def generate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        raise NotImplementedError("Stage 2")
    
    def chat(self, messages, max_tokens=200, **kwargs) -> LLMResponse:
        raise NotImplementedError("Stage 2")
    
    async def agenerate(self, prompt: str, max_tokens: int = 200) -> LLMResponse:
        raise NotImplementedError("Stage 2")
    
    async def achat(self, messages, max_tokens=200, **kwargs) -> LLMResponse:
        raise NotImplementedError("Stage 2")
```

### Wiring Changes
```python
# langroid/language_models/base.py - Update create method
def create(config: LLMConfig) -> LanguageModel:
    # ... existing cases ...
    elif config.type == "openai_responses":
        from .openai_responses import OpenAIResponses
        return OpenAIResponses(config)
```

## Stage 2: Non-Streaming Text Chat

### Objectives
- Implement basic chat() method
- Set up OpenAI client
- Convert messages to Responses format
- Handle basic text responses

### Tests to Write
```python
# tests/language_models/test_openai_responses_chat.py

@pytest.mark.openai_responses
@pytest.mark.slow
class TestNonStreamingChat:
    def test_simple_text_chat(self):
        """Basic text-only chat returns valid response."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL", "gpt-4o-mini"),
            stream=False,
        )
        llm = OpenAIResponses(config)
        
        messages = [
            LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
            LLMMessage(role=Role.USER, content="Say 'Hello World' and nothing else."),
        ]
        
        response = llm.chat(messages, max_tokens=20)
        
        assert response.message is not None
        assert "Hello World" in response.message
        assert response.usage is not None
        assert response.usage.total_tokens > 0
    
    def test_message_conversion(self):
        """Messages convert correctly to Responses format."""
        # This tests the helper function directly
        from langroid.language_models.openai_responses import messages_to_responses_input
        
        messages = [
            LLMMessage(role=Role.SYSTEM, content="System prompt"),
            LLMMessage(role=Role.USER, content="User message"),
        ]
        
        instructions, input_parts = messages_to_responses_input(messages)
        
        assert instructions == "System prompt"
        assert len(input_parts) == 1
        assert input_parts[0]["type"] == "input_text"
        assert input_parts[0]["text"] == "User message"
```

### Implementation to Add
```python
# langroid/language_models/openai_responses.py

def messages_to_responses_input(messages: List[LLMMessage]) -> Tuple[str, List[Dict]]:
    """Convert Langroid messages to Responses API format."""
    instructions = []
    input_parts = []
    
    for msg in messages:
        if msg.role == Role.SYSTEM:
            instructions.append(msg.content)
        elif msg.role == Role.USER:
            input_parts.append({"type": "input_text", "text": msg.content})
        elif msg.role == Role.ASSISTANT:
            input_parts.append({"type": "output_text", "text": msg.content})
        # Tool results handled in Stage 4
    
    return "\n\n".join(instructions), input_parts

class OpenAIResponses(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        self.client = self._create_client()
    
    def _create_client(self):
        # Use existing client creation logic from openai_gpt.py
        from langroid.client.openai_client import get_openai_client
        return get_openai_client(self.config)
    
    def chat(self, messages, max_tokens=200, **kwargs) -> LLMResponse:
        instructions, input_parts = messages_to_responses_input(messages)
        
        response = self.client.responses.create(
            model=self.config.chat_model,
            instructions=instructions,
            input=input_parts,
            max_output_tokens=max_tokens,
            temperature=self.config.temperature,
        )
        
        # Extract text from output
        message = self._extract_message(response.output)
        usage = self._extract_usage(response.usage)
        
        return LLMResponse(message=message, usage=usage)
```

## Stage 3: Streaming Support

### Objectives
- Implement streaming chat
- Handle stream events
- Aggregate streamed content
- Verify streaming matches non-streaming results

### Tests to Write
```python
# tests/language_models/test_openai_responses_streaming.py

@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.stream
class TestStreaming:
    def test_streaming_aggregates_correctly(self):
        """Streaming produces same result as non-streaming."""
        messages = [
            LLMMessage(role=Role.USER, content="Count from 1 to 5"),
        ]
        
        # Non-streaming
        config_no_stream = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
            stream=False,
        )
        llm_no_stream = OpenAIResponses(config_no_stream)
        response_no_stream = llm_no_stream.chat(messages, max_tokens=50)
        
        # Streaming
        config_stream = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
            stream=True,
        )
        llm_stream = OpenAIResponses(config_stream)
        response_stream = llm_stream.chat(messages, max_tokens=50)
        
        # Should produce similar content (not identical due to randomness)
        assert "1" in response_stream.message
        assert "5" in response_stream.message
        assert response_stream.usage.total_tokens > 0
    
    def test_stream_events_processed(self):
        """Stream events are properly handled."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
            stream=True,
        )
        llm = OpenAIResponses(config)
        
        # Track streaming callbacks
        streamed_chunks = []
        def capture_stream(chunk, event_type):
            streamed_chunks.append((chunk, event_type))
        
        config.streamer = capture_stream
        
        messages = [LLMMessage(role=Role.USER, content="Say 'test'")]
        response = llm.chat(messages, max_tokens=10)
        
        assert len(streamed_chunks) > 0
        assert any("test" in chunk for chunk, _ in streamed_chunks)
```

### Implementation to Add
```python
# langroid/language_models/openai_responses.py

def _process_stream_event(self, event, accumulated_text, tool_deltas):
    """Process a single stream event."""
    if event.type == "response.output_text.delta":
        delta = event.data.delta
        accumulated_text.append(delta)
        if self.config.streamer:
            self.config.streamer(delta, StreamEventType.TEXT)
    
    elif event.type == "response.tool_call.delta":
        # Handle in Stage 4
        pass
    
    elif event.type == "response.completed":
        return event.data.final_response
    
    return None

def _stream_response(self, request_params):
    """Handle streaming response."""
    stream = self.client.responses.stream(**request_params)
    
    accumulated_text = []
    tool_deltas = {}
    
    for event in stream:
        final = self._process_stream_event(event, accumulated_text, tool_deltas)
        if final:
            break
    
    message = "".join(accumulated_text)
    usage = self._extract_usage(final.usage) if final else None
    
    return LLMResponse(message=message, usage=usage)
```

## Stage 4: Tool/Function Calling

### Objectives
- Support tool definitions
- Handle tool calls in responses
- Process tool results in messages
- Implement roundtrip tool interaction

### Tests to Write
```python
# tests/language_models/test_openai_responses_tools.py

@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.tools
class TestToolCalling:
    def test_tool_call_invocation(self):
        """Model correctly calls provided tool."""
        from langroid.agent.tools.orchestration import OpenAIToolSpec
        
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Sunny in {location}"
        
        tool_spec = OpenAIToolSpec.from_function(get_weather)
        
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
            stream=False,
        )
        llm = OpenAIResponses(config)
        
        messages = [
            LLMMessage(role=Role.USER, content="What's the weather in Paris?"),
        ]
        
        response = llm.chat(messages, tools=[tool_spec])
        
        assert response.oai_tool_calls is not None
        assert len(response.oai_tool_calls) > 0
        assert response.oai_tool_calls[0].function.name == "get_weather"
        assert "Paris" in response.oai_tool_calls[0].function.arguments
    
    def test_tool_result_handling(self):
        """Tool results are correctly included in conversation."""
        messages = [
            LLMMessage(role=Role.USER, content="What's the weather?"),
            LLMMessage(
                role=Role.ASSISTANT,
                oai_tool_calls=[
                    OpenAIToolCall(
                        id="call_123",
                        function=OpenAIFunctionCall(
                            name="get_weather",
                            arguments='{"location": "Paris"}',
                        ),
                    )
                ],
            ),
            LLMMessage(
                role=Role.TOOL,
                content="Sunny in Paris",
                tool_call_id="call_123",
            ),
        ]
        
        instructions, input_parts = messages_to_responses_input(messages)
        
        # Find tool result in input_parts
        tool_results = [p for p in input_parts if p["type"] == "tool_result"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_call_id"] == "call_123"
        assert tool_results[0]["output"] == "Sunny in Paris"
```

## Stage 5: Structured Output (JSON Schema)

### Objectives
- Support response_format with JSON schema
- Validate structured outputs
- Handle JSON parsing

### Tests to Write
```python
# tests/language_models/test_openai_responses_structured.py

@pytest.mark.openai_responses
@pytest.mark.slow
class TestStructuredOutput:
    def test_json_schema_response(self):
        """Model returns valid JSON matching schema."""
        from langroid.agent.tools.orchestration import OpenAIJsonSchemaSpec
        from pydantic import BaseModel
        
        class PersonInfo(BaseModel):
            name: str
            age: int
            city: str
        
        schema_spec = OpenAIJsonSchemaSpec.from_pydantic(PersonInfo)
        
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
        )
        llm = OpenAIResponses(config)
        
        messages = [
            LLMMessage(
                role=Role.USER,
                content="Generate info for John, 30 years old, from Paris",
            ),
        ]
        
        response = llm.chat(messages, response_format=schema_spec)
        
        # Response should be valid JSON
        import json
        data = json.loads(response.message)
        
        assert data["name"] == "John"
        assert data["age"] == 30
        assert data["city"] == "Paris"
```

## Stage 6: Reasoning Models

### Objectives
- Support reasoning parameter
- Capture reasoning in responses
- Stream reasoning deltas

### Tests to Write
```python
# tests/language_models/test_openai_responses_reasoning.py

@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.reasoning
@pytest.mark.skipif(
    not os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL"),
    reason="Reasoning model not configured"
)
class TestReasoning:
    def test_reasoning_captured(self):
        """Reasoning is captured for o-models."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_REASONING_MODEL"),
            stream=False,
        )
        config.params.reasoning_effort = "low"
        
        llm = OpenAIResponses(config)
        
        messages = [
            LLMMessage(role=Role.USER, content="What is 2+2? Think step by step."),
        ]
        
        response = llm.chat(messages, max_tokens=100)
        
        assert response.message is not None
        assert "4" in response.message
        assert response.reasoning is not None
        assert len(response.reasoning) > 0
```

## Stage 7: Vision/Multimodal

### Objectives
- Support image inputs
- Handle FileAttachment conversion
- Test with vision models

### Tests to Write
```python
# tests/language_models/test_openai_responses_vision.py

@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.vision
class TestVision:
    def test_image_input(self):
        """Model can process image inputs."""
        from langroid.agent.chat_document import FileAttachment
        
        # Use a small test image (1x1 red pixel as data URI)
        red_pixel = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        
        config = OpenAIResponsesConfig(
            chat_model="gpt-4o",  # Vision-capable model
        )
        llm = OpenAIResponses(config)
        
        attachment = FileAttachment(url=red_pixel)
        messages = [
            LLMMessage(
                role=Role.USER,
                content="What color is this image?",
                attachments=[attachment],
            ),
        ]
        
        response = llm.chat(messages, max_tokens=50)
        
        assert response.message is not None
        assert "red" in response.message.lower()
```

## Stage 8: Usage and Cost Tracking

### Objectives
- Track token usage correctly
- Calculate costs
- Update usage_cost_dict

### Tests to Write
```python
# tests/language_models/test_openai_responses_usage.py

@pytest.mark.openai_responses
@pytest.mark.slow
class TestUsageTracking:
    def test_usage_tracked(self):
        """Token usage is tracked correctly."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
        )
        llm = OpenAIResponses(config)
        
        # Reset usage
        llm.reset_usage_cost()
        
        messages = [
            LLMMessage(role=Role.USER, content="Say 'test'"),
        ]
        
        response = llm.chat(messages, max_tokens=10)
        
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        
        # Check accumulated usage
        summary = llm.usage_cost_summary()
        assert summary["total_tokens"] == response.usage.total_tokens
```

## Stage 9: Caching

### Objectives
- Implement cache key generation
- Cache non-streaming responses
- Cache streaming after completion

### Tests to Write
```python
# tests/language_models/test_openai_responses_cache.py

@pytest.mark.openai_responses
@pytest.mark.cache
class TestCaching:
    def test_cache_hit(self):
        """Identical requests hit cache."""
        from langroid.cachedb import RedisCacheConfig
        
        cache_config = RedisCacheConfig(fake=True)  # In-memory cache for testing
        
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
            cache_config=cache_config,
            stream=False,
        )
        llm = OpenAIResponses(config)
        
        messages = [
            LLMMessage(role=Role.USER, content="Say exactly 'CACHED'"),
        ]
        
        # First call - goes to API
        response1 = llm.chat(messages, max_tokens=10)
        tokens1 = response1.usage.total_tokens
        
        # Second call - should hit cache
        response2 = llm.chat(messages, max_tokens=10)
        tokens2 = response2.usage.total_tokens
        
        assert response1.message == response2.message
        assert tokens2 == 0  # Cached responses don't consume tokens
```

## Stage 10: Error Handling and Retries

### Objectives
- Handle API errors gracefully
- Implement retry logic
- Test timeout handling

### Tests to Write
```python
# tests/language_models/test_openai_responses_errors.py

@pytest.mark.openai_responses
class TestErrorHandling:
    def test_invalid_api_key(self):
        """Invalid API key raises appropriate error."""
        config = OpenAIResponsesConfig(
            chat_model="gpt-4o-mini",
            api_key="invalid_key_for_testing",
        )
        llm = OpenAIResponses(config)
        
        messages = [LLMMessage(role=Role.USER, content="test")]
        
        with pytest.raises(Exception) as exc_info:
            llm.chat(messages)
        
        assert "authentication" in str(exc_info.value).lower()
    
    def test_timeout_handling(self):
        """Timeouts are handled correctly."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
            timeout=0.001,  # Impossibly short timeout
        )
        llm = OpenAIResponses(config)
        
        messages = [LLMMessage(role=Role.USER, content="test")]
        
        with pytest.raises(Exception) as exc_info:
            llm.chat(messages)
        
        assert "timeout" in str(exc_info.value).lower()
```

## Stage 11: Integration with LanguageModel.create

### Objectives
- Ensure proper routing in factory method
- Update __init__.py exports
- Verify config type handling

### Tests to Write
```python
# tests/language_models/test_openai_responses_integration.py

@pytest.mark.openai_responses
class TestIntegration:
    def test_create_factory(self):
        """LanguageModel.create properly routes to OpenAIResponses."""
        config = OpenAIResponsesConfig(
            chat_model=os.getenv("OPENAI_RESPONSES_TEST_MODEL"),
        )
        
        llm = LanguageModel.create(config)
        
        assert isinstance(llm, OpenAIResponses)
        assert llm.config.type == "openai_responses"
    
    def test_import_availability(self):
        """OpenAIResponses is available from package imports."""
        from langroid.language_models import OpenAIResponses, OpenAIResponsesConfig
        
        assert OpenAIResponses is not None
        assert OpenAIResponsesConfig is not None
```

## Stage 12: Examples and Documentation

### Objectives
- Create working examples
- Add docstrings
- Update README

### Examples to Create
```python
# examples/openai_responses/basic_chat.py
# examples/openai_responses/streaming.py
# examples/openai_responses/tools.py
# examples/openai_responses/structured_output.py
# examples/openai_responses/reasoning.py
# examples/openai_responses/vision.py
```

## Running Tests

### Run all Responses API tests
```bash
pytest tests/language_models/test_openai_responses*.py -m openai_responses
```

### Run specific stages
```bash
# Stage 1 (no API calls)
pytest tests/language_models/test_openai_responses_basic.py

# Stage 2 (requires API key)
pytest tests/language_models/test_openai_responses_chat.py -m "openai_responses and slow"
```

### Run with specific model
```bash
OPENAI_RESPONSES_TEST_MODEL=gpt-4.1 pytest tests/language_models/test_openai_responses*.py
```

## Progress Tracking

- [ ] Stage 1: Skeleton and Basic Infrastructure
- [ ] Stage 2: Non-Streaming Text Chat
- [ ] Stage 3: Streaming Support
- [ ] Stage 4: Tool/Function Calling
- [ ] Stage 5: Structured Output
- [ ] Stage 6: Reasoning Models
- [ ] Stage 7: Vision/Multimodal
- [ ] Stage 8: Usage and Cost Tracking
- [ ] Stage 9: Caching
- [ ] Stage 10: Error Handling and Retries
- [ ] Stage 11: Integration
- [ ] Stage 12: Examples and Documentation

## Notes

- Each stage should be completed and tested before moving to the next
- Tests may initially skip with `@pytest.mark.skip("Not implemented")` until the stage is reached
- Use `pytest -xvs` for detailed output during development
- Keep API costs low by using small models and minimal token limits in tests
- Consider rate limiting between tests if hitting API limits