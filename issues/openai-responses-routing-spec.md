# OpenAI Responses API Routing Specification

## Overview
Enable automatic routing to the OpenAI Responses API implementation when appropriate, making it a drop-in replacement for the existing OpenAI Chat Completions implementation while maintaining backward compatibility.

## Current State
- `OpenAIResponses` exists as a separate implementation with `type = "openai_responses"`
- Users must explicitly choose between `OpenAIGPTConfig` and `OpenAIResponsesConfig`
- `ChatAgent` has hardcoded checks for `isinstance(self.llm, OpenAIGPT)`
- No automatic routing based on model or configuration

## Proposed Changes

### 1. Configuration Changes

#### Add Flag to OpenAIGPTConfig
```python
# In langroid/language_models/openai_gpt.py
class OpenAIGPTConfig(LLMConfig):
    # ... existing fields ...
    use_responses_api: bool = False  # New flag, default False for backward compatibility
```

### 2. Factory Method Routing

#### Update LanguageModel.create()
```python
# In langroid/language_models/base.py, around line 490
@staticmethod
def create(config: Optional[LLMConfig]) -> Optional["LanguageModel"]:
    # ... existing code ...
    
    if config.type == "openai":
        from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig
        from langroid.language_models.openai_responses import OpenAIResponses
        
        if isinstance(config, OpenAIGPTConfig) and config.use_responses_api:
            # Route to Responses API implementation
            return OpenAIResponses(config)
        else:
            # Use traditional Chat Completions API
            return OpenAIGPT(config)
    
    # ... rest of existing routing ...
```

### 3. ChatAgent Compatibility Updates

#### Update isinstance Checks
Replace hardcoded `OpenAIGPT` checks with tuple checks:

```python
# In langroid/agent/chat_agent.py

# Line ~308
def _strict_tools_available(self) -> bool:
    from langroid.language_models.openai_gpt import OpenAIGPT
    from langroid.language_models.openai_responses import OpenAIResponses
    
    return (
        not self.disable_strict
        and self.llm is not None
        and isinstance(self.llm, (OpenAIGPT, OpenAIResponses))
        and self.llm.config.parallel_tool_calls is False
        and self.llm.supports_strict_tools
    )

# Line ~318
def _json_schema_available(self) -> bool:
    from langroid.language_models.openai_gpt import OpenAIGPT
    from langroid.language_models.openai_responses import OpenAIResponses
    
    return (
        not self.disable_strict
        and self.llm is not None
        and isinstance(self.llm, (OpenAIGPT, OpenAIResponses))
        and self.llm.supports_json_schema
    )
```

### 4. OpenAIResponses Compatibility

#### Ensure Required Properties
Add missing properties to `OpenAIResponses` class:

```python
# In langroid/language_models/openai_responses.py
class OpenAIResponses(LanguageModel):
    # ... existing code ...
    
    @property
    def supports_strict_tools(self) -> bool:
        """Check if this model supports strict tool schemas."""
        # Check model capabilities
        model = self.config.chat_model.lower()
        return "gpt-4" in model or "gpt-3.5" in model
    
    @property
    def supports_json_schema(self) -> bool:
        """Check if this model supports JSON schema output format."""
        model = self.config.chat_model.lower()
        return "gpt-4" in model or "gpt-3.5" in model
```

#### Config Compatibility
Ensure `OpenAIResponses` can accept `OpenAIGPTConfig`:

```python
def __init__(self, config: Optional[LLMConfig] = None):
    # Accept either OpenAIGPTConfig or OpenAIResponsesConfig
    if config is None:
        config = OpenAIResponsesConfig()
    elif isinstance(config, OpenAIGPTConfig) and not isinstance(config, OpenAIResponsesConfig):
        # Convert OpenAIGPTConfig to OpenAIResponsesConfig
        responses_config = OpenAIResponsesConfig(**config.model_dump())
        config = responses_config
    
    super().__init__(config)
    self.config: OpenAIResponsesConfig = config
    # ... rest of initialization ...
```

## Usage Examples

### Example 1: Explicit Opt-In
```python
from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig

# Opt into Responses API
config = ChatAgentConfig(
    llm=OpenAIGPTConfig(
        chat_model="gpt-4o",
        use_responses_api=True,  # Enable Responses API
    )
)
agent = ChatAgent(config)
# Agent will use OpenAIResponses implementation
```

### Example 2: Traditional Usage (Unchanged)
```python
# Default behavior remains unchanged
config = ChatAgentConfig(
    llm=OpenAIGPTConfig(
        chat_model="gpt-4o",
        # use_responses_api=False by default
    )
)
agent = ChatAgent(config)
# Agent will use OpenAIGPT implementation
```

### Example 3: Future Auto-Detection
```python
# Future enhancement: auto-detect based on model
config = ChatAgentConfig(
    llm=OpenAIGPTConfig(
        chat_model="o1-preview",  # Reasoning model
        # Could auto-set use_responses_api=True for o1 models
    )
)
```

## Migration Path

### Phase 1: Initial Implementation
1. Add `use_responses_api` flag (default False)
2. Update factory routing
3. Update ChatAgent compatibility checks
4. Ensure OpenAIResponses has required properties

### Phase 2: Testing & Validation
1. Test with existing code (should be unchanged with flag=False)
2. Test with flag=True for various models
3. Verify all ChatAgent features work with both implementations

### Phase 3: Gradual Rollout
1. Document the new flag in examples
2. Consider making it default True for new models (o1, future models)
3. Eventually deprecate Chat Completions for models that work better with Responses API

## Testing Requirements

### Backward Compatibility Tests
- Verify existing code works unchanged when `use_responses_api=False`
- Ensure all existing tests pass

### Routing Tests
```python
def test_routing_to_responses_api():
    config = OpenAIGPTConfig(
        chat_model="gpt-4o",
        use_responses_api=True
    )
    llm = LanguageModel.create(config)
    assert isinstance(llm, OpenAIResponses)

def test_routing_to_chat_completions():
    config = OpenAIGPTConfig(
        chat_model="gpt-4o",
        use_responses_api=False
    )
    llm = LanguageModel.create(config)
    assert isinstance(llm, OpenAIGPT)
```

### Feature Parity Tests
- Test that ChatAgent features work with both implementations:
  - Tool calling
  - Structured output
  - Streaming
  - Vision
  - Caching

## Benefits

1. **Seamless Migration**: Users can opt into Responses API with a single flag
2. **Backward Compatible**: No breaking changes for existing code
3. **Future Proof**: Easy to make Responses API the default for new models
4. **Clean Architecture**: No complex inheritance, clear separation of implementations

## Risks & Mitigations

### Risk 1: Feature Differences
- **Risk**: Responses API may not support all Chat Completions features
- **Mitigation**: Fallback mechanism already implemented in OpenAIResponses

### Risk 2: Performance Differences
- **Risk**: Different latency/throughput characteristics
- **Mitigation**: Keep flag optional, let users choose based on needs

### Risk 3: Import Cycles
- **Risk**: Circular imports when updating ChatAgent
- **Mitigation**: Use local imports in methods that need both classes

## Implementation Checklist

- [ ] Add `use_responses_api` flag to OpenAIGPTConfig
- [ ] Update LanguageModel.create() routing logic
- [ ] Update ChatAgent._strict_tools_available() check
- [ ] Update ChatAgent._json_schema_available() check
- [ ] Add supports_strict_tools property to OpenAIResponses
- [ ] Add supports_json_schema property to OpenAIResponses
- [ ] Ensure OpenAIResponses accepts OpenAIGPTConfig
- [ ] Add routing tests
- [ ] Add feature parity tests
- [ ] Update documentation with examples
- [ ] Test with real ChatAgent usage