# Framework Comparisons

This folder contains side-by-side examples comparing Langroid with other agent frameworks.

## Structured Extraction: Langroid vs Google ADK

These examples demonstrate extracting structured information (person details) from text passages.

### Files

| File | Framework | Description |
|------|-----------|-------------|
| `structured_extraction_langroid.py` | Langroid | Uses `ToolMessage`, `handle_llm_no_tool`, `done_if_tool` |
| `structured_extraction_google_adk.py` | Google ADK | Uses function tools, implicit termination |

### Running the Examples

**Langroid version:**
```bash
# Requires OPENAI_API_KEY
python examples/comparisons/structured_extraction_langroid.py
```

**Google ADK version:**
```bash
# First install google-adk
pip install google-adk

# Requires GOOGLE_API_KEY for Gemini
export GOOGLE_API_KEY="your-key-here"
python examples/comparisons/structured_extraction_google_adk.py
```

### Key Differences Illustrated

#### 1. Task Termination

| Aspect | Langroid | Google ADK |
|--------|----------|------------|
| **Control** | Explicit via `done_if_tool=True`, `done_sequences` | Implicit (ends when LLM stops calling tools) |
| **Flexibility** | Pattern-based DSL (`"T, A, L"`) | None - just "no more tools = done" |
| **Risk** | Low - you define exactly when to stop | LLM forgetting tool = premature exit |

**Langroid:**
```python
task_config = lr.TaskConfig(
    done_if_tool=True,  # Explicit: terminate when tool is called
)
```

**Google ADK:**
```python
# No equivalent - termination is implicit
# Task ends when is_final_response() returns True, which happens
# when there are no function_calls in the LLM response
```

#### 2. Handling LLM Forgetting to Use Tools

| Aspect | Langroid | Google ADK |
|--------|----------|------------|
| **Built-in** | Yes - `handle_llm_no_tool` config | No |
| **Options** | String nudge, callable, "done", "user" | Must write custom callback |
| **Effort** | One line of config | 20+ lines of boilerplate |

**Langroid:**
```python
agent_config = lr.ChatAgentConfig(
    handle_llm_no_tool="You FORGOT to use the tool! Try again.",
)
```

**Google ADK:**
```python
# Must implement a custom after_model_callback:
def check_for_tool_usage(callback_context, llm_response):
    # Check if response has function calls...
    # If not, somehow inject a retry (non-trivial)
    pass

agent = LlmAgent(
    after_model_callback=check_for_tool_usage,
)
```

#### 3. Tool Definition

| Aspect | Langroid | Google ADK |
|--------|----------|------------|
| **Structure** | `ToolMessage` class with Pydantic | Plain Python function |
| **Validation** | Built-in via Pydantic | Minimal |
| **Examples** | `examples()` classmethod | Not supported |
| **Handlers** | `handle()` method for custom logic | Return value only |

**Langroid:**
```python
class PersonInfo(lr.ToolMessage):
    request: str = "person_info"
    name: str = Field(..., description="The person's full name")

    def handle(self) -> ResultTool:
        return ResultTool(person=self.dict())

    @classmethod
    def examples(cls):
        return [cls(name="Jane", ...)]
```

**Google ADK:**
```python
def extract_person_info(name: str, age: int, ...) -> dict:
    return {"name": name, "age": age, ...}
```

### Summary

| Feature | Langroid | Google ADK |
|---------|----------|------------|
| Explicit termination control | Yes | No |
| LLM forgot tool handling | Built-in | DIY |
| Pattern-based done sequences | Yes | No |
| Tool validation | Pydantic | Basic type hints |
| Few-shot examples in tools | Yes | No |
| Learning curve | Moderate | Lower |

Google ADK is simpler but gives you less control. Langroid requires more upfront learning but provides robust guardrails for production use cases where LLM reliability matters.
