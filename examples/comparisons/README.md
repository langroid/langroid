# Framework Comparisons

This folder contains side-by-side examples comparing Langroid with other agent frameworks, demonstrating equivalent functionality to enable objective comparison.

## Structured Extraction: Langroid vs Google ADK

Both examples extract structured person information (name, age, occupation, location) from text passages. They demonstrate how each framework handles:

1. **Tool definition** - How to create a structured output tool
2. **Task termination** - How to control when the agent stops
3. **LLM reliability** - How to handle the LLM forgetting to use tools

### Files

| File | Framework | Lines of Code |
|------|-----------|---------------|
| `structured_extraction_langroid.py` | Langroid | ~90 lines |
| `structured_extraction_google_adk.py` | Google ADK | ~180 lines |

### Running the Examples

**Langroid version:**
```bash
# Requires OPENAI_API_KEY
python examples/comparisons/structured_extraction_langroid.py
```

**Google ADK version:**
```bash
# Install google-adk (included in dev dependencies)
pip install google-adk

# Requires GOOGLE_API_KEY for Gemini
export GOOGLE_API_KEY="your-key-here"
python examples/comparisons/structured_extraction_google_adk.py
```

---

## Feature Comparison

### 1. Tool Definition

| Aspect | Langroid | Google ADK |
|--------|----------|------------|
| **Approach** | `ToolMessage` class with Pydantic | Plain Python function |
| **Validation** | Built-in via Pydantic fields | Inferred from type hints |
| **Few-shot examples** | `examples()` classmethod | Not supported |
| **Custom handling** | `handle()` method | Return value only |

**Langroid:**
```python
class PersonInfo(lr.ToolMessage):
    request: str = "person_info"
    name: str = Field(..., description="The person's full name")
    age: int = Field(..., description="Age in years")

    def handle(self) -> ResultTool:
        return ResultTool(person=self.dict())

    @classmethod
    def examples(cls):
        return [cls(name="Jane", age=30, ...)]
```

**Google ADK:**
```python
def extract_person_info(
    name: str,
    age: int,
    occupation: str,
    location: str,
) -> dict:
    """Docstring provides descriptions for the schema."""
    return {"name": name, "age": age, ...}
```

**Assessment:** Langroid's approach is more verbose but provides richer metadata (descriptions, examples, custom handlers). Google ADK is simpler for basic cases.

---

### 2. Task Termination

| Aspect | Langroid | Google ADK |
|--------|----------|------------|
| **Control** | Explicit via config | Implicit |
| **Options** | `done_if_tool`, `done_sequences`, `single_round` | None built-in |
| **Pattern matching** | DSL: `"T, A, L"`, `"T[ToolName], A"` | Not available |

**Langroid:**
```python
task_config = lr.TaskConfig(
    done_if_tool=True,  # Terminate when any tool is called
)
# Or use pattern matching:
task_config = lr.TaskConfig(
    done_sequences=["T[PersonInfo], A, L"],  # Tool -> Agent -> LLM response
)
```

**Google ADK:**
```python
# No configuration needed - termination is implicit
# Task ends when is_final_response() returns True:
# - No pending function calls
# - No pending function responses
# - Response is complete (not partial/streaming)
```

**Assessment:** Langroid provides fine-grained control over termination. Google ADK's implicit model is simpler but less flexible—the task ends whenever the LLM stops calling tools.

---

### 3. Handling LLM Forgetting to Use Tools

This is the key differentiator. LLMs sometimes respond with plain text instead of using the required tool.

| Aspect | Langroid | Google ADK |
|--------|----------|------------|
| **Built-in** | Yes | No |
| **Configuration** | 1 line | ~110 lines of custom code |
| **Retry logic** | Automatic | Application-level |

**Langroid (1 line):**
```python
config = lr.ChatAgentConfig(
    handle_llm_no_tool="You MUST use the extract_person_info tool!",
)
# Framework automatically:
# 1. Detects when LLM responds without a tool
# 2. Sends the nudge message back to the LLM
# 3. Retries until tool is used or max attempts reached
```

**Google ADK (~110 lines):**
```python
# Step 1: Helper function to detect function calls (~20 lines)
def has_function_call(llm_response: Any) -> bool:
    try:
        if hasattr(llm_response, "candidates"):
            for candidate in llm_response.candidates:
                # ... nested structure traversal
        if hasattr(llm_response, "content"):
            # ... alternative pattern
    except Exception:
        pass
    return False

# Step 2: Callback class for detection (~40 lines)
class ToolEnforcementCallback:
    def __init__(self, nudge_message, max_retries):
        self.nudge_message = nudge_message
        self.max_retries = max_retries

    async def __call__(self, callback_context, llm_response):
        state = callback_context.state
        if has_function_call(llm_response):
            state["_tool_was_used"] = True
            return None
        # Set flags for application-level retry...
        state["_needs_tool_retry"] = True
        return None  # Cannot trigger retry from callback

# Step 3: Application-level retry loop (~50 lines)
async def extract_with_retry(text, max_retries=3):
    for attempt in range(max_retries + 1):
        session = await session_service.create_session(...)
        if attempt > 0:
            message_text = f"REMINDER: Use the tool!\n{text}"
        async for event in runner.run_async(...):
            # Check if tool was used...
        if tool_was_used:
            return result
    return None
```

**Assessment:** Langroid provides this as a built-in feature because it's a common real-world need. Google ADK requires developers to implement the detection, state management, and retry logic themselves.

---

## Summary Table

| Feature | Langroid | Google ADK |
|---------|----------|------------|
| **Tool definition** | Class-based with Pydantic | Function-based |
| **Termination control** | Explicit, pattern-based | Implicit |
| **LLM forgot tool handling** | Built-in (1 line) | Manual (~110 lines) |
| **Few-shot examples** | Supported | Not supported |
| **Learning curve** | Moderate | Lower for basics |
| **Production robustness** | Higher (more guardrails) | Requires more custom code |

---

## When to Use Which

**Choose Langroid when:**
- You need reliable structured output extraction
- LLM reliability is critical (production systems)
- You want fine-grained termination control
- You need few-shot examples in tool definitions

**Choose Google ADK when:**
- You want minimal boilerplate for simple cases
- You're already in the Google/Gemini ecosystem
- You prefer function-based tool definitions
- You're comfortable implementing custom retry logic

---

## References

- [Langroid Documentation](https://langroid.github.io/langroid/)
- [Google ADK Documentation](https://google.github.io/adk-docs/)
- [Langroid handle_llm_no_tool docs](https://langroid.github.io/langroid/notes/handle-llm-no-tool/)
- [Google ADK Callbacks](https://google.github.io/adk-docs/callbacks/)
