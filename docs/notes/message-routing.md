# Message Routing in Multi-Agent Systems

This document covers how messages are routed between agents in Langroid's
multi-agent systems.

## Recommended Approach: Orchestration Tools

The recommended way to route messages between agents is using **orchestration
tools**. These provide explicit, type-safe routing that is easier to debug and
reason about.

### Available Orchestration Tools

Langroid provides several tools in `langroid.agent.tools.orchestration`:

- **`SendTool`** - Send a message to a specific agent by name
- **`DoneTool`** - Signal task completion with a result
- **`PassTool`** - Pass control to another agent
- **`DonePassTool`** - Combine done and pass behaviors
- **`AgentDoneTool`** - Signal completion from a specific agent

Example:

```python
from langroid.agent.tools.orchestration import SendTool

# Enable the tool on your agent
agent.enable_message(SendTool)

# LLM can then use the tool to route messages:
# {"request": "send_message", "to": "AnalysisAgent", "content": "Please analyze this"}
```

**Benefits of tool-based routing:**

- Explicit and predictable behavior
- Type-safe with validation
- Easier to debug (tool calls are logged)
- Works consistently across all LLM providers

## Text-Based Routing (Alternative)

Langroid also supports text-based routing patterns, where the LLM can embed
routing information directly in its response text. This is controlled by the
`recognize_recipient_in_content` setting.

**Note:** While convenient, text-based routing is less explicit than tool-based
routing and may lead to accidental routing if the LLM's response happens to
match the patterns.

### `ChatAgentConfig.recognize_recipient_in_content`

Controls whether recipient routing patterns in LLM response text are parsed.

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig

# Default: recipient patterns are parsed
agent = ChatAgent(ChatAgentConfig(
    recognize_recipient_in_content=True
))

# Disable: patterns treated as plain text
agent = ChatAgent(ChatAgentConfig(
    recognize_recipient_in_content=False
))
```

**Recognized patterns:**

1. **TO-bracket format**: `TO[AgentName]: message content`
2. **JSON format**: `{"recipient": "AgentName", "content": "message"}`

**When `True` (default):**

- Patterns are parsed and recipient is extracted to `ChatDocument.metadata.recipient`
- The pattern prefix/wrapper is stripped from the message content
- Enables LLM-driven routing in multi-agent systems

**When `False`:**

- Patterns are preserved as literal text in the message content
- `metadata.recipient` remains empty
- Useful when you want explicit tool-based routing only

### OpenAI Assistant Support

The `recognize_recipient_in_content` setting is also honored by `OpenAIAssistant`:

```python
from langroid.agent.openai_assistant import OpenAIAssistant, OpenAIAssistantConfig

assistant = OpenAIAssistant(OpenAIAssistantConfig(
    name="MyAssistant",
    recognize_recipient_in_content=False,
))
```

## Related: String Signals for Routing

The `TaskConfig.recognize_string_signals` setting controls parsing of signals
like `DONE`, `PASS`, and `DONE_PASS`. While `DONE` is primarily about task
termination, `PASS` is a routing signal that passes control to another agent.

See [Task Termination - Text-Based Termination Signals](task-termination.md#text-based-termination-signals)
for details on `recognize_string_signals`.

## Disabling All Text-Based Routing

To completely disable text-based routing and rely solely on orchestration tools,
set both flags to `False`:

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.task import Task, TaskConfig

agent = ChatAgent(ChatAgentConfig(
    name="MyAgent",
    recognize_recipient_in_content=False,  # No TO[...] or JSON recipient parsing
))

task = Task(
    agent,
    config=TaskConfig(
        recognize_string_signals=False,  # No DONE/PASS parsing
    ),
)
```

This configuration ensures:

- LLM responses are treated as literal text
- No accidental routing based on text patterns
- All routing must be explicit via orchestration tools
