# Pattern: Stateful Tool Handler (on Agent)

## Problem

Your tool handler needs access to agent state, external resources, or needs
to track information across multiple calls.

## Solution

Define a handler method on the agent class. The method name must match the
tool's `request` field. The handler receives the tool message and can access
`self` for state.

## Complete Code Example

```python
import langroid as lr
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.pydantic_v1 import Field


class AddItemTool(ToolMessage):
    """Tool to add an item to the collection."""
    request: str = "add_item"
    purpose: str = "Add an item to the collection"
    item: str = Field(..., description="Item to add")


class GetTotalTool(ToolMessage):
    """Tool to get total count and finish."""
    request: str = "get_total"
    purpose: str = "Get total items and finish"


class CollectorAgent(lr.ChatAgent):
    """Agent that collects items and tracks count."""

    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        self.items: list[str] = []
        self.max_items: int = 10

    def init_state(self) -> None:
        """Reset between task runs."""
        super().init_state()
        self.items = []

    def add_item(self, msg: AddItemTool) -> str:
        """
        Handler for AddItemTool. Method name = request field.

        Returns str so LLM continues adding items.
        """
        if len(self.items) >= self.max_items:
            return f"ERROR: Maximum {self.max_items} items allowed. Use get_total to finish."

        if msg.item in self.items:
            return f"Item '{msg.item}' already exists. Add a different item."

        self.items.append(msg.item)
        remaining = self.max_items - len(self.items)
        return f"Added '{msg.item}'. Total: {len(self.items)}. Can add {remaining} more."

    def get_total(self, msg: GetTotalTool) -> AgentDoneTool:
        """
        Handler for GetTotalTool.

        Returns AgentDoneTool to terminate task.
        """
        return AgentDoneTool(
            content=f"Collection complete: {len(self.items)} items: {self.items}"
        )


def run_collector() -> str:
    config = lr.ChatAgentConfig(
        name="Collector",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4o"),
        system_message="""
        You collect items from the user. Use add_item to add each item.
        When done collecting, use get_total to finish.
        """,
    )

    agent = CollectorAgent(config)
    agent.enable_message([AddItemTool, GetTotalTool])

    # Use done_sequences to exit on GetTotalTool
    task = lr.Task(
        agent,
        interactive=True,
        config=lr.TaskConfig(done_sequences=["T[GetTotalTool], A"]),
    )

    result = task.run()
    return result.content if result else ""
```

## Handler Method Signatures

```python
# Minimal - just the tool message
def add_item(self, msg: AddItemTool) -> str:
    ...

# With ChatDocument for metadata
def add_item(self, msg: AddItemTool, chat_doc: ChatDocument) -> str:
    ...

# Async handler (preferred for I/O operations)
async def add_item_async(self, msg: AddItemTool) -> str:
    result = await some_async_operation()
    return result
```

## Return Values

| Return | Effect |
|--------|--------|
| `str` | Message goes to LLM, conversation continues |
| `AgentDoneTool(content=...)` | Task terminates with content |
| `AgentDoneTool(tools=[msg])` | Task terminates with tool as result |
| Another `ToolMessage` | That tool is processed next |

## Key Points

- Method name MUST match tool's `request` field exactly
- Handler has full access to agent state via `self`
- Return `str` for retry/feedback, `AgentDoneTool` for success
- Use `init_state()` to reset state between task runs
- Async handlers use `_async` suffix: `add_item_async`

## When to Use

- Handler needs agent state (counters, accumulators, context)
- External I/O operations (API calls, database, files)
- Complex validation against prior input
- Tracking state across multiple tool calls
