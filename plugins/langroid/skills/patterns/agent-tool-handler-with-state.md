# Stateful Tool Handler as Agent Method

## The Pattern

Instead of defining a `handle()` method inside the `ToolMessage` class, define a
method on the **agent** with the same name as the tool's `request` field. This
gives the handler access to agent state and resources.

## When to Use

- Handler needs to execute external operations (API calls, DB queries, shell cmds)
- Need to track state across retries (e.g., failure counter to limit retries)
- Handler needs access to agent-level resources (connections, configs, caches)
- Want Langroid's automatic retry loop: errors go back to LLM for self-correction

## Key Concepts

1. **Method name = `request` field**: If `request = "my_tool"`, define
   `def my_tool(self, msg: MyToolMessage)`

2. **Return types control flow**:
   - Return `str` (especially error messages) -> Langroid sends to LLM, can retry
   - Return `DoneTool(content="result")` -> Task terminates with this result

3. **State in `init_state()`**: Override `init_state()` to reset counters/state
   between uses. Called by `task.reset_all_sub_tasks()`.

## Example: Query Executor with Retry Limit

```python
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.orchestration import DoneTool
from pydantic import Field
from typing import Union


class QueryTool(ToolMessage):
    """Tool for LLM to emit a query."""
    request: str = "execute_query"
    purpose: str = "Execute a database query"

    query: str = Field(..., description="The SQL query to execute")


class QueryExecutorAgent(ChatAgent):
    """Agent that executes queries with retry limiting."""

    def __init__(self, config: ChatAgentConfig, db_connection, max_retries: int = 3):
        super().__init__(config)
        self.db_connection = db_connection
        self.max_retries = max_retries
        self.failure_count = 0

    def init_state(self):
        """Reset state between tasks. Called by task.reset_all_sub_tasks()."""
        super().init_state()
        self.failure_count = 0

    def execute_query(self, msg: QueryTool) -> Union[str, DoneTool]:
        """Handler for QueryTool. Name matches request field."""
        try:
            result = self.db_connection.execute(msg.query)
            # Success - terminate task with result
            return DoneTool(content=str(result))

        except Exception as e:
            self.failure_count += 1

            if self.failure_count >= self.max_retries:
                # Give up after max retries
                return DoneTool(content="")  # Empty = failure

            # Return error string - Langroid sends to LLM for retry
            return f"Query failed with error: {e}\nPlease fix and try again."


# Usage
config = ChatAgentConfig(
    name="QueryAgent",
    system_message="You execute SQL queries. Use the execute_query tool.",
)
agent = QueryExecutorAgent(config, db_connection=my_db, max_retries=3)
agent.enable_message([QueryTool])

task = lr.Task(agent, interactive=False)
result = task.run("Run a query to get all users")
# result.content will be query results or empty string on failure
```

## Example: External API with Validation

```python
class APICallTool(ToolMessage):
    request: str = "call_api"
    purpose: str = "Call an external API endpoint"

    endpoint: str = Field(..., description="API endpoint path")
    payload: dict = Field(default_factory=dict, description="Request payload")


class APIAgent(ChatAgent):
    def __init__(self, config, api_client):
        super().__init__(config)
        self.api_client = api_client
        self.call_count = 0

    def init_state(self):
        super().init_state()
        self.call_count = 0

    def call_api(self, msg: APICallTool) -> Union[str, DoneTool]:
        """Handler matches 'call_api' request field."""
        # Validate before calling
        if not msg.endpoint.startswith("/"):
            return "Error: endpoint must start with '/'. Please fix."

        try:
            response = self.api_client.post(msg.endpoint, json=msg.payload)

            if response.status_code != 200:
                return f"API returned {response.status_code}: {response.text}"

            self.call_count += 1
            return DoneTool(content=response.json())

        except Exception as e:
            return f"API call failed: {e}. Check endpoint and payload."
```

## Integration with Batch Processing

When using `run_batch_tasks()`, each item gets a cloned agent with fresh state:

```python
from langroid.agent.batch import run_batch_tasks

base_task = lr.Task(agent, interactive=False)

# Each item gets a cloned agent - no state leakage between items
results = run_batch_tasks(
    base_task,
    items=["query1", "query2", "query3"],
    input_map=lambda q: f"Execute: {q}",
    output_map=lambda r: r.content if r else None,
    sequential=False,  # Run in parallel
    batch_size=10,
)
```

## Important Notes

1. The handler method receives the parsed `ToolMessage` object, not raw JSON
2. Langroid automatically deserializes the LLM's tool call into the ToolMessage
3. If handler returns a string, Langroid treats it as a response and continues
   the conversation (LLM sees it, can emit another tool call)
4. `DoneTool` signals task completion - the task's `run()` returns
5. For async handlers, define `async def my_tool(self, msg)` - Langroid handles it
