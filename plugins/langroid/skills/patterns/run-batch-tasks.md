# Batch Processing with run_batch_tasks()

## The Pattern

Use `run_batch_tasks()` to process multiple inputs through the same task/agent
logic concurrently. Each input gets a **cloned** task+agent with isolated state.

## When to Use

- Process many items (prompts, questions, documents) with the same agent logic
- Need parallelism without manual asyncio/threading complexity
- Need state isolation between items (no message history leakage)
- Want to avoid connection exhaustion from creating agents manually
- Need ordered results matching input order

## Key Functions

### `run_batch_tasks()` - Simple Case

```python
from langroid.agent.batch import run_batch_tasks

results = run_batch_tasks(
    task,                    # Base task to clone
    items,                   # List of items to process
    input_map=lambda x: x,   # Convert item -> prompt string
    output_map=lambda x: x,  # Convert result -> desired output
    sequential=False,        # False = parallel, True = sequential
    batch_size=10,           # Max concurrent tasks (None = unlimited)
    turns=-1,                # Max turns per task (-1 = unlimited)
)
```

### `run_batch_task_gen()` - Custom Task Generation

```python
from langroid.agent.batch import run_batch_task_gen

def task_gen(i: int) -> Task:
    """Generate a custom task for item at index i."""
    return base_task.clone(i)  # or create entirely new task

results = run_batch_task_gen(
    gen_task=task_gen,       # Function that creates task for each index
    items=items,
    input_map=lambda x: x,
    output_map=lambda x: x,
    sequential=False,
)
```

## How Cloning Works

When `run_batch_tasks()` processes each item, it calls `task.clone(i)`:

1. **Task cloning** (`Task.clone()`):
   - Creates new Task with name `{original}-{i}`
   - Calls `agent.clone(i)` for the agent

2. **Agent cloning** (`ChatAgent.clone()`):
   - Deep copies the config
   - Creates fresh agent with new message history
   - Copies tool definitions (shared, not duplicated)
   - Clones vector store client if present
   - Assigns unique agent ID

**Result**: Each item is processed by an isolated agent with no state leakage.

## Example: Analyze Multiple Code Files

```python
import langroid as lr
from langroid.agent.batch import run_batch_tasks

# Create base agent and task
agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="CodeAnalyzer",
        llm=lr.language_models.OpenAIGPTConfig(chat_model="gpt-4"),
        system_message="Analyze code for security vulnerabilities.",
    )
)
agent.enable_message([VulnerabilityTool])
base_task = lr.Task(agent, interactive=False)

# Process multiple code files
code_files = [
    {"id": "file1", "code": "void foo() { strcpy(buf, input); }"},
    {"id": "file2", "code": "void bar() { strncpy(buf, input, sizeof(buf)); }"},
    {"id": "file3", "code": "void baz() { gets(buffer); }"},
]

results = run_batch_tasks(
    base_task,
    items=code_files,
    input_map=lambda f: f"Analyze this code:\n{f['code']}",
    output_map=lambda r: r.content if r else "ANALYSIS_FAILED",
    sequential=False,
    batch_size=5,  # Max 5 concurrent analyses
)

for file, result in zip(code_files, results):
    print(f"{file['id']}: {result}")
```

## Example: Q&A with Structured Output

```python
from langroid.agent.batch import run_batch_tasks

class AnswerTool(lr.ToolMessage):
    request: str = "answer"
    purpose: str = "Provide an answer"
    answer: str
    confidence: float

agent = lr.ChatAgent(config)
agent.enable_message([AnswerTool])

# Configure task to return tool directly
task = lr.Task(
    agent,
    interactive=False,
    config=lr.TaskConfig(done_if_tool=True)
)[AnswerTool]  # Bracket notation: task returns AnswerTool | None

questions = ["What is 2+2?", "Capital of France?", "Largest planet?"]

answers = run_batch_tasks(
    task,
    items=questions,
    input_map=lambda q: q,
    output_map=lambda tool: tool.answer if tool else "NO_ANSWER",
    sequential=False,
    batch_size=3,
)
# answers = ["4", "Paris", "Jupiter"]
```

## Example: With Stateful Agent Handler

Combining batch processing with stateful handlers (see pattern #2):

```python
class QueryAgent(lr.ChatAgent):
    def __init__(self, config, db_connection, max_retries=3):
        super().__init__(config)
        self.db = db_connection
        self.max_retries = max_retries
        self.failures = 0

    def init_state(self):
        super().init_state()
        self.failures = 0  # Reset per clone

    def execute_query(self, msg: QueryTool) -> str | DoneTool:
        try:
            result = self.db.execute(msg.query)
            return DoneTool(content=str(result))
        except Exception as e:
            self.failures += 1
            if self.failures >= self.max_retries:
                return DoneTool(content="")
            return f"Error: {e}. Fix and retry."

agent = QueryAgent(config, db_connection=my_db)
agent.enable_message([QueryTool])
base_task = lr.Task(agent, interactive=False)

# Each query gets a cloned agent with fresh failure counter
queries = ["SELECT * FROM users", "SELECT * FROM orders", ...]
results = run_batch_tasks(base_task, queries, ...)
```

## Parameters Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `task` | Task | Base task to clone for each item |
| `items` | List[T] | Items to process |
| `input_map` | Callable[[T], str] | Convert item to prompt |
| `output_map` | Callable[[Result], U] | Convert result to output |
| `sequential` | bool | True=one at a time, False=parallel |
| `batch_size` | int\|None | Max concurrent tasks (None=all) |
| `turns` | int | Max turns per task (-1=unlimited) |
| `handle_exceptions` | bool\|ExceptionHandling | How to handle errors |
| `max_cost` | float | Stop if cumulative cost exceeds |
| `max_tokens` | int | Stop if cumulative tokens exceed |

## Important Notes

1. **Order preserved**: Results list matches input items order
2. **Exceptions**: By default raised; use `handle_exceptions=RETURN_NONE` to continue
3. **Memory**: Each clone has separate message history - no accumulation
4. **Connections**: Cloned agents share underlying LLM client but have separate state
5. **Vector stores**: Each clone gets its own vector store client (same data, isolated state)
