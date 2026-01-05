# Pattern: Make Task Return a Specific ToolMessage Directly

## Problem

When an agent emits a ToolMessage, you need to extract it from the task result. The naive approach is to search through `task.agent.message_history` to find the tool, but this is **error-prone** and **inefficient**.

## Solution

Use **TaskConfig with `done_if_tool=True`** combined with **bracket notation** to make the task:
1. Terminate as soon as a tool is emitted
2. Return the tool directly (typed as `ToolClass | None`)

## Code Pattern

### Wrong Approach (searching message_history)

```python
from langroid.agent.task import Task

task = Task(agent, interactive=False)
result = task.run(prompt, turns=5)

# BAD: Searching message_history
pruned_classes = None
for msg in task.agent.message_history:
    if isinstance(msg, EmitPrunedModelTool):
        pruned_classes = msg.classes
        break

if not pruned_classes:
    print("❌ Agent did not use the tool")
    return 1
```

**Problems**:
- Iterating through entire message history
- Error-prone type checking with `isinstance`
- Can miss the tool if not searching correctly
- Not type-safe

### Correct Approach (TaskConfig + bracket notation)

```python
import langroid as lr
from langroid.agent.task import Task

# 1. Create TaskConfig with done_if_tool=True
task_config = lr.TaskConfig(done_if_tool=True)

# 2. Use bracket notation to specify return type
task = Task(agent, interactive=False, config=task_config)[EmitPrunedModelTool]

# 3. Run task - returns EmitPrunedModelTool | None
result: EmitPrunedModelTool | None = task.run(prompt, turns=5)

# 4. Check if tool was emitted
if not result:
    print("❌ Agent did not use the tool")
    return 1

# 5. Access tool data directly
pruned_classes = result.classes  # Type-safe!
```

**Benefits**:
- Task terminates immediately when tool is emitted (efficient)
- Return type is explicit and type-safe
- No need to search message_history
- Clean, readable code

## Key Components

### 1. TaskConfig(done_if_tool=True)

```python
task_config = lr.TaskConfig(done_if_tool=True)
```

This tells the task to **stop as soon as any tool is emitted**, rather than continuing for `turns` iterations.

### 2. Bracket Notation: `Task(...)[ToolClass]`

```python
task = Task(agent, interactive=False, config=task_config)[EmitPrunedModelTool]
```

The bracket notation **specifies the expected return type**:
- If the agent emits `EmitPrunedModelTool`, task returns it
- If the agent doesn't emit the tool, task returns `None`
- Return type is `EmitPrunedModelTool | None`

### 3. Type-Safe Result Handling

```python
result: EmitPrunedModelTool | None = task.run(prompt, turns=5)

if not result:
    # Agent didn't emit the tool
    handle_failure()
else:
    # Tool was emitted, access fields directly
    data = result.classes  # Type-safe attribute access
```

## Real-World Example

From `tools/prune_xsdata_models.py`:

```python
import langroid as lr
from langroid.agent.task import Task
from interop.agents.model_pruning_agent import (
    EmitPrunedModelTool,
    create_model_pruning_agent,
)

# Create agent
agent = create_model_pruning_agent(
    raw_generated_code=raw_content,
    reference_code=reference_code,
    target_entity="Aircraft",
    model="gpt-4o",
)

# Configure task to return tool directly
task_config = lr.TaskConfig(done_if_tool=True)
task = Task(agent, interactive=False, config=task_config)[EmitPrunedModelTool]

# Build prompt
prompt = f"""
Here is the raw xsdata-generated code for Aircraft:

```python
{raw_content[:50000]}
```

Please analyze this code and emit pruned class definitions using the tool.
"""

# Run task - returns tool or None
result: EmitPrunedModelTool | None = task.run(prompt, turns=5)

if not result:
    print("❌ Agent did not use the EmitPrunedModelTool")
    return 1

# Extract data from tool
pruned_classes = result.classes
print(f"✅ Agent produced {len(pruned_classes)} pruned classes")

# Use the data
for cls_def in pruned_classes:
    print(f"   • {cls_def.class_name}: {len(cls_def.fields)} fields")
```

## When to Use This Pattern

Use this pattern when:
- ✅ You expect the agent to emit a **specific tool** as its final output
- ✅ You want **type-safe access** to the tool data
- ✅ You want the task to **terminate immediately** when the tool is emitted
- ✅ The tool emission is the **primary goal** of the task (not intermediate step)

Don't use this pattern when:
- ❌ The agent might emit multiple different tools during conversation
- ❌ You need the full conversation history
- ❌ Tool emission is an intermediate step in a longer workflow

## Related Patterns

- **handle_llm_no_tool**: Use this in `ChatAgentConfig` to catch cases where the LLM doesn't use the tool
- **ToolMessage validation**: Use Pydantic models to ensure tool output is well-formed
- **Multi-turn tasks**: Combine with `turns` parameter for agents that need multiple attempts

## Common Mistakes

### Mistake 1: Forgetting `done_if_tool=True`

```python
# WRONG: Task will run for all turns even after tool is emitted
task = Task(agent)[EmitPrunedModelTool]
result = task.run(prompt, turns=5)  # Wastes turns!
```

**Fix**: Always use `TaskConfig(done_if_tool=True)`

### Mistake 2: Not checking for None

```python
# WRONG: Will crash if agent doesn't emit tool
result = task.run(prompt, turns=5)
pruned_classes = result.classes  # AttributeError if result is None!
```

**Fix**: Always check `if not result:` before accessing fields

### Mistake 3: Searching message_history instead

```python
# WRONG: Negates the entire point of bracket notation
result = task.run(prompt, turns=5)
for msg in task.agent.message_history:
    if isinstance(msg, EmitPrunedModelTool):
        # Why did you use bracket notation then?
```

**Fix**: Trust the bracket notation - result IS the tool

## Summary

**Pattern**: `Task(agent, config=TaskConfig(done_if_tool=True))[ToolClass]`

**Returns**: `ToolClass | None`

**Benefits**:
- Efficient (terminates immediately)
- Type-safe (explicit return type)
- Clean (no message_history iteration)
- Robust (can't miss the tool)

**Use when**: Tool emission is the primary goal of the task
