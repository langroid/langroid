# Batch Processing

Langroid can run many tasks or agent methods concurrently over a list of items.
The helpers in `langroid/agent/batch.py` are `run_batch_tasks`,
`run_batch_task_gen`, `run_batch_agent_method`, `llm_response_batch`,
`agent_response_batch`, and `run_batch_function`.

## Basic usage

Given a configured agent, create a task and map each item to its input and each
task result to the value needed by the caller:

```python
from langroid.agent.batch import run_batch_tasks
from langroid.agent.task import Task

task = Task(agent, interactive=False)
items = ["first question", "second question"]

answers = run_batch_tasks(
    task,
    items,
    input_map=lambda item: item,
    output_map=lambda result: None if result is None else result.content,
    sequential=False,
)
```

The returned list has the same length and ordering as `items`.

## Stopping at the first valid result

Set `stop_on_first_result=True` for a search-style batch that should return as
soon as any task produces a valid result. A result is valid when it is non-`None`
*after* `output_map` runs. The usual idiom is therefore to map unusable output
to `None`; the batch will keep waiting for another task.

Results retain their original input positions. Entries are `None` for tasks
that were cancelled or whose output mapped to `None`. If every task finishes
with a `None`-mapped result, the function returns an all-`None` list after all
tasks have completed.

For helpers that accept `handle_exceptions`, its value also affects stopping:

- `ExceptionHandling.RETURN_NONE` converts an exception to `None`, so the batch
  keeps waiting.
- `ExceptionHandling.RETURN_EXCEPTION` returns the exception object. Because
  that object is non-`None`, it counts as a stopping result and occupies the
  failed task's original position.
- `ExceptionHandling.RAISE`, the default, propagates the exception.

For example, suppose the first search finishes quickly without a usable result,
the second finds an answer later, and the third is still running:

```python
items = ["fast miss", "slower match", "long-running search"]

results = run_batch_tasks(
    search_task,
    items,
    input_map=lambda query: query,
    output_map=lambda result: (
        result.content if result is not None and is_usable(result) else None
    ),
    stop_on_first_result=True,
    sequential=False,
)

# The fast miss does not stop the batch. The long search is cancelled.
assert results == [None, "the valid answer", None]
```

### Behavior change

In earlier versions, a fast result that mapped to `None` could cancel the
remaining tasks and return an all-`None` list. This was fixed in
[GitHub issue #1068](https://github.com/langroid/langroid/issues/1068). Users do
not need to make any API changes.

## Batching

Set `batch_size` to process fixed-size batches in input order. With
`stop_on_first_result=True`, once a valid result is found, later batches are
skipped and their positions are filled with `None`.

Set `sequential=True` to process items one at a time when normal batch
processing is used. The `stop_on_first_result` path schedules the tasks in the
current batch concurrently so that it can return whichever valid result
finishes first.

See `tests/main/test_batch.py` for executable examples and edge-case coverage.
