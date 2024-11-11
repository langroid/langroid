# Handling large tool results

Available since Langroid v0.22.0.

In some cases, the result of handling a `ToolMessage` could be very large,
e.g. when the Tool is a database query that returns a large number of rows,
or a large schema. When used in a task loop, this large result may then be
sent to the LLM to generate a response, which in some scenarios may not
be desirable, as it increases latency, token-cost and distractions. 
Langroid allows you to set two optional parameters in a `ToolMessage` to
handle this situation:

- `_max_result_tokens`: *immediately* truncate the result to this number of tokens.
- `_max_retained_tokens`: *after* a responder (typically the LLM) responds to this 
   tool result (which optionally may already have been 
   truncated via `_max_result_tokens`),
   edit the message history to truncate the result to this number of tokens.

You can set one, both or none of these parameters. If you set both, you would 
want to set `_max_retained_tokens` to a smaller number than `_max_result_tokens`.

See the test `test_reduce_raw_tool_result` in `test_tool_messages.py` for an example.

Here is a conceptual example. Suppose there is a Tool called `MyTool`,
with parameters `_max_result_tokens=20` and `_max_retained_tokens=10`.
Imagine a task loop where the user says "hello", 
and then LLM generates a call to `MyTool`, 
and the tool handler (i.e. `agent_response`) generates a result of 100 tokens.
This result is immediately truncated to 20 tokens, and then the LLM responds to it
with a message `response`.


The agent's message history looks like this:

```
1. System msg.
2. user: hello
3. LLM: MyTool
4. Agent (Tool handler): 100-token result => reduced to 20 tokens
5. LLM: response
```

Immediately after the LLM's response at step 5, the message history is edited
so that the message contents at position 4 are truncated to 10 tokens,
as specified by `_max_retained_tokens`.

