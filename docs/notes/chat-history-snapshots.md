# Chat History Snapshots

`ChatAgent` can export and restore its message history as versioned JSON. This
is useful when a conversation must resume in a later process or move between
workers without serializing the entire agent.

```python
from pathlib import Path

from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig

history_path = Path(".cache/agent-state/history.json")
agent = ChatAgent(ChatAgentConfig())

if history_path.exists():
    agent.import_history(history_path.read_text())

# Run the agent or task, then persist its current history.
history_path.parent.mkdir(parents=True, exist_ok=True)
history_path.write_text(agent.export_history())
```

The snapshot contains a format version and the agent's `LLMMessage` records.
Binary file attachments are Base64-encoded so the result remains valid JSON.
Process-local `chat_document_id` values are cleared during export and import.

Snapshots do not capture the agent configuration or registered tools. Restore
them separately when constructing the new agent. On the next LLM call, the
configured system message (including current tool instructions) replaces the
restored system entry.

When imported history includes OpenAI tool calls, the agent rebuilds its call
lookup and pending-call metadata. A call remains pending only when the history
does not contain a matching `tool` result. This bookkeeping does not execute or
automatically resume an interrupted tool call.

Import validates the complete snapshot before replacing the current history.
Malformed JSON, unsupported versions, an empty message list, invalid message
structure, a non-system first message, or invalid Base64 data raise
`ValueError` and leave the existing conversation unchanged. Import does not
validate tool-call and result ordering or sequencing. Real exported histories
are valid by construction; malformed hand-edited sequencing instead surfaces
as a provider error on the next LLM call. Attachment decoding has a cumulative
100 MiB default limit. Set a different budget explicitly when needed:

```python
agent.import_history(snapshot, max_size_bytes=20 * 1024 * 1024)
```

Snapshots contain conversation text and attachment contents. Store them with
the same access controls used for other user data. The previous
`examples/basic/chat-persist.py` pickle file is intentionally not loaded or
migrated automatically because unpickling untrusted data can execute code.
