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

When imported history includes OpenAI tool calls, the agent rebuilds its call
lookup and pending-call list. A call remains pending only when the history does
not contain a matching `tool` result. This allows an interrupted tool workflow
to resume without treating completed calls as pending.

Import validates the complete snapshot before replacing the current history.
Malformed JSON, unsupported versions, invalid messages, or invalid Base64 data
raise `ValueError` and leave the existing conversation unchanged.

Snapshots contain conversation text and attachment contents. Store them with
the same access controls used for other user data. The previous
`examples/basic/chat-persist.py` pickle file is intentionally not loaded or
migrated automatically because unpickling untrusted data can execute code.
