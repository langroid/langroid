# `OpenAIAssistant` removed

**`OpenAIAssistant` and `OpenAIAssistantConfig` were removed in Langroid 0.68.0.**

## What happened

OpenAI [sunset the Assistants API beta on 2026-08-26][sunset]. Every endpoint under
`/v1/assistants` (and the associated threads, runs, and messages endpoints) now
returns HTTP 404.

Langroid's `OpenAIAssistant` was a thin wrapper over exactly those endpoints, so
after the sunset it could not function under any configuration. This was not
fixable by retrying, switching models, or upgrading the `openai` package, and
there was no migration window worth preserving — the class was inert. It was
therefore removed outright rather than deprecated.

Removed in 0.68.0:

- `langroid/agent/openai_assistant.py` (`OpenAIAssistant`, `OpenAIAssistantConfig`,
  `AssistantTool`, `ToolType`, and related helpers)
- `tests/main/test_openai_assistant.py`, `tests/main/test_openai_assistant_async.py`
- `examples/basic/oai-asst-chat.py`, `examples/basic/oai-code-chat.py`,
  `examples/docqa/oai-multi-extract.py`, `examples/docqa/oai-retrieval-2.py`,
  `examples/docqa/oai-retrieval-assistant.py`

The last release containing them is
[0.67.5](https://github.com/langroid/langroid/tree/0.67.5).

## What is *not* affected

**`OpenAIGPT` is unaffected**, and that is what nearly all Langroid code uses. It
talks to the Chat Completions API, which OpenAI continues to support. If your code
looks like this, nothing changes:

```python
import langroid.language_models as lm

llm_cfg = lm.OpenAIGPTConfig(chat_model="gpt-4.1")
```

Also unaffected: `ChatAgent`, `Task`, `DocChatAgent`, all tool/function-calling
machinery, and every non-OpenAI provider.

## Migrating

Only code that explicitly imported the assistant module needs changing:

```python
# Before (no longer works -- ImportError as of 0.68.0)
from langroid.agent.openai_assistant import OpenAIAssistant, OpenAIAssistantConfig

agent = OpenAIAssistant(
    OpenAIAssistantConfig(
        name="MyAssistant",
        system_message="You are a helpful assistant.",
    )
)
```

```python
# After
import langroid as lr
import langroid.language_models as lm

agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="MyAssistant",
        system_message="You are a helpful assistant.",
        llm=lm.OpenAIGPTConfig(chat_model="gpt-4.1"),
    )
)
```

`ChatAgent` is a drop-in replacement for the conversational parts. The
Assistants-API-specific features map onto Langroid equivalents as follows.

| Assistants API feature | Langroid replacement |
|---|---|
| Server-side conversation state (threads) | `ChatAgent.message_history`, plus [chat-history snapshots](chat-history-snapshots.md) if you need to save and restore |
| `file_search` / retrieval | [`DocChatAgent`](../notes/overview.md) over any Langroid vector store |
| `code_interpreter` | A `ToolMessage` handler, or an MCP tool (see the Pyodide executor in `examples/mcp/`) |
| Assistant-side function calling | Langroid [tools / function-calling](tool-message-handler.md), portable across providers |
| Persistent assistant IDs | Not needed — agent config is defined in your own code |

One behavioral difference worth noting: the Assistants API kept conversation state
on OpenAI's servers, whereas `ChatAgent` keeps it in `message_history` in your
process. If you relied on reconnecting to a thread across runs, persist the history
yourself — see [Chat History Snapshots](chat-history-snapshots.md).

## References

- [OpenAI: Assistants API beta deprecation (Aug 26, 2026 sunset)][sunset]
- [OpenAI: migration guide to the Responses API](https://developers.openai.com/api/docs/assistants/migration)
- [OpenAI: deprecations](https://platform.openai.com/docs/deprecations)

[sunset]: https://community.openai.com/t/assistants-api-beta-deprecation-august-26-2026-sunset/1354666
