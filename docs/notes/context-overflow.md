# Context-Overflow Handling

When a `ChatAgent`'s message history grows so large that
(history tokens + requested output tokens) exceeds the model's context length,
Langroid steps in before the LLM API call (in chat mode, i.e. the standard
`llm_response` path) and tries to make the request fit. This happens in stages:

1. **Shrink the requested output length.** The max output tokens for the request
   is reduced to `chat_context_length - history_tokens - CHAT_HISTORY_BUFFER`,
   where `CHAT_HISTORY_BUFFER` (300 tokens) is a safety margin that guards
   against token-count inaccuracies. If this reduced output length is still at
   least `llm.min_output_tokens` (default 64), nothing else is done.
2. **Shrink the message history.** If the output length would fall below
   `llm.min_output_tokens`, the history itself is compressed, using the
   strategy configured via `ChatAgentConfig.context_overflow_strategy`.
3. **Give up.** If even after compressing the history there is not enough room
   for `llm.min_output_tokens` of output, a `ValueError` is raised with hints
   (increase `chat_context_length`, decrease `max_output_tokens`, or shorten
   the system/user messages).

## The `context_overflow_strategy` config

`ChatAgentConfig.context_overflow_strategy` accepts two values:

- `"truncate"` (default): truncate the *content* of individual early messages,
  while preserving the message sequence. Keeping every message in place is
  important for LLM APIs that require strictly alternating USER/ASSISTANT
  messages. The system message (index 0) and the final user message are never
  touched; only the messages in between are eligible, oldest first. Each
  truncated message gets a `... [Contents truncated!]` marker appended.
- `"drop_turns"`: drop complete conversation turns, oldest first. A turn is a
  USER message together with all messages that follow it, up to (but not
  including) the next USER message. The system message and the last turn are
  preserved. This is more aggressive but cleaner, e.g. for voice agents with
  limited context.

Example:

```python
import langroid as lr
import langroid.language_models as lm

config = lr.ChatAgentConfig(
    llm=lm.OpenAIGPTConfig(
        chat_model="gpt-4.1-mini",
        max_output_tokens=1024,
        min_output_tokens=64,
    ),
    context_overflow_strategy="drop_turns",  # or "truncate" (the default)
)
agent = lr.ChatAgent(config)
```

## Even distribution of truncation

With the `"truncate"` strategy, the required token reduction is distributed
evenly across the eligible messages, so each message retains as much content
as possible. Walking forward from the oldest eligible message, at each step
the current excess

```text
budget = chat_context_length - min_output_tokens - CHAT_HISTORY_BUFFER
excess = history_tokens - budget
```

is divided by the number of remaining *compressible* messages — those whose
content can actually shrink, i.e. is above the 30-token floor plus the
truncation-marker overhead — and the current message is trimmed by (roughly)
that share of the excess, subject to:

- a floor of 30 tokens: no message's content is ever cut below 30 tokens, and
  messages already at/below the floor plus the truncation-marker overhead are
  left untouched and excluded from the even-share split (truncating them
  would save nothing — it could only *grow* them by the appended marker);
- a capacity check: each message absorbs at least the portion of the excess
  that the messages after it *cannot* absorb (they can each shed only their
  content above the floor), so a single forward pass suffices whenever
  fitting is possible at all.

The excess is recomputed after every message, so the process self-corrects as
it walks forward through the history.

For example, if the history is 400 tokens over budget and there are 4 eligible
messages of 300 tokens each, each one is trimmed by about 100 tokens, keeping
roughly 200 tokens apiece — instead of the previous behavior of collapsing
each message to a fixed 30 tokens (which destroyed far more context than
necessary; see issue
[#838](https://github.com/langroid/langroid/issues/838)).

If even cutting all eligible messages down to the 30-token floor cannot make
the history fit, a `ValueError` is raised as described above.
