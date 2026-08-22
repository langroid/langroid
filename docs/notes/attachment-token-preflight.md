# Attachment Token Preflight

Before each LLM call, Langroid runs a local context-length preflight: it counts
the tokens in the pending message history (`ChatAgent.chat_num_tokens()`) and,
if needed, reduces the requested output length, truncates old messages, or
drops old turns so the request fits the model's context window.

Historically this preflight counted only each message's text `content`, ignoring
`LLMMessage.files` (`FileAttachment` objects). But attachments ARE serialized
into the actual API request — for Gemini and other OpenAI-compatible paths,
`FileAttachment.to_dict()` turns an attached PDF or image into a `data:` URI
containing the full base64 payload. A ~1.9 MB PDF becomes roughly 2.6 million
base64 characters, so the local check could pass while the real payload was far
larger (issue #996).

## Behavior

The preflight now includes an estimate for each attachment on `USER` messages:

- Each attachment is serialized exactly as it would be for the API request
  (`FileAttachment.to_dict(model)`), and the resulting JSON is tokenized with
  the agent's parser, the same way message text is counted.
- Attachments that are NOT inlined — e.g. a `FileAttachment` carrying an
  `http(s)` URL, which is sent as the URL itself rather than base64 bytes —
  contribute only the tokens of their small serialized form, not the size of
  the remote file.
- Messages with no attachments are counted exactly as before; token accounting
  is unchanged for attachment-free histories.

When attachments contribute to the count for an agent, a warning is logged once
(per agent, not per message) noting that the count is an estimate.

## How conservative is the estimate?

The estimate measures the serialized request payload (base64 data URI plus the
surrounding JSON), tokenized like ordinary text. Provider-side accounting may
differ in either direction:

- Providers that bill inlined files as text-like payload will be close to this
  estimate.
- Providers that convert files to their own internal representation (e.g.
  per-page or per-image token charges) may count fewer or more tokens.

The goal is to keep the local preflight from drastically *under*-estimating the
request size when large files are attached, so context-window overflows are
caught locally instead of surfacing as provider-side errors.
