# Missing and empty LLM message content

`LLMResponse.message` and `LLMMessage.content` are `Optional[str]`. Their values
have distinct meanings:

- `None` means the provider returned no text content.
- `""` means the provider returned text content that was present but empty.
- Non-empty strings are ordinary message text.

This distinction matters for tool-call-only assistant turns. Gemini 3.x rejects
assistant messages that combine a tool or function call with padded text. Langroid
therefore preserves `None` through `ChatDocument.content_is_none` and omits the
`content` field when serializing such a turn with `LLMMessage.api_dict()`.

`ChatDocument.content` remains a string for compatibility. Check
`content_is_none` when reconstructing an `LLMMessage` or otherwise preserving the
provider's original message shape.

An entirely empty message with no tool or function call is different: Langroid
serializes it with a single-space `content` value because some provider APIs reject
messages with no usable fields.
