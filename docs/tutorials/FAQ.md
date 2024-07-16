# Frequently Asked Questions

### How langroid handles long chat histories

You may encounter an error like this:

```
Error: Tried to shorten prompt history but ... longer than context length
```

This might happen when your chat history bumps against various limits.
Here is how Langroid handles long chat histories. Ultimately the LLM API is invoked with two key inputs:
the message history $h$, and the desired output length $n$ (defaults to the `max_output_tokens` in the 
`ChatAgentConfig`). These inputs are determined as follows (see the `ChatAgent._prep_llm_messages` method):

- let $H$ be the current message history, and $M$ be the value of `ChatAgentConfig.max_output_tokens`, and $C$ be 
  the context-length of the LLM.
- If $\text{tokens}(H) + M \leq C$, then langroid uses $h = H$ and $n = M$, since there is enough room to fit both the 
  actual chat history as well as the desired max output length.
- If $\text{tokens}(H) + M > C$, this means the context length is too small to accommodate the message history $H$ 
  and 
  the desired output length $M$. Then langroid tries to use a _shortened_ output length $n' = C - \text{tokens}(H)$, 
  i.e. the output is effectively _truncated_ to fit within the context length. 
    - If $n'$ is at least equal to `min_output_tokens` $m$ (default 10), langroid proceeds with $h = H$ and $n=n'$.
    - otherwise, this means that the message history $H$ is so long that the remaining space in the LLM's 
      context-length $C$ is unacceptably small (i.e. smaller than the minimum output length $m$). In this case,
      Langroid tries to shorten the message history by dropping early messages, and updating the message history $h$ as 
      long as $C - \text{tokens}(h) <  m$, until there are no more messages to drop (it will not drop the system 
      message or the last message, which is a user message), and throws the error mentioned above. 

If you are getting this error, you will want to check whether:

- you have set the `chat_context_length` too small, if you are setting it manually
- you have set the `max_output_tokens` too large
- you have set the `min_output_tokens` too large

If these look fine, then the next thing to look at is whether you are accumulating too much context into the agent 
history, for example retrieved passages (which can be very long) in a RAG scenario. One common case is when a query 
$Q$ is being answered using RAG, the retrieved passages $P$ are added to $Q$ to create a (potentially very long) prompt 
like 
> based on the passages P, answer query Q

Once the LLM returns an answer (if appropropriate for your context), you should avoid retaining the passages $P$ in the 
agent history, i.e. the last user message should be simply $Q$, rather than the prompt above. This functionality is exactly what you get when you 
use `ChatAgent._llm_response_temp_context`, which is used by default in the `DocChatAgent`. 

Another way to keep chat history tokens from growing too much is to use the `llm_response_forget` method, which 
erases both the query and response, if that makes sense in your scenario.
