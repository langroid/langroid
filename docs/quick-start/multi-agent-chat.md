# Two-Agent Chat

!!! tip "Script in `langroid-examples`"
        A full working example for the material in this section is
        in the `multi-agent-chat.py` script in the `langroid-examples` repo:
        [`examples/quick-start/mult-agent-chat.py`](https://github.com/langroid/langroid-examples/tree/main/examples/quick-start/multi-agent-chat.py).

## Why multiple agents?

In non-trivial LLM applications, we will want to decompose the overall task into 
multiple sub-tasks, each requiring different capabilities. We could have an agent


In any non-trivial application, we will want to set up multiple `ChatAgent`
instances, each instructed differently, and maintaining their own 
LLM conversation histories (and possibly equipped with different memories via vector-stores,
and different tools/plugins/function-calling capabilities).
We saw in the previous section that a `Task` orchestrates how a `ChatAgent`'s
responders update the current `pending_message`. 

_____