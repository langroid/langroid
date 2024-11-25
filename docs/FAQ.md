# Frequently Asked Questions

## Does Langroid work with non-OpenAI LLMs?

Yes! Langroid works with practically any LLM, local or remote, closed or open.

See these two guides:

- [Using Langroid with local/open LLMs](https://langroid.github.io/langroid/tutorials/local-llm-setup/)
- [Using Langroid with non-OpenAI proprietary LLMs](https://langroid.github.io/langroid/tutorials/non-openai-llms/)

## Where can I find out about Langroid's architecture?

There are a few documents that can help:

- A work-in-progress [architecture description](https://langroid.github.io/langroid/blog/2024/08/15/overview-of-langroids-multi-agent-architecture-prelim/)
  on the Langroid blog.
- The Langroid [Getting Started](https://langroid.github.io/langroid/quick-start/) guide walks you 
  step-by-step through Langroid's features and architecture.
- An article by LanceDB on [Multi-Agent Programming with Langroid](https://lancedb.substack.com/p/langoid-multi-agent-programming-framework)

## How langroid handles long chat histories

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

## How can I handle large results from Tools?

As of version 0.22.0, Langroid allows you to control the size of tool results
by setting [optional parameters](https://langroid.github.io/langroid/notes/large-tool-results/) 
in a `ToolMessage` definition.

## Can I handle a tool without running a task?

Yes, if you've enabled an agent to both _use_ (i.e. generate) and _handle_ a tool. 
See the `test_tool_no_task` for an example of this. The `NabroskiTool` is enabled
for the agent, and to get the agent's LLM to generate the tool, you first do 
something like:
```python
response = agent.llm_response("What is Nabroski of 1 and 2?")
```
Now the `response` is a `ChatDocument` that will contain the JSON for the `NabroskiTool`.
To _handle_ the tool, you will need to call the agent's `agent_response` method:

```python
result = agent.agent_response(response)
```

When you wrap the agent in a task object, and do `task.run()` the above two steps are done for you,
since Langroid operates via a loop mechanism, see docs 
[here](https://langroid.github.io/langroid/quick-start/multi-agent-task-delegation/#task-collaboration-via-sub-tasks).
The *advantage* of using `task.run()` instead of doing this yourself, is that this method
ensures that tool generation errors are sent back to the LLM so it retries the generation.

## OpenAI Tools and Function-calling support

Langroid supports OpenAI tool-calls API as well as OpenAI function-calls API.
Read more [here](https://github.com/langroid/langroid/releases/tag/0.7.0).

Langroid has always had its own native tool-calling support as well, 
which works with **any** LLM -- you can define a subclass of `ToolMessage` (pydantic based) 
and it is transpiled into system prompt instructions for the tool. 
In practice, we don't see much difference between using this vs OpenAI fn-calling. 
Example [here](https://github.com/langroid/langroid/blob/main/examples/basic/fn-call-local-simple.py).
Or search for `ToolMessage` in any of the `tests/` or `examples/` folders.

## Some example scripts appear to return to user input immediately without handling a tool.

This is because the `task` has been set up with `interactive=True` 
(which is the default). With this setting, the task loop waits for user input after
either the `llm_response` or `agent_response` (typically a tool-handling response) 
returns a valid response. If you want to progress through the task, you can simply 
hit return, unless the prompt indicates that the user needs to enter a response.

Alternatively, the `task` can be set up with `interactive=False` -- with this setting,
the task loop will _only_ wait for user input when an entity response (`llm_response` 
or `agent_response`) _explicitly_ addresses the user. Explicit user addressing can
be done using either:

- an orchestration tool, e.g. `SendTool` (see details in
the release notes for [0.9.0](https://github.com/langroid/langroid/releases/tag/0.9.0)), an example script is the [multi-agent-triage.py](https://github.com/langroid/langroid/blob/main/examples/basic/multi-agent-triage.py), or 
- a special addressing prefix, see the example script [1-agent-3-tools-address-user.py](https://github.com/langroid/langroid/blob/main/examples/basic/1-agent-3-tools-address-user.py)


## Can I specify top_k in OpenAIGPTConfig (for LLM API calls)?

No; Langroid currently only supports parameters accepted by OpenAI's API, and `top_k` is _not_ one of them. See:

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat/create)
- [Discussion on top_k, top_p, temperature](https://community.openai.com/t/temperature-top-p-and-top-k-for-chatbot-responses/295542/5)
- [Langroid example](https://github.com/langroid/langroid/blob/main/examples/basic/fn-call-local-numerical.py) showing how you can set other OpenAI API parameters, using the `OpenAICallParams` object.


## Can I persist agent state across multiple runs?

For example, you may want to stop the current python script, and 
run it again later, resuming your previous conversation.
Currently there is no built-in Langroid mechanism for this, but you can 
achieve a basic type of persistence by saving the agent's `message_history`:

-  if you used `Task.run()` in your script, make sure the task is 
set up with `restart=False` -- this prevents the agent state from being reset when 
the task is run again.
- using python's pickle module, you can save the `agent.message_history` to a file,
and load it (if it exists) at the start of your script.

See the example script [`chat-persist.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-persist.py)

For more complex persistence, you can take advantage of the `GlobalState`,
where you can store message histories of multiple agents indexed by their name.
Simple examples of `GlobalState` are in the [`chat-tree.py`](https://github.com/langroid/langroid/blob/main/examples/basic/chat-tree.py) example, 
and the [`test_global_state.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_global_state.py) test.

## How can I suppress LLM output?

You can use the `quiet_mode` context manager for this, see 
[here](https://langroid.github.io/langroid/notes/quiet-mode/)

## How can I deal with LLMs (especially weak ones) generating bad JSON in tools?

Langroid already attempts to repair bad JSON (e.g. unescaped newlines, missing quotes, etc)  
using the [json-repair](https://github.com/mangiucugna/json_repair) library and other
custom methods, before attempting to parse it into a `ToolMessage` object.
However this type of repair may not be able to handle all edge cases of bad JSON 
from weak LLMs. There are two existing ways to deal with this, and one coming soon:

- If you are defining your own `ToolMessage` subclass, considering deriving it instead
  from `XMLToolMessage` instead, see the [XML-based Tools](https://langroid.github.io/langroid/notes/xml-tools/)
- If you are using an existing Langroid `ToolMessage`, e.g. `SendTool`, you can 
  define your own subclass of `SendTool`, say `XMLSendTool`,
  inheriting from both `SendTool` and `XMLToolMessage`; see this 
  [example](https://github.com/langroid/langroid/blob/main/examples/basic/xml_tool.py)
- Coming soon: strict decoding to leverage the Structured JSON outputs supported by OpenAI
  and open LLM providers such as `llama.cpp` and `vllm`.

The first two methods instruct the LLM to generate XML instead of JSON,
and any field that is designated with a `verbatim=True` will be enclosed 
within an XML `CDATA` tag, which does *not* require any escaping, and can
be far more reliable for tool-use than JSON, especially with weak LLMs.

## How can I handle an LLM "forgetting" to generate a `ToolMessage`? 

Sometimes the LLM (especially a weak one) forgets to generate a 
[`ToolMessage`][langroid.agent.tool_message.ToolMessage]
(either via OpenAI's tools/functions API, or via Langroid's JSON/XML Tool mechanism),
despite being instructed to do so. There are a few remedies Langroid offers for this:

**Improve the instructions in the `ToolMessage` definition:**

- Improve instructions in the `purpose` field of the `ToolMessage`.
- Add an `instructions` class-method to the `ToolMessage`, as in the
  [`chat-search.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/chat-search.py) script:

```python
@classmethod
def instructions(cls) -> str:
    return """
        IMPORTANT: You must include an ACTUAL query in the `query` field,
        """
```
  These instructions are meant to be general instructions on how to use the tool
  (e.g. how to set the field values), not to specifically about the formatting.

- Add a `format_instructions` class-method, e.g. like the one in the 
  [`chat-multi-extract-3.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/chat-multi-extract-3.py) 
  example script.

```python
@classmethod
def format_instructions(cls, tool: bool = True) -> str:
    instr = super().format_instructions(tool)
    instr += """
    ------------------------------
    ASK ME QUESTIONS ONE BY ONE, to FILL IN THE FIELDS 
    of the `lease_info` function/tool.
    First ask me for the start date of the lease.
    DO NOT ASK ANYTHING ELSE UNTIL YOU RECEIVE MY ANSWER.
    """
    return instr
```

**Override the `handle_message_fallback` method in the agent:**

This method is called when the Agent's `agent_response` method receives a non-tool
message as input. The default behavior of this method is to return None, but it
is very useful to override the method to handle cases where the LLM has forgotten
to use a tool. You can define this method to return a "nudge" to the LLM
telling it that it forgot to do a tool-call, e.g. see how it's done in the 
example script [`chat-multi-extract-local.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/chat-multi-extract-local.py):

```python
class LeasePresenterAgent(ChatAgent):
    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        """Handle scenario where Agent failed to present the Lease JSON"""
        if isinstance(msg, ChatDocument) and msg.metadata.sender == Entity.LLM:
            return """
            You either forgot to present the information in the JSON format
            required in `lease_info` JSON specification,
            or you may have used the wrong name of the tool or fields.
            Try again.
            """
        return None
```

Note that despite doing all of these, the LLM may still fail to generate a `ToolMessage`.
In such cases, you may want to consider using a better LLM, or an up-coming Langroid
feature that leverages **strict decoding** abilities of specific LLM providers
(e.g. OpenAI, llama.cpp, vllm) that are able to use grammar-constrained decoding
to force the output to conform to the specified structure.

## Can I use Langroid to converse with a Knowledge Graph (KG)?

Yes, you can use Langroid to "chat with" either a Neo4j or ArangoDB KG, 
see docs [here](https://langroid.github.io/langroid/notes/knowledge-graphs/)

## How can I improve `DocChatAgent` (RAG) latency?

The behavior of `DocChatAgent` can be controlled by a number of settings in 
the `DocChatAgentConfig` class.
The top-level query-answering method in `DocChatAgent` is `llm_response`, which use the 
`answer_from_docs` method. At a high level, the response to an input message involves
the following steps:

- **Query to StandAlone:** LLM rephrases the query as a stand-alone query. 
   This can incur some latency. You can 
    turn it off by setting `assistant_mode=True` in the `DocChatAgentConfig`.
- **Retrieval:** The most relevant passages (chunks) are retrieved using a collection of semantic/lexical 
      similarity searches and ranking methods. There are various knobs in `DocChatAgentConfig` to control
      this retrieval.
- **Relevance Extraction:** LLM is used to retrieve verbatim relevant portions from
  the retrieved chunks. This is typically the biggest latency step. You can turn it off
  by setting the `relevance_extractor_config` to None in `DocChatAgentConfig`.
- **Answer Generation:** LLM generates answer based on retrieved passages.


See the [`doc-aware-chat.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/doc-aware-chat.py)
example script, which illustrates some of these settings.

In some scenarios you want to *only* use the **retrieval** step of a `DocChatAgent`.
For this you can use the [`RetrievalTool`][langroid.agent.tools.retrieval_tool.RetrievalTool].
See the `test_retrieval_tool` in 
[`test_doc_chat_agent.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_doc_chat_agent.py).
to learn how to use it. The above example script uses `RetrievalTool` as well.

## Is there support to run multiple tasks concurrently?

Yes, see the `run_batch_tasks` and related functions in 
[batch.py](https://github.com/langroid/langroid/blob/main/langroid/agent/batch.py).

See also:

- tests: [test_batch.py](https://github.com/langroid/langroid/blob/main/tests/main/test_batch.py),
   [test_relevance_extractor.py](https://github.com/langroid/langroid/blob/main/tests/main/test_relevance_extractor.py),
- example: [multi-agent-round-table.py](https://github.com/langroid/langroid/blob/main/examples/basic/multi-agent-round-table.py)

Another example is within 
[`DocChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/doc_chat_agent.py), 
which uses batch tasks for relevance extraction,
see the `get_verbatim_extracts` method -- when there are k relevant passages,
this runs k tasks concurrently, 
each of which uses an LLM-agent to extract relevant verbatim text from a passage.

## Can I use Langroid in a FastAPI server?

Yes, see the [langroid/fastapi-server](https://github.com/langroid/fastapi-server) repo. 