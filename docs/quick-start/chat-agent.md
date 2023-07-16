# A simple chat agent

!!! tip "Script in `langroid-examples`"
        A full working example for the material in this section is
        in the `chat-agent.py` script in the `langroid-examples` repo:
        [`examples/quick-start/chat-agent.py`](https://github.com/langroid/langroid-examples/tree/main/examples/quick-start/chat-agent.py).

## Agents 

A [`ChatAgent`](/reference/agent/chat_agent) is an abstraction that 
wraps a few components, including:

- an LLM (`ChatAgent.llm`), possibly equipped with tools/function-calling. 
  The `ChatAgent` class maintains LLM conversation history.
- optionally a vector-database (`ChatAgent.vecdb`)

## Agents as message transformers
In Langroid, a core function of `ChatAgents` is _message transformation_.
There are three special message transformation methods, which we call **responders**.
Each of these takes a message and returns a message. 
More specifically, their function signature is (simplified somewhat):
```py
str | ChatDocument -> ChatDocument
```
where `ChatDocument` is a class that wraps a message content (text) and its metadata.
There are three responder methods in `ChatAgent`, one corresponding to each 
[responding entity](/reference/mytypes) (LLM, USER, or AGENT):

- `llm_response`: returns the LLM response to the input message.
  (The input message is added to the LLM history, and so is the subsequent response.)
- `agent_response`: a method that can be used to implement a custom agent response. 
   Typically, an `agent_response` is used to handle messages containing a 
   "tool" or "function-calling" (more on this later). Another use of `agent_response` 
   is _message validation_.
- `user_response`: return input from the user. Useful to allow a human user to 
   intervene or quit.

Creating an agent is easy. First define a `ChatAgentConfig` object, and then
instantiate a `ChatAgent` object with that config:
```py
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
config = ChatAgentConfig(
    llm = OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT4) #(1)!
)
agent = ChatAgent(config)
```

1. This agent only has an LLM, and no vector-store. Examples of agents with
   vector-stores will be shown later.

We can now use the agent's responder methods:
```py
response = agent.llm_response("What is 2 + 4?").content
```

## Task: orchestrator for agents
In order to do anything useful with a `ChatAgent`, we need to have a way to 
sequentially invoke its responder methods, in some principled way.
For example in the simple chat loop we saw in the 
[previous section](llm-interaction.md), in the 
[`try-llm.py`](https://github.com/langroid/langroid-examples/blob/main/examples/quick-start/try-llm.py)
script, we had a loop that alternated between getting a human input and an LLM response.
This is one of the simplest possible loops, but in more complex applications, 
we need a general way to orchestrate the agent's responder methods.

The [`Task`](/reference/agent/task) class is an abstraction around a 
`ChatAgent`, responsible for orchestrating the agent's responder methods.
A `Task` is initialized with a specific `ChatAgent` instance, and some 
optional arguments, including an initial message to "kick-off" the agent.
The `Task.run()` method is the main entry point for `Task` objects, and works 
as follows:

- it first calls the `Task.init()` method to initialize the `pending_message`, 
  which represents the latest message that needs a response.
- it repeatedly calls `Task.step()` until `Task.done()` is True, and returns
  `Task.result()` as the final result of the task.

`Task.step()` is where all the action happens. It represents a "turn" in the 
"conversation": in the case of a single `ChatAgent`, the conversation involves 
only the three responders mentioned above, but when a `Task` has sub-tasks, 
it can involve other tasks well (ignore this for now). `Task.step()` loops over 
the `ChatAgent`'s responders (plus sub-tasks if any) until it finds a _valid_ 
response to the current `pending_message`, i.e. a "meaningful" response, 
something other than None for example.
Once `Task.step()` finds a valid response, it updates the `pending_message` and 
the next invocation of `Task.step()` will search for a valid response to this 
updated message, and so on.
`Task.step()` incorporates mechanisms to ensure proper handling of messages,
e.g. the USER a chance to respond after each non-USER response
(to avoid infinite runs without human intervention),
and preventing an entity from responding if it has just responded, etc.


The above details were only provided to give you a glimpse into how Agents and 
Tasks work. Unless you are creating a custom orchestration mechanism, you do not
need to be aware of these details. In fact our basic human + LLM chat loop can be trivially 
implemented with a `Task`:
```py
from langroid.agent.task import Task
task = Task(agent, system_message="You are a helpful assistant")
```
We can then run the task:
```py
task.run()
```
Note that the agent's `agent_response()` always returns None (since the default 
implementation of this method looks for a tool/function-call, and these never occur
in this task). So the calls to `task.step()` alternates between 
responses from the LLM and the user.

See the [`chat-agent.py](https://github.com/langroid/langroid-examples/blob/main/examples/quick-start/chat-agent.py)
for a working example that you can run with
```sh
python3 examples/quick-start/chat-agent.py
```

In the [next section](multi-agent-chat.md) we will see how to have two agents collaborate on a task.



