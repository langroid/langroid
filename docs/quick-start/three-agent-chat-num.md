# Three-Agent Collaboration

!!! tip "Script in `langroid-examples`"
    A full working example for the material in this section is
    in the `three-agent-chat-num.py` script in the `langroid-examples` repo:
    [`examples/quick-start/three-agent-chat-num.py`](https://github.com/langroid/langroid-examples/tree/main/examples/quick-start/three-agent-chat-num.py).


Let us set up a simple numbers exercise between 3 agents.
The `Processor` agent receives a list of numbers, and its goal is to 
apply a transformation to each number $n$. However it does not know how to apply these
transformations, and takes the help of two other agents to do so.
Given a number $n$,

- The `EvenHandler` returns $n/2$ if n is even, otherwise says `DO-NOT-KNOW`.
- The `OddHandler` returns $3n+1$ if n is odd, otherwise says `DO-NOT-KNOW`.

As before we first create a common `ChatAgentConfig` to use for all agents:

```py
config = lr.ChatAgentConfig(
    llm = lr.language_models.OpenAIGPTConfig(
        chat_model=lr.language_models.OpenAIChatModel.GPT4,
    ),
    vecdb=None,
)
```

Next, set up the `processor_agent`, along with instructions for the task:
```py
processor_agent = lr.ChatAgent(config)
processor_task = lr.Task(
    processor_agent,
    name = "Processor",
    system_message="""
        You will receive a list of numbers from the user.
        Your goal is to apply a transformation to each number.
        However you do not know how to do this transformation,
        so the user will help you. 
        You can simply send the user each number FROM THE GIVEN LIST
        and the user will return the result 
        with the appropriate transformation applied.
        IMPORTANT: only send one number at a time, concisely, say nothing else.
        Once you have accomplished your goal, say DONE and show the result.
        Start by asking the user for the list of numbers.
        """,
    llm_delegate=True, #(1)!
    single_round=False, #(2)!
)
```

1. Setting the `llm_delegate` option to `True` means that the `processor_task` is
    delegated to the LLM (as opposed to the User), 
    in the sense that the LLM is the one "seeking" a response to the latest 
    number. Specifically, this means that in the `processor_task.step()` 
    when a sub-task returns `DO-NOT-KNOW`,
    it is _not_ considered a valid response, and the search for a valid response 
    continues to the next sub-task if any.
2. `single_round=False` means that the `processor_task` should _not_ terminate after 
    a valid response from a responder.

Set up the other two agents and tasks:

```py
NO_ANSWER = lr.utils.constants.NO_ANSWER

even_agent = lr.ChatAgent(config)
even_task = lr.Task(
    even_agent,
    name = "EvenHandler",
    system_message=f"""
    You will be given a number. 
    If it is even, divide by 2 and say the result, nothing else.
    If it is odd, say {NO_ANSWER}
    """,
    single_round=True,  # task done after 1 step() with valid response
)

odd_agent = lr.ChatAgent(config)
odd_task = lr.Task(
    odd_agent,
    name = "OddHandler",
    system_message=f"""
    You will be given a number n. 
    If it is odd, return (n*3+1), say nothing else. 
    If it is even, say {NO_ANSWER}
    """,
    single_round=True,  # task done after 1 step() with valid response
)
```

Now add the `even_task` and `odd_task` as subtasks of the `processor_task`, 
and then run it as before:

```python
processor_task.add_sub_task([even_task, odd_task])
processor_task.run()
```


Feel free to try the working example script
[`three-agent-chat-num.py`]()
`langroid-examples` repo:
[`examples/quick-start/three-agent-chat-num.py`](https://github.com/langroid/langroid-examples/tree/main/examples/quick-start/three-agent-chat-num.py):

```bash
python3 examples/quick-start/three-agent-chat-num.py
```

Here's a screenshot of what it looks like:
![three-agent-num.png](three-agent-num.png)

## Next steps


In the [next section](chat-agent-tool.md) you will learn how to use Langroid
to equip a `ChatAgent` with tools or function-calling.

