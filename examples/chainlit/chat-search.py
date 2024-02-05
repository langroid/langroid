"""
Basic single-agent chat example, using a web Search Tool, using ChainlitTaskCallbacks.

- User asks a question
- LLM either responds directly or generates a Metaphor web search Tool/function-call
    - if Tool used:
         - Agent handler recognizes this tool and returns search results
         - LLM sees search results and composes a response.
- user asks another question


After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-search.py
"""

import chainlit as cl
import langroid as lr
from langroid.agent.callbacks.chainlit import ChainlitTaskCallbacks
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool
from langroid.agent.callbacks.chainlit import add_instructions
from textwrap import dedent


@cl.on_chat_start
async def on_chat_start():
    await add_instructions(
        title="Agent with access to a web search Tool",
        content=dedent(
            """
        Agent uses the `MetaphorSearchTool` tool/fn-call to search the web using the 
        [Metaphor Search API](https://docs.exa.ai/reference/getting-started).
        - User asks question
        - Agent LLM uses `MetaphorSearchTool` to generate search results
        - Agent handler recognizes this tool and returns search results
        - User hits `c` to continue
        - Agent LLM composes answer
        """
        ),
    )

    tool_name = MetaphorSearchTool.default_value("request")
    sys_msg = f"""
        You are an astute, self-aware AI assistant, and you are adept at 
        responding to a user's question in one of two ways:
        - If you KNOW the answer from your own knowledge, respond directly.
        - OTHERWISE, request up to 5 results from a web search using 
          the `{tool_name}` tool/function-call.
          In this case you will receive the web search results, and you can 
          then compose a response to the user's question. 
    """
    config = lr.ChatAgentConfig(
        name="Searcher",
        system_message=sys_msg,
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(MetaphorSearchTool)
    task = lr.Task(agent, interactive=True)
    ChainlitTaskCallbacks(task)
    cl.user_session.set("task", task)


@cl.on_message
async def on_message(message: cl.Message):
    task = cl.user_session.get("task")
    task.run(message.content)
