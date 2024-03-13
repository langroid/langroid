"""
Basic single-agent chat example, using a web Search Tool, without streaming.

DEPCRECATED: Script kept only for reference. The better way is shown in
chat-search.py, which uses ChainlitTaskCallbacks.

- User asks a question
- LLM either responds directly or generates a Metaphor web search Tool/function-call
    - if Tool used:
         - Agent handler recognizes this tool and returns search results
         - LLM sees search results and composes a response.
- user asks another question


After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-search-no-callback.py
"""

import chainlit as cl

import langroid as lr
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool


@cl.step(name="LLM Response")
async def llm_response(msg: str) -> lr.ChatDocument:
    agent: lr.ChatAgent = cl.user_session.get("agent")
    response = await agent.llm_response_async(msg)
    return response


@cl.step(name="Agent Tool Handler")
async def agent_response(msg: lr.ChatDocument) -> lr.ChatDocument:
    agent: lr.ChatAgent = cl.user_session.get("agent")
    response = await agent.agent_response_async(msg)
    return response


@cl.on_chat_start
async def on_chat_start():
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
        system_message=sys_msg,
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(MetaphorSearchTool)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent: lr.ChatAgent = cl.user_session.get("agent")
    msg = cl.Message(content="")
    # expecting a tool here
    response = await llm_response(message.content)
    if agent.has_tool_message_attempt(response):
        search_results = await agent_response(response)
        response = await llm_response(search_results)
    msg.content = response.content
    await msg.send()
