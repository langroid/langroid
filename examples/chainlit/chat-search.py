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

import logging
from textwrap import dedent
from typing import Optional

import chainlit as cl

import langroid as lr
from langroid import ChatDocument
from langroid.agent.callbacks.chainlit import (
    add_instructions,
    make_llm_settings_widgets,
    setup_llm,
    update_llm,
)
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool

logger = logging.getLogger(__name__)


def search_system_message(search_tool: lr.ToolMessage) -> str:
    tool_name = search_tool.default_value("request")
    sys_msg = f"""
        You are an astute, self-aware AI assistant, and you are adept at 
        responding to a user's question in one of two ways:
        - If you KNOW the answer from your own knowledge, respond directly.
        - OTHERWISE, request up to 5 results from a web search using 
          the `{tool_name}` tool/function-call.
          In this case you will receive the web search results, and you can 
          then compose a response to the user's question. 
    """
    return sys_msg


class SearchAgent(lr.ChatAgent):
    async def user_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        response = await super().user_response_async(message)
        if response is None:
            return None
        content = response.content
        search_tool = MetaphorSearchTool
        if content.startswith("/"):
            match content[1]:
                case "d":
                    search_tool = DuckduckgoSearchTool
                    self.enable_message(DuckduckgoSearchTool)
                    self.enable_message(MetaphorSearchTool, use=False, handle=False)
                case "m":
                    search_tool = MetaphorSearchTool
                    self.enable_message(MetaphorSearchTool)
                    self.enable_message(DuckduckgoSearchTool, use=False, handle=False)

            self.clear_history(0)
            sys_msg = search_system_message(search_tool)
            self.set_system_message(sys_msg)

            response.content = content[2:]
        return response

    async def agent_response_async(self, message: ChatDocument) -> ChatDocument:
        response = await super().agent_response_async(message)
        if response is None:
            return None
        # ensure tool result goes to LLM
        response.metadata.recipient = lr.Entity.LLM
        return response


async def setup_agent_task(search_tool: lr.ToolMessage):
    """Set up Agent and Task from session settings state."""

    # set up LLM and LLMConfig from settings state
    await setup_llm()
    llm_config = cl.user_session.get("llm_config")
    sys_msg = search_system_message(search_tool)
    config = lr.ChatAgentConfig(
        llm=llm_config,
        name="Searcher",
        system_message=sys_msg,
    )
    agent = SearchAgent(config)
    agent.enable_message(search_tool)
    task = lr.Task(agent, interactive=True)
    cl.user_session.set("agent", agent)
    cl.user_session.set("task", task)


@cl.on_settings_update
async def on_update(settings):
    await update_llm(settings)
    await setup_agent_task(MetaphorSearchTool)


@cl.on_chat_start
async def on_chat_start():
    await add_instructions(
        title="Agent with access to a web search Tool",
        content=dedent(
            """
        Agent uses a tool/fn-call to search the web 
        
        Default search is using DuckDuckGo. You can switch the search to 
        - Duckduckgo by typing `/d` at the start of your question
        - Metaphor by typing `/m` at the start of your question
        
        This is the flow:
        - User asks question
        - Agent LLM uses an internet search tool to generate search results
        - Agent handler recognizes this tool and returns search results
        - User hits `c` to continue
        - Agent LLM composes answer
        
        To change LLM settings, including model name, click the settings symbol on the 
        left of the chat window.        
        """
        ),
    )

    await make_llm_settings_widgets()
    await setup_agent_task(MetaphorSearchTool)


@cl.on_message
async def on_message(message: cl.Message):
    task = cl.user_session.get("task")
    lr.ChainlitTaskCallbacks(task)
    await task.run_async(message.content)
