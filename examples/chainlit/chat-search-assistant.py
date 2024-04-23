"""
Chainlit version of examples/basic/chat-search-assistant.py,
with only a small change to add the Chainlit callbacks.

See that script for details.

Run like this:

chainlit run examples/chainlit/chat-search-assistant.py
"""

from dotenv import load_dotenv
from textwrap import dedent

import chainlit as cl
import langroid as lr
from langroid.agent.callbacks.chainlit import add_instructions
import langroid.language_models as lm
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.utils.configuration import Settings, set_global


@cl.on_chat_start
async def main(
    debug: bool = False,
    # e.g. ollama/mistral or local/localhost:5000/v1 default is GPT4_TURBO
    model: str = "",
    provider: str = "ddg",  # or "google", "metaphor"
    nocache: bool = False,
):
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    load_dotenv()

    await add_instructions(
        title="2-Agent Search Assistant",
        content=dedent(
            """
        Enter a complex question; 
        - The Assistant will break it down into smaller questions for the Searcher
        - The Searcher will search the web and compose a concise answer
        
        Once the Assistant has enough information, it will say DONE and present the answer.        
        """
        ),
    )

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        chat_context_length=8_000,
        temperature=0,
        max_output_tokens=200,
        timeout=45,
    )

    assistant_config = lr.ChatAgentConfig(
        system_message="""
        You are a resourceful assistant, able to think step by step to answer
        complex questions from the user. You must break down complex questions into
        simpler questions that can be answered by a web search. You must ask me 
        (the user) each question ONE BY ONE, and I will do a web search and send you
        a brief answer. Once you have enough information to answer my original
        (complex) question, you MUST say DONE and present the answer to me.
        """,
        llm=llm_config,
        vecdb=None,
    )
    assistant_agent = lr.ChatAgent(assistant_config)

    match provider:
        case "google":
            search_tool_class = GoogleSearchTool
        case "metaphor":
            from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool

            search_tool_class = MetaphorSearchTool
        case "ddg":
            search_tool_class = DuckduckgoSearchTool
        case _:
            raise ValueError(f"Unsupported provider {provider} specified.")

    search_tool_handler_method = search_tool_class.default_value("request")

    search_agent_config = lr.ChatAgentConfig(
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a web-searcher. For any question you get, you must use the
        `{search_tool_handler_method}` tool/function-call to get up to 5 results.
        Once you receive the results, you must compose a CONCISE answer 
        based on the search results and say DONE and show the answer to me,
        in this format:
        DONE [... your CONCISE answer here ...]
        """,
    )
    search_agent = lr.ChatAgent(search_agent_config)
    search_agent.enable_message(search_tool_class)

    assistant_task = lr.Task(
        assistant_agent,
        name="Assistant",
        llm_delegate=True,
        single_round=False,
        interactive=False,
    )
    search_task = lr.Task(
        search_agent,
        name="Searcher",
        llm_delegate=True,
        single_round=False,
        interactive=False,
    )
    assistant_task.add_sub_task(search_task)
    cl.user_session.set("assistant_task", assistant_task)


@cl.on_message
async def on_message(message: cl.Message):
    assistant_task = cl.user_session.get("assistant_task")
    lr.ChainlitTaskCallbacks(assistant_task, message)
    await assistant_task.run_async(message.content)
