"""
Chainlit version of examples/basic/chat-search-assistant-local.py,
with a minor change to enable Chainlit callbacks.
Tested and works ok nous-hermes2-mixtral, but may still have issues.
See that script for details.

You can specify a local in a few different ways, e.g. `-m local/localhost:8000/v1`
or `-m ollama/mistral` etc. See here how to use Langroid with local LLMs:
https://langroid.github.io/langroid/tutorials/local-llm-setup/


"""
from typing import List, Optional, Type

from dotenv import load_dotenv
from textwrap import dedent

import chainlit as cl
import langroid as lr
from langroid.agent.callbacks.chainlit import add_instructions
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.utils.configuration import Settings, set_global


class QuestionTool(lr.ToolMessage):
    request: str = "question_tool"
    purpose: str = "Ask a SINGLE <question> that can be answered from a web search."
    question: str

    @classmethod
    def examples(cls) -> List["ToolMessage"]:
        return [
            cls(question="Which superconductor material was discovered in 2023?"),
            cls(question="What AI innovation did Meta achieve in 2024?"),
        ]


class FinalAnswerTool(lr.ToolMessage):
    request: str = "final_answer_tool"
    purpose: str = "Present the final <answer> to the user's original query."
    answer: str


class AssistantAgent(lr.ChatAgent):
    n_questions: int = 0  # how many questions in THIS round
    has_asked: bool = False  # has ANY question been asked
    original_query: str | None = None

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.USER:
            # either first query from user, or returned result from Searcher
            self.n_questions = 0  # reset search count

        if isinstance(msg, ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            if self.has_asked:
                return f"""
                You must do one of the following:
                - If you are ready with the final answer to the user's ORIGINAL QUERY
                    [ Remember it was: {self.original_query} ],
                  then present this answer using the `final_answer_tool` in the 
                  specified JSON format.
                - If you still need to ask a question, then use the `question_tool`
                  to ask a SINGLE question that can be answered from a web search.
                """
            elif self.original_query is not None:
                return f"""
                You must ask a question using the `question_tool` in the specified format,
                to break down the user's original query: {self.original_query} into 
                smaller questions that can be answered from a web search.
                """

    def question_tool(self, msg: QuestionTool) -> str:
        self.n_questions += 1
        self.has_asked = True
        if self.n_questions > 1:
            # there was already a search, so ignore this one
            return ""
        # valid question tool: re-create it so Searcher gets it
        return msg.to_json()

    def final_answer_tool(self, msg: FinalAnswerTool) -> str:
        return lr.utils.constants.DONE + " " + msg.answer

    async def llm_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if self.original_query is None:
            self.original_query = (
                message if isinstance(message, str) else message.content
            )
        result = await super().llm_response_async(message)
        if result is None:
            return result
        # result.content may contain a premature DONE
        # (because weak LLMs tend to repeat their instructions)
        # We deem a DONE to be accidental if no search query results were received
        if not isinstance(message, ChatDocument) or not (
            message.metadata.sender_name == "Searcher"
        ):
            # no search results received yet, so should NOT say DONE
            if isinstance(result, str):
                return result.content.replace(lr.utils.constants.DONE, "")
            result.content = result.content.replace(lr.utils.constants.DONE, "")
            return result

        return result


class SearcherAgentConfig(lr.ChatAgentConfig):
    search_tool_class: Type[lr.ToolMessage]


class SearcherAgent(lr.ChatAgent):
    n_searches: int = 0
    curr_query: str | None = None

    def __init__(self, config: SearcherAgentConfig):
        super().__init__(config)
        self.config: SearcherAgentConfig = config
        self.enable_message(config.search_tool_class)
        self.enable_message(QuestionTool, use=False, handle=True)

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        if (
            isinstance(msg, ChatDocument)
            and msg.metadata.sender == lr.Entity.LLM
            and self.n_searches == 0
        ):
            search_tool_name = self.config.search_tool_class.default_value("request")
            return f"""
            You forgot to use the web search tool to answer the 
            user's question : {self.curr_query}.
            Please use the `{search_tool_name}` tool 
            using the specified JSON format, then compose your answer.
            """

    def question_tool(self, msg: QuestionTool) -> str:
        self.curr_query = msg.question
        search_tool_name = self.config.search_tool_class.default_value("request")
        return f"""
        User asked this question: {msg.question}.
        Perform a web search using the `{search_tool_name}` tool
        using the specified JSON format, to find the answer.
        """

    async def llm_response_async(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if (
            isinstance(message, ChatDocument)
            and message.metadata.sender == lr.Entity.AGENT
            and self.n_searches > 0
        ):
            # must be search results from the web search tool,
            # so let the LLM compose a response based on the search results
            self.n_searches = 0  # reset search count

            result = await super().llm_response_async(message)
            # Augment the LLM's composed answer with a helpful nudge
            # back to the Assistant
            result.content = f"""
            Here are the web-search results for the question: {self.curr_query}.
            ===
            {result.content}
            ===
            Decide if you want to ask any further questions, for the 
            user's original question.             
            """
            self.curr_query = None
            return result

        # Handling query from user (or other agent)
        result = await super().llm_response_async(message)
        tools = self.get_tool_messages(result)
        if all(not isinstance(t, self.config.search_tool_class) for t in tools):
            # LLM did not use search tool;
            # make the response empty so curr pend msg doesn't get updated,
            # and the agent fallback_handler will remind the LLM
            result.content = lr.utils.constants.DONE
            return result

        self.n_searches += 1
        # result includes a search tool, but may contain DONE in content,
        # so remove that
        result.content = result.content.replace(lr.utils.constants.DONE, "")
        return result


@cl.on_chat_start
async def main(
    debug: bool = False,
    model: str = "ollama/nous-hermes2-mixtral:latest",
    nocache: bool = True,
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
        )
    )
    await add_instructions(
        title="2-Agent Search Assistant",
        content=dedent(
            """
        Enter a complex question; 
        - The Assistant will break it down into smaller questions for the Searcher
        - The Searcher will search the web and compose a concise answer
        Once the Assistant has enough information, it will say DONE and present the answer.
        
        To answer a new question, click "New Chat".        
        """
        ),
    )

    load_dotenv()

    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4_TURBO,
        chat_context_length=16_000,
        temperature=0.2,
        max_output_tokens=200,
        timeout=45,
    )

    assistant_config = lr.ChatAgentConfig(
        system_message="""
        You are a resourceful assistant, able to think step by step to answer complex
        complex questions from the user. You must break down complex questions into
        simpler questions that can be answered by a web search. You must ask me 
        (the user) each question ONE BY ONE, using the `question_tool` in
         the specified format, and I will do a web search and send you
        a brief answer. Once you have enough information to answer my original
        (complex) question, you MUST say DONE and present the answer to me.
        """,
        llm=llm_config,
        vecdb=None,
    )
    assistant_agent = AssistantAgent(assistant_config)
    assistant_agent.enable_message(QuestionTool)
    assistant_agent.enable_message(FinalAnswerTool)

    search_tool_handler_method = DuckduckgoSearchTool.default_value("request")

    search_agent_config = SearcherAgentConfig(
        search_tool_class=DuckduckgoSearchTool,
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a web-searcher. For ANY question you get, you must use the
        `{search_tool_handler_method}` tool/function-call to get up to 5 results.
        Once you receive the results, you must compose a CONCISE answer 
        based on the search results and say DONE and show the answer to me,
        along with references, in this format:
        DONE [... your CONCISE answer here ...]
        SOURCES: [links from the web-search that you used]
        
        EXTREMELY IMPORTANT: DO NOT MAKE UP ANSWERS, ONLY use the web-search results.
        """,
    )
    search_agent = SearcherAgent(search_agent_config)

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
