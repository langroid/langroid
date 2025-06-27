"""
SearcherAgent flow:

[A] stands for Agent response (i.e. agent_response)
[L] stands for LLM response (i.e. llm_response)

QuestionTool ->
[A] -> natural lang question for LLM ->
[L] -> DuckduckgoSearchTool ->
[A] -> results ->
[L] -> AnswerTool(results) ->
[A] -> AgentDoneTool(AnswerTool)

Note that this Agent's task enforces QuestionTool -> AnswerTool, i.e.
- incoming msg must be a QuestionTool
- outgoing msg must be an AnswerTool
"""

from typing import Optional

import typer
from dotenv import load_dotenv

import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.utils.configuration import Settings, set_global

from .tools import AnswerTool, QuestionTool

app = typer.Typer()

# class MyDDGSearchTool(DuckduckgoSearchTool):
#     request = "my_ddg_search"


class SearcherAgent(lr.ChatAgent):
    def init_state(self):
        super().init_state()
        self.curr_query: str | None = None
        self.expecting_search_results: bool = False
        self.expecting_search_tool: bool = False

    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        self.config = config
        self.enable_message(MetaphorSearchTool)  # DuckduckgoSearchTool
        self.enable_message(QuestionTool, use=False, handle=True)
        # agent is producing AnswerTool, so LLM should not be allowed to "use" it
        self.enable_message(AnswerTool, use=False, handle=True)

    def duckduckgo_search(self, msg: DuckduckgoSearchTool) -> str:
        """Override the DDG handler to update state"""
        self.expecting_search_results = True
        self.expecting_search_tool = False
        return msg.handle()

    def metaphor_search(self, msg: MetaphorSearchTool) -> str:
        """Override the Metaphor handler to update state"""
        self.expecting_search_results = True
        self.expecting_search_tool = False
        return msg.handle()

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        # we're here because msg has no tools
        if self.curr_query is None:
            # did not receive a question tool, so short-circuit and return None
            return None
        if self.expecting_search_tool:
            search_tool_name = MetaphorSearchTool.default_value("request")
            return f"""
            You forgot to use the web search tool`{search_tool_name}`  
            to answer the user's question : {self.curr_query}!!
            REMEMBER - you must ONLY answer the user's questions based on 
             results from a web-search, and you MUST NOT ANSWER them yourself.
             
            Please use the `{search_tool_name}` tool 
            using the specified JSON format, then compose your answer based on 
            the results from this web-search tool.
            """

    def question_tool(self, msg: QuestionTool) -> str:
        self.curr_query = msg.question
        self.expecting_search_tool = True
        search_tool_name = MetaphorSearchTool.default_value("request")
        return f"""
        User asked this question: {msg.question}.
        Perform a web search using the `{search_tool_name}` tool
        using the specified JSON format, to find the answer.
        """

    def answer_tool(self, msg: AnswerTool) -> AgentDoneTool:
        # signal DONE, and return the AnswerTool
        return AgentDoneTool(tools=[msg])

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if self.expecting_search_results:
            # message must be search results from the web search tool,
            # so let the LLM compose a response based on the search results

            curr_query = self.curr_query
            # reset state
            self.curr_query = None
            self.expecting_search_results = False
            self.expecting_search_tool = False

            result = super().llm_response_forget(message)

            # return an AnswerTool containing the answer,
            # with a nudge meant for the Assistant
            answer = f"""
                Here are the web-search results for the question: {curr_query}.
                ===
                {result.content}
                """

            ans_tool = AnswerTool(answer=answer)
            # cannot return a tool, so use this to create a ChatDocument
            return self.create_llm_response(tool_messages=[ans_tool])

        # Handling query from user (or other agent) => expecting a search tool
        result = super().llm_response_forget(message)
        return result


def make_search_task(model: str):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=16_000,
        temperature=0.2,
        max_output_tokens=500,
        timeout=45,
    )

    search_tool_handler_method = MetaphorSearchTool.default_value("request")

    search_agent_config = lr.ChatAgentConfig(
        llm=llm_config,
        vecdb=None,
        system_message=f"""
        You are a web-searcher. For ANY question you get, you must use the
        `{search_tool_handler_method}` tool/function-call to get up to 5 results.
        Once you receive the results, you must compose a CONCISE answer 
        based on the search results and present the answer in this format:
        ANSWER: [... your CONCISE answer here ...]
        SOURCES: [links from the web-search that you used]
        
        EXTREMELY IMPORTANT: DO NOT MAKE UP ANSWERS, ONLY use the web-search results.
        """,
    )
    search_agent = SearcherAgent(search_agent_config)
    search_task = lr.Task(
        search_agent,
        name="Searcher",
        llm_delegate=True,
        single_round=False,
        interactive=False,
    )
    return search_task


if __name__ == "__main__":

    @app.command()
    def main(
        debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
        model: str = typer.Option("", "--model", "-m", help="model name"),
        nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    ) -> None:
        set_global(
            Settings(
                debug=debug,
                cache=not nocache,
            )
        )

        load_dotenv()

        search_task = make_search_task(model)
        # simulate an incoming message to this Task, from the Assistant agent
        q_doc = search_task.agent.create_agent_response(
            tool_messages=[QuestionTool(question="Who was Beethoven's teacher?")]
        )
        result = search_task.run(q_doc)
        tools = search_task.agent.get_tool_messages(result)
        assert len(tools) == 1
        assert isinstance(tools[0], AnswerTool)

    app()
