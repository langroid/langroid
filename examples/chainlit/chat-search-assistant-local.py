"""
Chainlit version of examples/basic/chat-search-assistant-local.py,
with a minor change to enable Chainlit callbacks.
Tested and works ok nous-hermes2-mixtral, but may still have issues.
See that script for details.

You can specify a local model in a few different ways, e.g. `groq/llama3-70b-8192`
or `ollama/mistral` etc. See here how to use Langroid with local LLMs:
https://langroid.github.io/langroid/tutorials/local-llm-setup/

Since chainlit does not take cmd line args in the normal way, you have to specify
the model via an environment variable, e.g. `MODEL=ollama/mistral` before the
script is run, e.g.

MODEL=ollama/mistral chainlit run  examples/chainlit/chat-search-assistant-local.py

Note - this is just an example of using an open/local LLM;
 it does not mean that this will work with ANY local LLM.

You may get good results using `groq/llama3-70b-8192` (see the above-linked guide
to using open/local LLMs with Langroid for more details).

"""

import os
from textwrap import dedent
from typing import List, Optional, Type

import chainlit as cl
from dotenv import load_dotenv

import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.callbacks.chainlit import add_instructions
from langroid.agent.tools.metaphor_search_tool import MetaphorSearchTool
from langroid.utils.configuration import Settings, set_global


class QuestionTool(lr.ToolMessage):
    request: str = "question_tool"
    purpose: str = "Ask a SINGLE <question> that can be answered from a web search."
    question: str

    @classmethod
    def examples(cls) -> List[lr.ToolMessage]:
        return [
            cls(question="Which superconductor material was discovered in 2023?"),
            cls(question="What AI innovation did Meta achieve in 2024?"),
        ]


class FinalAnswerTool(lr.ToolMessage):
    request: str = "final_answer_tool"
    purpose: str = """
        Present the intermediate <steps> and 
        final <answer> to the user's original query.
        """
    steps: str
    answer: str

    @classmethod
    def examples(cls) -> List["lr.ToolMessage"]:
        return [
            cls(
                steps="1. Man is mortal. 2. Plato was a man.",
                answer="Plato was mortal.",
            ),
            cls(
                steps="1. The moon landing was in 1969. 2. Kennedy was president "
                "during 1969.",
                answer="Kennedy was president during the moon landing.",
            ),
        ]


class FeedbackTool(lr.ToolMessage):
    request: str = "feedback_tool"
    purpose: str = "Provide <feedback> on the user's answer."
    feedback: str

    @classmethod
    def examples(cls) -> List["lr.ToolMessage"]:
        return [
            cls(feedback=""),
            cls(
                feedback="""
                The answer is invalid because the conclusion does not follow from the
                steps. Please check your reasoning and try again.
                """
            ),
        ]


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
                You may have intended to use a tool, but your JSON format may be wrong.
                
                REMINDER: You must do one of the following:
                - If you are ready with the final answer to the user's ORIGINAL QUERY
                    [ Remember it was: {self.original_query} ],
                  then present your reasoning steps and final answer using the 
                  `final_answer_tool` in the specified JSON format.
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
        if not self.has_asked or self.n_questions > 1:
            # not yet asked any questions, or LLM is currently asking
            # a question (and this is the second one in this turn, and so should
            # be ignored), ==>
            # cannot present final answer yet (LLM may have hallucinated this json)
            return ""
        # valid final answer tool: PASS it on so Critic gets it
        return lr.utils.constants.PASS_TO + "Critic"

    def feedback_tool(self, msg: FeedbackTool) -> str:
        if msg.feedback == "":
            return lr.utils.constants.DONE
        else:
            return f"""
            Below is feedback about your answer. Take it into account to 
            improve your answer, and present it again using the `final_answer_tool`.
            
            FEEDBACK:
            
            {msg.feedback}
            """

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


class CriticAgent(lr.ChatAgent):
    def final_answer_tool(self, msg: FinalAnswerTool) -> str:
        # received from Assistant. Extract the components as plain text,
        # so that the Critic LLM can provide feedback
        return f"""
        The user has presented the following intermediate steps and final answer
        shown below. Please provide feedback using the `feedback_tool`.
        Remember to set the `feedback` field to an empty string if the answer is valid,
        otherwise give specific feedback on what the issues are and how the answer 
        can be improved.
        
        STEPS: {msg.steps}
        
        ANSWER: {msg.answer}
        """

    def feedback_tool(self, msg: FeedbackTool) -> str:
        # say DONE and PASS to the feedback goes back to Assistant to handle
        return lr.utils.constants.DONE + " " + lr.utils.constants.PASS


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
            REMEMBER - you must ONLY answer the user's questions based on 
             results from a web-search, and you MUST NOT ANSWER them yourself.
             
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

            result = await super().llm_response_forget_async(message)
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
        result = await super().llm_response_forget_async(message)
        if result is None:
            return result
        tools = self.get_tool_messages(result)
        if all(not isinstance(t, self.config.search_tool_class) for t in tools):
            # LLM did not use search tool;
            # Replace its response with a placeholder message
            # and the agent fallback_handler will remind the LLM
            result.content = "Did not use web-search tool."
            return result

        self.n_searches += 1
        # result includes a search tool, but may contain DONE in content,
        # so remove that
        result.content = result.content.replace(lr.utils.constants.DONE, "")
        return result


@cl.on_chat_start
async def main(
    debug: bool = True,
    model: str = os.getenv("MODEL", "gpt-4o"),
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
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=16_000,
        temperature=0.2,
        max_output_tokens=500,
        timeout=45,
    )

    assistant_config = lr.ChatAgentConfig(
        system_message="""
        You are a resourceful assistant, able to think step by step to answer
        complex questions from the user. You must break down complex questions into
        simpler questions that can be answered by a web search. You must ask me 
        (the user) each question ONE BY ONE, using the `question_tool` in
         the specified format, and I will do a web search and send you
        a brief answer. Once you have enough information to answer my original
        (complex) question, you MUST present your INTERMEDIATE STEPS and FINAL ANSWER
        using the `final_answer_tool` in the specified JSON format.
        You will then receive FEEDBACK from the Critic, and if needed
        you should try to improve your answer based on this feedback.
        """,
        llm=llm_config,
        vecdb=None,
    )
    assistant_agent = AssistantAgent(assistant_config)
    assistant_agent.enable_message(QuestionTool)
    assistant_agent.enable_message(FinalAnswerTool)
    assistant_agent.enable_message(FeedbackTool, use=False, handle=True)

    search_tool_handler_method = MetaphorSearchTool.name()

    search_agent_config = SearcherAgentConfig(
        search_tool_class=MetaphorSearchTool,
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

    critic_agent_config = lr.ChatAgentConfig(
        llm=llm_config,
        vecdb=None,
        system_message="""
        You excel at logical reasoning and combining pieces of information.
        The user will send you a summary of the intermediate steps and final answer.
        You must examine these and provide feedback to the user, using the 
        `feedback_tool`, as follows:
        - If you think the answer is valid, 
            simply set the `feedback` field to an empty string "".
        - Otherwise set the `feedback` field to a reason why the answer is invalid,
            and suggest how the user can improve the answer.
        """,
    )
    critic_agent = CriticAgent(critic_agent_config)
    critic_agent.enable_message(FeedbackTool)
    critic_agent.enable_message(FinalAnswerTool, use=False, handle=True)
    critic_task = lr.Task(
        critic_agent,
        name="Critic",
        interactive=False,
    )
    assistant_task.add_sub_task([search_task, critic_task])
    cl.user_session.set("assistant_task", assistant_task)


@cl.on_message
async def on_message(message: cl.Message):
    assistant_task = cl.user_session.get("assistant_task")
    lr.ChainlitTaskCallbacks(assistant_task)
    await assistant_task.run_async(message.content)
