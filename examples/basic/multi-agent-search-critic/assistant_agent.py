"""
AssistantAgent takes a user's question, breaks it down into smaller questions
for SearcherAgent to answer, and then presents the final answer; It then considers
feedback from CriticAgent, and may ask more questions or present the final answer
using a corrected reasoning.

Flow:

User Q ->
[L] -> QuestionTool(q1) ->
[A] -> validate, return QuestionTool(q1) ->
... AnswerTool(a1) from SearcherAgent ->
[A] -> AnswerTool(a1) -> natural lang ans for LLM
[L] -> either QuestionTool(q2) or FinalAnswerTool(steps, ans) ->
... if FinalAnswerTool(steps, ans) ->
[A] -> validate, return FinalAnswerTool(steps, ans) with recipient=Critic ->
... FeedbackTool(feedback, suggested_fix) from CriticAgent ->
[A] -> FeedbackTool(feedback, suggested_fix) -> natural lang feedback for LLM
[L] -> either QuestionTool(q2) or FinalAnswerTool(steps, ans) ->
...
"""

from typing import Optional

import typer

import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.tools.orchestration import AgentDoneTool, ForwardTool, PassTool

from .tools import AnswerTool, FeedbackTool, FinalAnswerTool, QuestionTool

app = typer.Typer()


class AssistantAgent(lr.ChatAgent):

    def init_state(self):
        super().init_state()
        self.expecting_question_tool: bool = False
        self.expecting_question_or_final_answer: bool = False  # expecting one of these
        # tools
        self.expecting_search_answer: bool = False
        self.original_query: str | None = None  # user's original query

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        if self.expecting_question_or_final_answer:
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
        elif self.expecting_question_tool:
            return f"""
            You must ask a question using the `question_tool` in the specified format,
            to break down the user's original query: {self.original_query} into 
            smaller questions that can be answered from a web search.
            """

    def question_tool(self, msg: QuestionTool) -> str | PassTool:
        self.expecting_search_answer = True
        self.expecting_question_tool = False
        # return the tool so it is handled by SearcherAgent
        # validated incoming, pass it on
        return PassTool()

    def answer_tool(self, msg: AnswerTool) -> str:
        self.expecting_question_or_final_answer = True
        self.expecting_search_answer = False
        return f"""
        Here is the answer to your question from the web search:
        {msg.answer}
        Now decide whether you want to:
        - present your FINAL answer to the user's ORIGINAL QUERY, OR
        - ask another question using the `question_tool`
            (Maybe REPHRASE the question to get BETTER search results).
        """

    def final_answer_tool(self, msg: FinalAnswerTool) -> ForwardTool | str:
        if not self.expecting_question_or_final_answer:
            return ""
        self.expecting_question_or_final_answer = False
        # insert the original query into the tool, in case LLM forgot to do so.
        msg.query = self.original_query
        # fwd to critic
        return ForwardTool(agent="Critic")

    def feedback_tool(self, msg: FeedbackTool) -> str:
        if msg.suggested_fix == "":
            return AgentDoneTool()
        else:
            self.expecting_question_or_final_answer = True
            # reset question count since feedback may initiate new questions
            return f"""
            Below is feedback about your answer. Take it into account to 
            improve your answer, EITHER by:
            - using the `final_answer_tool` again but with improved REASONING, OR
            - asking another question using the `question_tool`, and when you're 
                ready, present your final answer again using the `final_answer_tool`.
            
            FEEDBACK: {msg.feedback}
            SUGGESTED FIX: {msg.suggested_fix}
            """

    def llm_response(
        self, message: Optional[str | ChatDocument] = None
    ) -> Optional[ChatDocument]:
        if self.original_query is None:
            self.original_query = (
                message if isinstance(message, str) else message.content
            )
            # just received user query, so we expect a question tool next
            self.expecting_question_tool = True

        if self.expecting_question_or_final_answer or self.expecting_question_tool:
            return super().llm_response(message)


def make_assistant_task(
    model: str,
    restart: bool = True,
) -> lr.Task:
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
        you should try to improve your answer based on this feedback,
        possibly by asking more questions.
        """,
        llm=llm_config,
        vecdb=None,
    )
    assistant_agent = AssistantAgent(assistant_config)
    assistant_agent.enable_message(QuestionTool)
    assistant_agent.enable_message(AnswerTool, use=False, handle=True)
    assistant_agent.enable_message(FinalAnswerTool)
    assistant_agent.enable_message(ForwardTool)
    assistant_agent.enable_message(PassTool)
    assistant_agent.enable_message(FeedbackTool, use=False, handle=True)

    assistant_task = lr.Task(
        assistant_agent,
        name="Assistant",
        llm_delegate=True,
        single_round=False,
        interactive=False,
        restart=restart,
    )

    return assistant_task
