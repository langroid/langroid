"""
CriticAgent task enforces FinalAnswerTool -> FeedbackTool, i.e.
- incoming msg must be a FinalAnswerTool
- outgoing msg must be a FeedbackTool

Flow:

FinalAnswerTool ->
[A] -> natural lang presentation to LLM
[L] -> FeedbackTool ->
[A] -> AgentDoneTool(FeedbackTool)

"""

import typer
from dotenv import load_dotenv

import langroid as lr
import langroid.language_models as lm
from langroid import ChatDocument
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.utils.configuration import Settings, set_global

from .tools import FeedbackTool, FinalAnswerTool

app = typer.Typer()


class CriticAgent(lr.ChatAgent):
    def init_state(self):
        super().init_state()
        self.expecting_feedback_tool: bool = False

    def final_answer_tool(self, msg: FinalAnswerTool) -> str:
        # received from Assistant. Extract the components as plain text,
        # so that the Critic LLM can provide feedback
        self.expecting_feedback_tool = True

        return f"""
        The user has presented the following query, intermediate steps and final answer
        shown below. Please provide feedback using the `feedback_tool`, 
        with the `feedback` field containing your feedback, and 
        the `suggested_fix` field containing a suggested fix, such as fixing how
        the answer or the steps, or how it was obtained from the steps, or 
        asking new questions.
        
        REMEMBER to set the `suggested_fix` field to an EMPTY string if the answer is 
        VALID.
        
        QUERY: {msg.query}
        
        STEPS: {msg.steps}
        
        ANSWER: {msg.answer}
        """

    def feedback_tool(self, msg: FeedbackTool) -> FeedbackTool:
        # validate, signal DONE, include the tool
        self.expecting_feedback_tool = False
        return AgentDoneTool(tools=[msg])

    def handle_message_fallback(
        self, msg: str | ChatDocument
    ) -> str | ChatDocument | None:
        if self.expecting_feedback_tool:
            return """
            You forgot to provide feedback using the `feedback_tool` 
            on the user's reasoning steps and final answer.
            """


def make_critic_task(model: str):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o,
        chat_context_length=16_000,
        temperature=0.2,
        max_output_tokens=500,
        timeout=45,
    )
    critic_agent_config = lr.ChatAgentConfig(
        llm=llm_config,
        vecdb=None,
        system_message="""
        You excel at logical reasoning and combining pieces of information.
        You will receive a summary of the original query, intermediate steps and final 
        answer.
        You must examine these and provide feedback to the user, using the 
        `feedback_tool`, as follows:
        - If you think the answer and reasoning are valid, 
            simply set the `suggested_fix` field to an empty string "".
        - Otherwise set the `feedback` field to a reason why the answer is invalid,
            and in the `suggested_fix` field indicate how the user can improve the 
            answer, for example by reasoning differently, or asking different questions.
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
    return critic_task


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

        llm_config = lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4o,
            chat_context_length=16_000,
            temperature=0.2,
            max_output_tokens=500,
            timeout=45,
        )

        critic_agent_config = lr.ChatAgentConfig(
            llm=llm_config,
            vecdb=None,
            system_message="""
            You excel at logical reasoning and combining pieces of information.
            The user will send you a summary of the intermediate steps and final answer.
            You must examine these and provide feedback to the user, using the 
            `feedback_tool`, as follows:
            - If you think the answer and reasoning are valid, 
                simply set the `suggested_fix` field to an empty string "".
            - Otherwise set the `feedback` field to a reason why the answer is invalid,
                and in the `suggested_fix` field indicate how the user can improve the 
                answer, for example by reasoning differently, or asking different questions.
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
        final_ans_tool = FinalAnswerTool(
            steps="""
            1. The moon landing was in 1969.
            2. Kennedy was president during 1969.            
            """,
            answer="Kennedy was president during the moon landing.",
        )
        # simulate receiving the tool from Assistant
        final_ans_doc = critic_agent.create_agent_response(
            tool_messages=[final_ans_tool]
        )
        result = critic_task.run(final_ans_doc)
        tools = critic_agent.get_tool_messages(result)
        assert len(tools) == 1
        assert isinstance(tools[0], FeedbackTool)

    app()
