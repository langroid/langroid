import langroid as lr
import langroid.language_models as lm
import logging
from langroid.pydantic_v1 import BaseModel, Field
from langroid.utils.configuration import settings
from langroid.agent.tools.orchestration import ForwardTool, AgentDoneTool, ResultTool
from langroid.agent.task import TaskConfig
import datasets
from rich.prompt import Prompt
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
lr.utils.logging.setup_colored_logging()

# MODEL = lm.OpenAIChatModel.GPT4o
MODEL = "ollama/llama3:8b"

CP_NAME = "CP"
PA_NAME = "PA"


class ExpectedText(BaseModel):

    final_decision: str = Field(..., description="binary yes/no answer")
    long_answer: str = Field(..., description="explanation for the final decision")


# inherit only from ToolMessage, not (ResultTool, ForwardTool)
class ExpectedTextTool(lr.ToolMessage):
    request: str = "expected_text_tool"
    purpose: str = """
    To write the final <expectedText> AFTER having a multi-turn discussion about ... with the Assistant Agent,
    with all fields of the appropriate type filled out;
    SIMPLY TALK IN NATURAL LANGUAGE.
    """
    expectedText: ExpectedText

    def handle(self) -> ResultTool:
        """Handle LLM's structured output if it matches ExpectedText structure"""
        print("SUCCESS! Got Valid ExpectedText Info")

        return ResultTool(status="**DONE!**", expectedText=self.expectedText)

    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | lr.ChatDocument
    ) -> ForwardTool:
        """
        We end up here when there was no recognized tool msg from the LLM;
        In this case forward the message to the Assistant agent (PA) using ForwardTool.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return ForwardTool(agent=PA_NAME)


# inherit from ToolMessage, not AgentDoneTool;
# But this tool is not really needed since we can set the assistant_task with
# single_round=True, and its response will immediately be returned to the senior agent.
class DiscussionTextTool(lr.ToolMessage):
    """Write an answer to senior agent"""

    request: str = "discussion_text_tool"
    purpose: str = """
    To express your <answer> AFTER receiving a message from the senior agent (CP);
    SIMPLY TALK IN NATURAL LANGUAGE.
    """
    answer: str

    def handle(self) -> AgentDoneTool:
        return AgentDoneTool(content=self.answer)

    @staticmethod
    def handle_message_fallback(
        agent: lr.ChatAgent, msg: str | lr.ChatDocument
    ) -> AgentDoneTool:
        """
        We end up here when there was no recognized tool msg from the LLM;
        In this case forward the message to the user using ForwardTool.
        """
        if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
            return AgentDoneTool(content=msg.content)


# Define fixed system messages outside of the question-loop
# Pass each question as senior_task.run(question)

SENIOR_SYS_MSG = f"""You are Dr. X, the Chief Physician, collaborating with Dr. Y, your assistant. 
                    Your task is to come up with concise answers to medical questions. 
                    To make better decisions, when you receive a question, you should follow a TWO-PHASE procedure:
                    
                    PHASE 1: Ask your assistant NATURAL LANGUAGE questions (NO TOOLS), which may span
                        MULTIPLE ROUNDS. ASK EXACTLY ONE QUESTION in each round. DO NOT ASK MULTIPLE QUESTIONS AT ONCE.
                        Avoid fabricating interactions or simulating dialogue with Dr. Y. 
                        Instead, clearly articulate your questions or follow-ups, analyze Dr. Y's responses, 
                        and use this information to guide your decision-making.
                    PHASE 2: Once you have gathered sufficient information, return your final decision 
                        using the TOOL `{ExpectedTextTool.name()}`:
                        - `final_decision` should be your BINARY yes/no answer
                        - `long_answer` should provide a detailed explanation for your final decision.
                    DO NOT mention the TOOL to Dr. Y. It is your responsibility to write and submit the expectedText.
                    """

ASSISTANT_SYS_MSG = """You are Dr. Y, an assistant physician working under the supervision of Dr. X, the chief physician.                                     
                            Your role is to check a medical question and provide your initial evaluation, which will guide Dr. X 
                            toward finalizing the answer. Dr X may ask you a series of questions, and you should respond
                            based on your expertise and the preceding discussion.
                            ### Instructions:
                            1. Ensure your evaluation is clear, precise, and structured to facilitate an informed discussion. 
                            2. In each round of the discussion, limit yourself to a CONCISE message.
                        ### Process:
                        You will first receive a message from Dr. X, asking for your initial assessment. 
                        Afterward, you can follow up in each discussion round to collaboratively refine the answer.
                        """

# no need for discussion tool -- commenting out
#                        f"3. ALWAYS use the TOOL `{DiscussionTextTool.name()}` to return your answers." +


# no need to inherit this from ChatAgent - it's not a real agent, i.e.e
# we are not trying to use its llm_response or other *_response methods
class ChatManager:
    def __init__(
        self,
        d: bool = False,  # pass -d to enable debug mode (see prompts etc)
        nc: bool = False,  # pass -nc to disable cache-retrieval (i.e. get fresh answer)
    ):
        settings.debug = d
        settings.cache = not nc

        self.ass_lm_config = lm.OpenAIGPTConfig(
            chat_model=MODEL,
            chat_context_length=1040_000,
            seed=42,
        )
        self.ass_agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                name=PA_NAME,
                llm=self.ass_lm_config,
                system_message=ASSISTANT_SYS_MSG,
            ),
        )
        # no need for the DiscussionTextTool
        # self.ass_agent.enable_message(DiscussionTextTool)
        self.senior_lm_config = lm.OpenAIGPTConfig(
            chat_model=MODEL,
            chat_context_length=1040_000,
            seed=42,
        )
        self.senior_agent = lr.ChatAgent(
            lr.ChatAgentConfig(
                llm=self.senior_lm_config,
                name=CP_NAME,
                system_message=SENIOR_SYS_MSG,
            ),
        )
        self.senior_agent.enable_message(ExpectedTextTool)

    # This will not be used since MainChatAgent is not a real agent with a task wrapper;
    # it's just managing the other 2 agents.

    # def handle_message_fallback(self, msg: str | lr.ChatDocument) -> ForwardTool:
    #     """
    #     We'd be here if there were no recognized tools in the incoming msg.
    #     If this was from LLM, forward to Assistant agent.
    #     """
    #     if isinstance(msg, lr.ChatDocument) and msg.metadata.sender == lr.Entity.LLM:
    #         return ForwardTool(agent="PA")

    def start_chat(
        self, question: str
    ) -> ExpectedText:  # this is our main function to start the chat
        task_config = TaskConfig(inf_loop_cycle_len=0)
        self.ass_task = lr.Task(
            self.ass_agent,
            llm_delegate=True,
            interactive=False,
            single_round=True,  # set to True, eliminates need for DiscussionTextTool
            restart=True,  # ignored for a subtask
            config=task_config,
        )

        self.senior_task = lr.Task(
            self.senior_agent,
            llm_delegate=True,
            interactive=False,
            single_round=False,
            restart=True,
            config=task_config,
        )[
            ResultTool
        ]  # specialize task to strictly return ResultTool

        self.senior_task.add_sub_task(self.ass_task)
        response_tool: ResultTool | None = self.senior_task.run(
            question, turns=100
        )  # dialogues usually take less than 70 turns

        if response_tool is None:
            print(
                """
                RETURNED ANSWER DOES NOT HAVE A TOOL! LLM DID NOT FORMAT THE DISCHARGE TEXT!!!
                """
            )
            return ExpectedText(final_decision="unknown", long_answer="null")
        else:
            print("ResultTool has been received successfully!!!")
            print(response_tool.expectedText)
            return response_tool.expectedText


if __name__ == "__main__":
    chatAgent = ChatManager()

    pubmed_ds = pd.DataFrame(
        datasets.load_dataset(
            "burcusayin/pubmedqa_binary_with_plausible_gpt4_long_answers"
        )["test"]
    )
    model_responses = []
    questions = [row.QUESTION for row in pubmed_ds.itertuples()]
    print(f"Processing {len(questions)} questions")
    for i, question in enumerate(questions[:10]):
        print(f"QUESTION: {question}")
        response: ExpectedText = chatAgent.start_chat(question=question)
        model_responses.append(response)
        print(f"Got response {i}: {response.final_decision}")
        cont = Prompt.ask("Continue? (y/n)", default="y")
        if cont.lower() != "y":
            break

        # we save the responses here.
