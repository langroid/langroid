from typing import List
import langroid as lr
import langroid.language_models as lm
from langroid.pydantic_v1 import Field
from langroid.agent.tools.orchestration import AgentDoneTool, ForwardTool
from fire import Fire
import logging

logger = logging.getLogger(__name__)


class BurifyTool(lr.ToolMessage):
    request: str = "burify_tool"
    purpose: str = "To apply the 'Burify' process to a <number>"
    number: int = Field(..., description="The number (int) to Burify")

    def handle(self) -> str:
        # stateless tool: handler used in BurifyAgent
        return f"Burify this number: {self.number}"


class TonifyTool(lr.ToolMessage):
    request: str = "tonify_tool"
    purpose: str = "To apply the 'Tonify' process to a <number>"
    number: int = Field(..., description="The number (int) to Tonify")

    def handle(self) -> str:
        # stateless tool: handler used in TonifyAgent
        return f"Tonify this number: {self.number}"


class BurifyCheckTool(lr.ToolMessage):
    request: str = "burify_check_tool"
    purpose: str = "To check if the Burify process is complete"
    number: int = Field(..., description="The number (int) to check")
    original_number: int = Field(
        ...,
        description="The original number (int) given to the BurifyAgent",
    )

    def handle(self) -> str:
        # stateless tool
        if self.number == self.original_number + 3:
            return AcceptTool(result=self.number)
        else:
            return BurifyRevisionTool(
                feedback="Burify is NOT complete! Please try again.",
                recipient="Burify",
            )


class TonifyCheckTool(lr.ToolMessage):
    request: str = "tonify_check_tool"
    purpose: str = "To check if the Tonify process is complete"
    number: int = Field(..., description="The number (int) to check")
    original_number: int = Field(
        ...,
        description="The original number (int) given to the TonifyAgent",
    )

    def handle(self):
        # stateless tool
        if self.number == self.original_number * 4:
            return AcceptTool(result=self.number)
        else:
            return TonifyRevisionTool(
                feedback="Tonify is NOT complete! Please try again.",
                recipient="Tonify",
            )


class BurifyRevisionTool(lr.ToolMessage):
    request: str = "burify_revision_tool"
    purpose: str = "To give <feedback> to the  'BurifyAgent' on their Burify Attempt"
    feedback: str = Field(..., description="Feedback for the BurifyAgent")

    def handle(self):
        return f"""
        Below is feedback on your attempt to Burify: 
        <Feedback>
        {self.feedback}
        </Feedback>
        Please try again!
        """


class TonifyRevisionTool(lr.ToolMessage):
    request: str = "tonify_revision_tool"
    purpose: str = "To give <feedback> to the  'TonifyAgent' on their Tonify Attempt"
    feedback: str = Field(..., description="Feedback for the TonifyAgent")

    def handle(self):
        return f"""
        Below is feedback on your attempt to Tonify: 
        <Feedback>
        {self.feedback}
        </Feedback>
        Please try again!
        """


class BurifySubmitTool(lr.ToolMessage):
    request: str = "burify_submit_tool"
    purpose: str = "To submit the result of an attempt of the Burify process"
    result: int = Field(..., description="The result (int) to submit")

    def handle(self):
        return AgentDoneTool(content=str(self.result))


class TonifySubmitTool(lr.ToolMessage):
    request: str = "tonify_submit_tool"
    purpose: str = "To submit the result of an attempt of the Tonify process"
    result: int = Field(..., description="The result (int) to submit")

    def handle(self):
        return AgentDoneTool(content=str(self.result))


class AcceptTool(lr.ToolMessage):
    request: str = "accept_tool"
    purpose: str = "To accept the result of the 'Burify' or 'Tonify' process"
    result: int


class PlannerConfig(lr.ChatAgentConfig):
    name: str = "Planner"
    steps: List[str] = ["Burify", "Tonify"]
    handle_llm_no_tool: str = "You FORGOT to use one of your TOOLs!"
    system_message: str = f"""
    You are a Planner in charge of PROCESSING a given integer through
    a SEQUENCE of 2 processing STEPS, which you CANNOT do by yourself, but you must
    rely on WORKER AGENTS who will do these for you:
    - Burify - will be done by the BurifyAgent
    - Tonify - will be done by the TonifyAgent
    
    In order to INITIATE each process, you MUST use the appropriate TOOLs:
    - `{BurifyTool.name()}` to Burify the number (the tool will be handled by the BurifyAgent)
    - `{TonifyTool.name()}` to Tonify the number (the tool will be handled by the TonifyAgent)
    
    Each of the WORKER AGENTS works like this:
    - The Agent will ATTEMPT a processing step, using the number you give it.
    - You will VERIFY whether the processing step is COMPLETE or NOT
         using the CORRESPONDING CHECK TOOL:
         - check if the Burify step is complete using the `{BurifyCheckTool.name()}`
         - check if the Tonify step is complete using the `{TonifyCheckTool.name()}`
    - If the step is NOT complete, you will ask the Agent to try again,
        by using the CORRESPONDING Revision TOOL where you can include your FEEDBACK: 
        - `{BurifyRevisionTool.name()}` to revise the Burify step
        - `{TonifyRevisionTool.name()}` to revise the Tonify step
    - If you determine (see below) that the step is COMPLETE, you MUST
        use the `{AcceptTool.name()}` to ACCEPT the result of the step.    
    """


class PlannerAgent(lr.ChatAgent):
    current_step: int
    current_num: int
    original_num: int

    def __init__(self, config: PlannerConfig):
        super().__init__(config)
        self.config: PlannerConfig = config
        self.current_step = 0
        self.current_num = 0

    def burify_tool(self, msg: BurifyTool) -> str:
        """Handler of BurifyTool: uses/updates Agent state"""
        self.original_num = msg.number
        logger.warning(f"Planner handled BurifyTool: {self.current_num}")

        return ForwardTool(agent="Burify")

    def tonify_tool(self, msg: TonifyTool) -> str:
        """Handler of TonifyTool: uses/updates Agent state"""
        self.original_num = msg.number
        logger.warning(f"Planner handled TonifyTool: {self.current_num}")

        return ForwardTool(agent="Tonify")

    def accept_tool(self, msg: AcceptTool) -> str:
        """Handler of AcceptTool: uses/updates Agent state"""
        curr_step_name = self.config.steps[self.current_step]
        n_steps = len(self.config.steps)
        self.current_num = msg.result
        if self.current_step == n_steps - 1:
            # last step -> done
            return AgentDoneTool(content=str(self.current_num))

        self.current_step += 1
        next_step_name = self.config.steps[self.current_step]
        return f"""
            You have ACCEPTED the result of the {curr_step_name} step.
            Your next step is to apply the {next_step_name} process
            to the result of the {curr_step_name} step, which is {self.current_num}.
            So use a TOOL to initiate the {next_step_name} process!
            """


class BurifyAgentConfig(lr.ChatAgentConfig):
    name: str = "Burify"
    handle_llm_no_tool: str = f"You FORGOT to use the TOOL `{BurifySubmitTool.name()}`!"
    system_message: str = f"""
    You will receive an integer from your supervisor, to apply
    a process Burify to it, which you are not quite sure how to do,
    but you only know that it involves INCREMENTING the number by 1 a few times
    (but you don't know how many times).
    When you first receive a number to Burify, simply return the number + 1.
    If this is NOT sufficient, you will be asked to try again, and 
    you must CONTINUE to return your last number, INCREMENTED by 1.
    To send your result, you MUST use the TOOL `{BurifySubmitTool.name()}`. 
    """


class TonifyAgentConfig(lr.ChatAgentConfig):
    name: str = "Tonify"
    handle_llm_no_tool: str = f"You FORGOT to use the TOOL `{TonifySubmitTool.name()}`!"
    system_message: str = """
    You will receive an integer from your supervisor, to apply
    a process Tonify to it, which you are not quite sure how to do,
    but you only know that it involves MULTIPLYING the number by 2 a few times
    (and you don't know how many times).
    When you first receive a number to Tonify, simply return the number * 2.
    If this is NOT sufficient, you will be asked to try again, and 
    you must CONTINUE to return your last number, MULTIPLIED by 2.
    To send your result, you MUST use the TOOL `{TonifySubmitTool.name()}`.
    """


def main(model: str = ""):
    planner = PlannerAgent(
        PlannerConfig(
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4_1,
            )
        ),
    )

    planner.enable_message(
        [
            BurifyRevisionTool,
            TonifyRevisionTool,
        ],
        use=True,  # LLM allowed to generate
        handle=False,  # agent cannot handle
    )

    planner.enable_message(  # can use and handle
        [
            AcceptTool,
            BurifyCheckTool,
            TonifyCheckTool,
            BurifyTool,
            TonifyTool,
        ]
    )

    burifier = lr.ChatAgent(
        BurifyAgentConfig(
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4_1,
            )
        )
    )
    burifier.enable_message(
        [
            BurifyTool,
            BurifyRevisionTool,
        ],
        use=False,  # LLM cannot generate
        handle=True,  # agent can handle
    )
    burifier.enable_message(BurifySubmitTool)

    tonifier = lr.ChatAgent(
        TonifyAgentConfig(
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4_1,
            )
        )
    )

    tonifier.enable_message(
        [
            TonifyTool,
            TonifyRevisionTool,
        ],
        use=False,  # LLM cannot generate
        handle=True,  # agent can handle
    )
    tonifier.enable_message(TonifySubmitTool)

    planner_task = lr.Task(planner, interactive=False)
    burifier_task = lr.Task(burifier, interactive=False)
    tonifier_task = lr.Task(tonifier, interactive=False)

    planner_task.add_sub_task(
        [
            burifier_task,
            tonifier_task,
        ]
    )

    # Buify(5) = 5+3 = 8; Tonify(8) = 8*4 = 32
    result = planner_task.run("Sequentially all processes to this number: 5")
    assert "32" in result.content, f"Expected 32, got {result.content}"


if __name__ == "__main__":
    Fire(main)
