"""
A customer calls, and we want to accomplish these tasks:

- Identify: get name, ssn, and intent (schedule or cancel)
- Schedule: schedule appointment
- Cancel: cancel appointment

Run like this:
python3 examples/experimental/customer-support.py
"""

from typing import List
from enum import Enum

from langroid.experimental.team import Team, TaskNode
import langroid as lr
import langroid.language_models as lm
from langroid.pydantic_v1 import BaseModel, Field
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.language_models.openai_gpt import OpenAIGPTConfig
import logging

# Fix logging level type
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class CallerInfo(BaseModel):
    name: str = Field(..., title="Name of the caller")
    ssn: str = Field(..., title="Social Security Number of the caller")


class Intent(str, Enum):
    schedule = "schedule"
    cancel = "cancel"


class CallerInfoTool(lr.ToolMessage):
    request: str = "caller_info_tool"
    purpose: str = """
        Get caller's <patient_name>, <ssn> and <intent> (Schedule or Cancel)
    """

    patient_name: str
    ssn: str
    intent: Intent

    def handle(self) -> str:
        return AgentDoneTool(
            # Note content is passed to listeners
            content=f"""
            Caller {self.patient_name} with SSN {self.ssn} wants to {self.intent}
            """,
            # returning tool is useful in orchestrating the workflow deterministically
            # without having to parse the content
            tools=[self],
        )


class ScheduleTool(lr.ToolMessage):
    request: str = "schedule_tool"
    purpose: str = """
    Schedule an appointment for a caller with a given <patient_name> and <ssn>,
    with <doctor> on <date>
    """

    patient_name: str
    ssn: str
    doctor: str
    date: str

    def handle(self) -> str:
        print(
            f"""
         Got it! I'll Schedule {self.patient_name}'s appointment 
         with {self.doctor} on {self.date}
         """
        )
        # insert appropriate API call here, using self.* fields
        return AgentDoneTool(
            content=f"""
            Appt scheduled for {self.patient_name} with {self.doctor} on {self.date}
            """,
            tools=[self],
        )


class CancelTool(lr.ToolMessage):
    request: str = "cancel_tool"
    purpose: str = """
    Cancel appointment for a caller with a given <patient_name> and <ssn> 
    with <doctor> on <date>
    """

    patient_name: str
    ssn: str
    doctor: str
    date: str

    def handle(self) -> str:
        print(
            f"""
         Got it! I'll cancel {self.patient_name}'s appointment 
         with {self.doctor} on {self.date}
         """
        )
        # insert appropriate API call here, making use of self.* fields
        return AgentDoneTool(
            content=f"""
            Appt canceled for {self.patient_name} with {self.doctor} on {self.date}
            """,
            tools=[self],
        )


def make_task(name: str, tool: lr.ToolMessage, sys: str) -> TaskNode:
    llm_config = OpenAIGPTConfig(
        chat_model=lm.OpenAIChatModel.GPT4o,
    )
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            name=name,
            system_message=sys,
        )
    )
    agent.enable_message(tool)
    task = lr.Task(agent, interactive=True, only_user_quits_root=False)
    return TaskNode(task)


def make_intake_task(name="Intake") -> TaskNode:
    return make_task(
        name,
        CallerInfoTool,
        sys=f"""
            You are operating a clinic front-desk.
            Your job is to ask enough questions to get the 
            caller's name and SSN, and INTENT (i.e. schedule or cancel appointment),
            and return it using the TOOL `{CallerInfoTool.name()}`
            """,
    )


def make_schedule_task(name="Schedule") -> TaskNode:
    return make_task(
        name,
        ScheduleTool,
        sys=f"""
            You are operating a clinic front-desk, responsible for SCHEDULING
            an appointment for the caller, with a doctor.
            From the user identified in your context, elicit which doctor
            and what date they would like to cancel the appointment,
            and cancel it using the TOOL `{CancelTool.name()}`.
            """,
    )


def make_cancel_task(name="Cancel") -> TaskNode:
    return make_task(
        name,
        CancelTool,
        sys=f"""
            You are operating a clinic front-desk, responsible for CANCELLING
            an appointment for the caller, with a doctor.
            From the user identified in your context, elicit which doctor
            and what date they would like to cancel an appointment,
            and use the TOOL `{CancelTool.name()}` to return this info.
            """,
    )


class ClinicTeam(Team):
    """Custom team with custom run method, no scheduler involved"""

    def __init__(
        self,
        name: str,
    ) -> None:
        super().__init__(name=name)
        self.intake_task = make_intake_task()
        self.scheduler_task = make_schedule_task()
        self.cancel_task = make_cancel_task()

        # set up the listening relations
        # scheduler, cancel tasks should get info from intake task
        self.scheduler_task.listen(self.intake_task)
        self.cancel_task.listen(self.intake_task)

    def run(self, input: str | lr.ChatDocument | None = None) -> List[lr.ChatDocument]:
        caller_result: lr.ChatDocument = self.intake_task.run("get_started")[0]

        caller_info: CallerInfoTool = caller_result.tool_messages[0]

        # the scheduler and cancel tasks will have this info since
        # they listen to the id task
        if caller_info.intent == Intent.schedule.value:
            self.scheduler_task.run()
        else:
            self.cancel_task.run()


if __name__ == "__main__":
    intake_task = make_intake_task()
    scheduler_task = make_schedule_task()
    cancel_task = make_cancel_task()

    # set up the listening relations
    # scheduler, cancel tasks should get info from intake task
    scheduler_task.listen(intake_task)
    cancel_task.listen(intake_task)

    caller_result: lr.ChatDocument = intake_task.run("get_started")[0]

    caller_info: CallerInfoTool = caller_result.tool_messages[0]

    # the scheduler and cancel tasks will have this info since
    # they listen to the id task
    if caller_info.intent == Intent.schedule.value:
        scheduler_task.run()
    else:
        cancel_task.run()
