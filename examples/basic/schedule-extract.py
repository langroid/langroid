"""
Extract schedule/availability information from unstructured text.

Enter vague, unstructured info like:

M-F 8-3pm at home or Tue/Wed 9-1030am at daycare

Run like this -- (omit the -m arg for default gpt-4o-mini LLM)

```bash
uv run examples/basic/schedule-extract.py -m gpt-4o
"""

from typing import Dict, List, Literal, Tuple

from fire import Fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.orchestration import FinalResultTool
from langroid.pydantic_v1 import BaseModel, Field


class Slot(BaseModel):
    start_time: str = Field(..., description="start time of the slot, e.g. 11:30AM")
    end_time: str = Field(..., description="end time of the slot, e.g. 12:30PM")
    location: str = Field(..., description="location of the slot or UNKNOWN")


class DaySchedule(BaseModel):
    """
    A class to represent a day's schedule.
    """

    slots: List[Slot] = Field(..., description="List of time slots for the day")


Weekday = Literal["Mon", "Tue", "Wed", "Thu", "Fri"]


class Availability(BaseModel):
    """
    A class to represent schedule information.
    """

    week_availability: Dict[Weekday, DaySchedule] = Field(
        ...,
        description="""
        Dictionary mapping weekday to DaySchedule,
        where weekday is one of "Mon", "Tue", "Wed", "Thu", "Fri"
        """,
    )


class AvailabilityTool(lr.ToolMessage):
    request: str = "availability_tool"
    purpose: str = """
        To present the available slots from a piece of text.
    """
    availabilities: Availability

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        """
        Example of how to use the tool.
        """
        return [
            (
                """
                I figured out that the availability is 10am-4pm on Mon and Wed at 
                home, and 3-4pm on Monday at daycare
                """,
                cls(
                    availabilities=Availability(
                        week_availability={
                            "Mon": DaySchedule(
                                slots=[
                                    Slot(
                                        start_time="10:00",
                                        end_time="16:00",
                                        location="home",
                                    ),
                                    Slot(
                                        start_time="15:00",
                                        end_time="16:00",
                                        location="daycare",
                                    ),
                                ]
                            ),
                            "Wed": DaySchedule(
                                slots=[
                                    Slot(
                                        start_time="10:00",
                                        end_time="16:00",
                                        location="home",
                                    )
                                ]
                            ),
                        }
                    )
                ),
            )
        ]

    def handle(self) -> str:
        """
        This method is called when the tool is invoked.
        It processes the input and returns the availability information.
        """
        # Here, we would implement the logic to extract availability information
        # from the input text. For this example, we'll just return a placeholder.
        print("Successfully extracted availability information.")
        print(self.availabilities.json(indent=2))
        return FinalResultTool(avails=self.availabilities)


def make_schedule_task(model: str = ""):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model or lm.OpenAIChatModel.GPT4o_MINI,
    )
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=llm_config,
            system_message=f"""
            You are an expert at figuring out schedules from unstructured text.
            You will be given a string that represents availability information.
            Your task is to figure out the available slots and present this info
            using the TOOL `{AvailabilityTool.name()}`, with the `week_availability` 
            field set to a dictionary showing the available slots for certain days
            of the week if any. The string you will get may contain MULTIPLE 
            availabilities for the same day, but at different locations. 
            You have to present the availability information in the `availabilities`
            field, as an Availability object, which is a dictionary mapping
            the day of the week to a DaySchedule object, which is a list of
            Slot objects. The Slot object contains the start time of the slot,
            the duration of the slot in minutes, and the location of the slot.
            """,
        )
    )
    agent.enable_message(AvailabilityTool)
    task = lr.Task(agent, interactive=False, restart=True)[Availability]
    return task


def main(model: str = ""):
    task = make_schedule_task(model)
    while True:
        sched = Prompt.ask("Enter your schedule text")
        avails = task.run(sched, allow_restart=True)
        print(avails)


if __name__ == "__main__":
    Fire(main)
