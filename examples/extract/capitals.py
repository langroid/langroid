"""
Extract structured information from a passage using a tool/function.


python3 examples/extract/capitals.py

"""

from typing import List

from rich import print

import langroid as lr
from langroid.pydantic_v1 import BaseModel


class City(BaseModel):
    name: str
    country: str
    population: int


class CitiesData(BaseModel):
    cities: List[City]


PASSAGE = """
Berlin is the capital of Germany. It has a population of 3,850,809. 
Paris, France's capital, has 2.161 million residents. 
Lisbon is the capital and the largest city of Portugal with the population of 504,718.
"""


class CitiesMessage(lr.agent.ToolMessage):
    """Tool/function to use to extract/present structured capitals info"""

    request: str = "capital_info"
    purpose: str = "Collect information about city <capitals> from a passage"
    capitals: List[CitiesData]

    def handle(self) -> str:
        """Tool handler: Print the info about the capitals.
        Any format errors are intercepted by Langroid and passed to the LLM to fix."""
        print(f"Correctly extracted Capitals Info: {self.capitals}")
        return "DONE"  # terminates task


agent = lr.ChatAgent(
    lr.ChatAgentConfig(
        name="CitiesExtractor",
        use_functions_api=True,
        use_tools=False,
        system_message=f"""
        From the passage below, extract info about city capitals, and present it 
        using the `capital_info` tool/function.
        PASSAGE: {PASSAGE}
        """,
    )
)
# connect the Tool to the Agent, so it can use it to present extracted info
agent.enable_message(CitiesMessage)

# wrap the agent in a task and run it
task = lr.Task(
    agent,
    interactive=False,
)

task.run()
