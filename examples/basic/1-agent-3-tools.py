"""
Barebones example of a single agent using 3 tools.

"""

from typing import List, Tuple

import langroid as lr

# (1) DEFINE THE TOOLS


class UpdateTool(lr.ToolMessage):
    request: str = "update"
    purpose: str = "To update the stored number to the given <number>"
    number: int

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        # Examples that will be compiled into few-shot examples for the LLM.
        # Each example can either be...
        return [
            cls(number=3),  # ... just instances of the tool-class, OR
            (  # ...a tuple of "thought leading to tool", and the tool instance
                "I want to update the stored number to number 4 from the user",
                cls(number=4),
            ),
        ]


class AddTool(lr.ToolMessage):
    request: str = "add"
    purpose: str = "To add the given <number> to the stored number"
    number: int

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        return [
            cls(number=3),
            (
                "I want to add number 10 to the stored number",
                cls(number=10),
            ),
        ]


class ShowTool(lr.ToolMessage):
    request: str = "show"
    purpose: str = "To show the user the stored <number>"

    @classmethod
    def examples(cls) -> List["lr.ToolMessage" | Tuple[str, "lr.ToolMessage"]]:
        return [
            cls(number=3),
            (
                "I want to show the user the stored number 10",
                cls(number=10),
            ),
        ]


# (2) DEFINE THE AGENT, with the tool-handling methods
class NumberAgent(lr.ChatAgent):
    secret: int = 0

    def update(self, msg: UpdateTool) -> str:
        self.secret = msg.number
        return f"Ok I updated the stored number to {msg.number}"

    def add(self, msg: AddTool) -> str:
        self.secret += msg.number
        return f"Added {msg.number} to stored number => {self.secret}"

    def show(self, msg: ShowTool) -> str:
        return f"HERE IS MY STORED SECRET NUMBER! {self.secret}"


# (3) CREATE THE AGENT
agent_config = lr.ChatAgentConfig(
    name="NumberAgent",
    system_message="""
    When the user's request matches one of your available tools, use it, 
    otherwise respond directly to the user.
    """,
)

agent = NumberAgent(agent_config)


# (4) ENABLE/ATTACH THE TOOLS to the AGENT

agent.enable_message(UpdateTool)
agent.enable_message(AddTool)
agent.enable_message(ShowTool)


# (5) CREATE AND RUN THE TASK
task = lr.Task(agent, interactive=True)

"""
Note: try saying these when it waits for user input:

add 10
update 50
add 3
show
"""

task.run()
