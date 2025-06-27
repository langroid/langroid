"""
Toy example showing how to combine results from multiple tasks running concurrently.

- main agent/task uses `multi_task_tool` tool to specify what to send to tasks t2, t3
- t2, t3 are run concurrently
- results from t2, t3 are combined and returned to main agent/task
- main agent/task then uses the combined results to generate a final response
"""

from typing import Dict

from fire import Fire
from rich.prompt import Prompt

import langroid as lr
import langroid.language_models as lm
from langroid.agent.batch import run_batch_task_gen
from langroid.agent.tools.orchestration import AgentDoneTool
from langroid.utils.globals import GlobalState

CITY_AGENT_NAME = "CityAgent"
NAME_AGENT_NAME = "NameAgent"


class MyGlobalState(GlobalState):
    name_task_map: Dict[str, str] = {}


class MultiTaskTool(lr.ToolMessage):
    request: str = "multi_task_tool"
    purpose: str = """
        Specify messages to send to multiple agents, via <agent_msgs>
        which is a dict mapping agent names to messages.
    """
    agent_msgs: Dict[str, str]

    def handle(self) -> AgentDoneTool:
        inputs = list(self.agent_msgs.values())
        agent_names = list(self.agent_msgs.keys())
        name_task_map = MyGlobalState.get_value("name_task_map")
        tasks = [name_task_map[name] for name in agent_names]

        def result2content_fn(chat_doc: lr.ChatDocument) -> str:
            return chat_doc.content

        def task_gen(i: int):  # task generator
            return tasks[i]

        results = run_batch_task_gen(task_gen, inputs, output_map=result2content_fn)
        output = "\n".join(
            f"{agent_names[i]}: {result}" for i, result in enumerate(results)
        )
        return AgentDoneTool(content=output)


def chat(model: str = "") -> None:

    cities_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name=CITY_AGENT_NAME,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4o,
            ),
            system_message="""
            You'll receive a sentence. 
            Simply show the the list of cities in the sentence if any,
            as a comma-separated list, say nothing else.
            If no cities are found, say "NO CITIES".
            """,
        )
    )

    names_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name=NAME_AGENT_NAME,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4o,
            ),
            system_message="""
            You'll receive a sentence. 
            Simply show the the list of names in the sentence if any,
            as a comma-separated list, say nothing else.
            If no names are found, say "NO NAMES".
            """,
        )
    )

    cities_task = lr.Task(cities_agent, interactive=False, single_round=True)
    names_task = lr.Task(names_agent, interactive=False, single_round=True)

    MyGlobalState.set_values(
        name_task_map={CITY_AGENT_NAME: cities_task, NAME_AGENT_NAME: names_task}
    )

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            name="MainAgent",
            llm=lm.OpenAIGPTConfig(
                chat_model=model or lm.OpenAIChatModel.GPT4o,
            ),
            system_message=f"""
            You'll receive a sentence. Your end-goal is to get the 
            list of cities and names mentioned in the sentence,
            BUT YOU DO NOT KNOW HOW TO EXTRACT THEM;
            you'll receive the help of {CITY_AGENT_NAME} and {NAME_AGENT_NAME} for this.
            You must use the TOOL `{MultiTaskTool.name()}` to send the sentence 
            to them.
            Once you receive the consolidated results,
            say "DONE" and show the list of cities and names.
            """,
        )
    )

    agent.enable_message(MultiTaskTool)

    task = lr.Task(agent, interactive=False, single_round=False)

    sentence = Prompt.ask(
        "Enter a sentence, to extract cities and names from",
        default="Satoshi will meet Alice in New York and Bob in London",
    )

    result = task.run(sentence)

    print(
        f"""
        [bold]Final Result:[/bold]
        {result}
        """
    )


if __name__ == "__main__":
    Fire(chat)
