import typer
from rich import print

from examples_dev.codechat.code_chat_tools import (
    ShowDirContentsMessage,
    ShowFileContentsMessage,
)
from examples_dev.dockerchat.docker_chat_agent import (
    CODE_CHAT_INSTRUCTIONS,
    PLANNER_SYSTEM_MSG,
    PLANNER_USER_MSG,
    DockerChatAgent,
    DockerChatAgentConfig,
)
from examples_dev.dockerchat.dockerchat_agent_messages import (
    AskURLMessage,
    RunContainerMessage,
    ValidateDockerfileMessage,
)
from langroid.agent.special.recipient_validator_agent import (
    RecipientValidator,
    RecipientValidatorConfig,
)
from langroid.agent.task import Task
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import OpenAIChatModel
from langroid.utils.configuration import Settings, set_global
from langroid.utils.logging import setup_colored_logging

app = typer.Typer()

setup_colored_logging()


def chat(config: DockerChatAgentConfig) -> None:
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4

    print("[blue]Hello I am here to make your dockerfile!")
    print("[cyan]Enter x or q to quit")

    task_messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="""
            You are a helpful assistant, able to think step by step.
            """,
        ),
        LLMMessage(
            role=Role.USER,
            content="""
            You are a devops engineer, and your goal is to create a working dockerfile 
            to containerize a python repo. Think step by step about the information 
            you need, to accomplish your task, and ask me questions for what you need.  
            If I cannot answer, further refine your question into smaller questions.
            Even if the repo already has a dockerfile, you should completely ignore 
            anything contained in it. You are not allowed to look at the existing 
            dockerfile.  
            Do not create a dockerfile until you have all the information you need.
            Before showing me a dockerfile, first ask my permission if you can show it.
            Make sure you ask about only ONE item at a time. Do not combine multiple 
            questions into one request.
            If you are asking for information in plain english, write it as 
            "INFO: <question>". 
            When you are simply making a comment, and not seeking information, 
            write it as 
            "COMMENT: <comment>".
            When you are ready to show a docker file, start the sentence with "READY:".

            Start by asking me for the URL of the github repo.
            """,
        ),
    ]

    agent = DockerChatAgent(config, task_messages)
    agent.enable_message(AskURLMessage)
    agent.enable_message(ValidateDockerfileMessage)
    agent.enable_message(RunContainerMessage)

    agent.planner_agent.enable_message(ShowDirContentsMessage)
    agent.planner_agent.enable_message(ShowFileContentsMessage)
    agent.code_chat_agent.enable_message(ShowDirContentsMessage, use=False, handle=True)
    agent.code_chat_agent.enable_message(
        ShowFileContentsMessage, use=False, handle=True
    )

    # set up tasks and their hierarchy
    docker_task = Task(
        agent,
        name="DockerExpert",
        llm_delegate=True,
        single_round=False,
        only_user_quits_root=True,
    )
    planner_task = Task(
        agent.planner_agent,
        name="Planner",
        llm_delegate=True,
        single_round=False,
        system_message=PLANNER_SYSTEM_MSG,
        user_message=PLANNER_USER_MSG,
    )
    code_chat_task = Task(
        agent.code_chat_agent,
        name="Coder",
        llm_delegate=False,
        single_round=True,
        system_message=CODE_CHAT_INSTRUCTIONS,
    )

    validator_config = RecipientValidatorConfig(
        vecdb=None,
        llm=None,
        name="Validator",
        recipients=["DockerExpert", "Coder"],
        tool_recipient="Coder",
    )
    validator_agent = RecipientValidator(validator_config)
    validator_task = Task(validator_agent, single_round=True)
    docker_task.add_sub_task(planner_task)
    planner_task.add_sub_task([validator_task, code_chat_task])
    docker_task.run()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    gpt3: bool = typer.Option(False, "--gpt3_5", "-3", help="use gpt-3.5"),
    gpt4: bool = typer.Option(False, "--gpt4", "-4", help="use gpt-4"),
    fn_api: bool = typer.Option(False, "--fn_api", "-f", help="use functions api"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nofunc: bool = typer.Option(False, "--nofunc", "-nf", help="no function_call"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    no_human: bool = typer.Option(
        False, "--nohuman", "-nh", help="no human input (for stepping in debugger)"
    ),
) -> None:
    gpt4 = gpt4  # ignore since we default to gpt4 anyway
    config = DockerChatAgentConfig(
        debug=debug,
        gpt4=True,
        cache=not nocache,
        use_functions_api=fn_api,
        use_tools=not fn_api,
    )

    set_global(
        Settings(
            debug=debug,
            gpt3_5=gpt3,
            nofunc=nofunc,  # if true, use the GPT4 model before fn calls release
            cache=not nocache,
            interactive=not no_human,
            stream=not no_stream,
        )
    )

    chat(config)


if __name__ == "__main__":
    app()
