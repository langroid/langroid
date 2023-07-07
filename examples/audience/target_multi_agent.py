from llmagent.utils.logging import setup_colored_logging
from llmagent.agent.task import Task
from llmagent.agent.special.validator_agent import ValidatorAgent, ValidatorAgentConfig
from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from examples.audience.agents.segmentor import Segmentor, SegmentorConfig
from llmagent.language_models.openai_gpt import OpenAIChatModel
from llmagent.utils.configuration import update_global_settings, set_global, Settings

import typer

from rich.console import Console
from rich import print
from rich.prompt import Prompt

console = Console()
app = typer.Typer()

setup_colored_logging()


def chat(config: SegmentorConfig) -> None:
    update_global_settings(config, keys=["debug", "stream", "cache"])
    if config.gpt4:
        config.llm.chat_model = OpenAIChatModel.GPT4
    config.filename = "examples/audience/iab-taxonomy.csv"
    segmentor = Segmentor(config)
    collections = segmentor.vecdb.list_collections()
    if segmentor.config.vecdb.collection_name in collections:
        print(
            f"Using audience taxonomy in collection: "
            f"[bold]{segmentor.config.vecdb.collection_name}[/bold]"
        )
    else:
        with console.status("[bold green]Loading IAB audience taxonomy..."):
            segmentor.ingest()

    segmentor_task = Task(
        segmentor,
        name="Segmentor",
        llm_delegate=False,
        single_round=True,
        system_message=""""
        You are a market segment expert. You will receive a description of a customer 
        profile, and your job is to map this to the IAB taxonomy. 
        """,
    )

    print("[blue]Welcome to the audience targeting bot!")
    biz_description = Prompt.ask(
        "Please describe your business, or hit enter to use default",
        default="An online store selling shoes for trendy teens",
    )

    researcher = ChatAgent(
        ChatAgentConfig(
            name="Researcher",
            use_llmagent_tools=config.use_llmagent_tools,
            use_functions_api=config.use_functions_api,
            vecdb=None,
            llm=config.llm,
        )
    )
    researcher_task = Task(
        researcher,
        llm_delegate=False,
        single_round=True,
        system_message=f"""
        You are a market-researcher, and you have deep knowledge of ideal customers 
        for your business, described below:
        
        BUSINESS: {biz_description}
        
        You will receive questions about customer profiles for your business, 
        and your job is to answer these CONCISELY in ONE SENTENCE, in simple 
        language.
        """,
    )
    marketer = ChatAgent(
        ChatAgentConfig(
            name="Marketer",
            use_llmagent_tools=config.use_llmagent_tools,
            use_functions_api=config.use_functions_api,
            vecdb=None,
            llm=config.llm,
        )
    )
    marketer_task = Task(
        marketer,
        llm_delegate=True,
        single_round=False,
        system_message="""
        You are a marketer, and you are trying to create a list of standard audience
        segments from the IAB taxonomy for your business. However you don't know much
        about the business. You will be speaking to two people: 
        - "Researcher" who knows a lot about the business, and
        - "Segmentor" who knows a lot about the IAB taxonomy.
        To clarify who you are talking to, you must always start your message with 
        "TO[<recipient>]:..." where <recipient> is either "Researcher" or "Segmentor".
        
        You can ask the Researcher about likely customers for the business, such as 
        what age, demographic, or lifestyle, etc, and the 
        Researcher will respond with a simple description of a customer profile. 
        You can ask the Segmentor to map a specific customer profile description 
        to the IAB taxonomy, ONE AT A TIME. Simply provide the customer profile, 
        do not say anything else, e.g. :
        
        TO[Segmentor]: trendy males under 40 who like to play basketball
        
        You may receive multiple STANDARD SEGMENTS in response one per line.
        Once you have accumulated 5 different STANDARD SEGMENTS,
        OR asked the Segmentor 5 times, whichever comes first, say DONE, and 
        put these STANDARD SEGMENTS together into a list.
        """,
    )

    validator_agent = ValidatorAgent(
        ValidatorAgentConfig(
            vecdb=None,
            llm=None,
            name="Validator",
            recipients=["Researcher", "Segmentor"],
        )
    )
    validator_task = Task(validator_agent, single_round=True)

    marketer_task.add_sub_task([validator_task, researcher_task, segmentor_task])

    marketer_task.run()


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
    config = SegmentorConfig(
        debug=debug,
        gpt4=True,
        cache=not nocache,
        use_functions_api=fn_api,
        use_llmagent_tools=not fn_api,
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
