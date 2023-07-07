from llmagent.utils.logging import setup_colored_logging
from llmagent.agent.task import Task
from examples.audience.agents.segmentor import Segmentor, SegmentorConfig
from llmagent.language_models.openai_gpt import OpenAIChatModel
from llmagent.utils.configuration import update_global_settings, set_global, Settings
import re
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
        segmentor.vecdb.set_collection(segmentor.config.vecdb.collection_name)
        with console.status("[bold green]Loading IAB audience taxonomy..."):
            segmentor.ingest()

    print("[blue]Welcome to the audience targeting bot!")

    system_msg = Prompt.ask(
        """
        [blue] Tell me who I am; complete this sentence: You are...
        [or hit enter for default] 
        [blue] Human
        """,
        default="a marketing expert.",
    )
    system_msg = re.sub("you are", "", system_msg, flags=re.IGNORECASE)
    task = Task(
        segmentor,
        name="Segmentor",
        llm_delegate=False,
        single_round=False,
        system_message="You are " + system_msg,
    )
    task.run()


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
