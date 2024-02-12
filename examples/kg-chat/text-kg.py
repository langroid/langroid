"""
Simple text to knowledge graph example, using the Neo4jChatAgent
"""

import typer
from rich import print
from dotenv import load_dotenv

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
import langroid as lr
import langroid.language_models as lm
from langroid.utils.configuration import set_global, Settings

app = typer.Typer()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=nocache,
        )
    )
    print(
        """
        [blue]Welcome to the Text-to-KG chatbot!
        Enter x or q to quit at any point.[/blue]
        """
    )

    load_dotenv()

    # Look inside Neo4jSettings and explicit set each param based on your Neo4j instance
    neo4j_settings = Neo4jSettings(database="neo4j")

    system_message = """
        You are an information representation expert, and you are especially 
        knowledgeable about representing information in a Knowledge Graph such as Neo4j.
        
        When the user gives you a TEXT, you should INFER the triplets from the TEXT.
        Each triplet is a tuple of the form `(subject, relationship, object)`.
        SHOW me the triplets you inferred as a list and ask the user to confirm them,
        then you should generate a Cypher query that will create the 
        entities/relationships based on the triplets. USE the `create_query` 
        tool/function to create the entities/relationships, 

        Here is an example:
        ```
        TEXT: "Albert Einstein, born in Ulm, won the Nobel Prize in Physics in 1921."
        Triplets:
        (Albert Einstein, born in, Ulm)
        (Albert Einstein, won, Nobel Prize in Physics)
        (Nobel Prize in Physics, awarded in, 1921)
        ```
        """

    config = Neo4jChatAgentConfig(
        name="TextNeo",
        system_message=system_message,
        neo4j_settings=neo4j_settings,
        show_stats=False,
        llm=lm.AzureConfig(
            chat_model=lm.OpenAIChatModel.GPT4_TURBO,
        ),
    )

    agent = Neo4jChatAgent(config=config)

    TEXT = """
    Apple Inc. (formerly Apple Computer, Inc.) is an American multinational technology 
    company headquartered in Cupertino, California, in Silicon Valley. 
    It designs, develops, and sells consumer electronics, computer software, 
    and online services. Devices include the iPhone, iPad, Mac, Apple Watch, and 
    Apple TV; operating systems include iOS and macOS; and software applications and 
    services include iTunes, iCloud, and Apple Music.

    As of March 2023, Apple is the world's largest company by market capitalization.[6] 
    In 2022, it was the largest technology company by revenue, with US$394.3 billion.[7] 
    As of June 2022, Apple was the fourth-largest personal computer vendor by unit sales, 
    the largest manufacturing company by revenue, and the second-largest 
    manufacturer of mobile phones in the world. It is one of the Big Five American 
    information technology companies, alongside Alphabet (the parent company of Google), 
    Amazon, Meta (the parent company of Facebook), and Microsoft.    
    """

    task = lr.Task(
        agent,
        interactive=True,
        single_round=False,
    )
    task.run(
        f"""
    TEXT: {TEXT}
    """
    )

    # schema = agent.get_schema(None)
    # print(f"SCHEMA: {schema}")

    # for subsequent text -> conversions, you can use the `schema` obtained above,
    # and insert it into as the value of CURRENT_SCHEMA in the next run.


if __name__ == "__main__":
    app()
