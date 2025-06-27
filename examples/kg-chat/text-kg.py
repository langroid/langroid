"""
Example showing how to chat with a graph database generated from
unstructured data.

This example will automatically:
- create triplets that represent various entities and relationships from the text
- generate the cypher query to populate the triplets in the graph database
- generate all the required Cypher queries for Neo4j to answer user's questions.

This example relies on neo4j. The easiest way to get access to neo4j is by
creating a cloud account at `https://neo4j.com/cloud/platform/aura-graph-database/`

Upon creating the account successfully, neo4j will create a text file that contains
account settings, please provide the following information (uri, username, password) as
described here
`https://github.com/langroid/langroid/tree/main/examples/kg-chat#requirements`

Run like this

python3 examples/kg-chat/text-kg.py

Optional args:
* -d or --debug to enable debug mode
* -nc or --nocache to disable caching
* -m or --model to specify a model name

"""

import typer
from dotenv import load_dotenv
from rich import print

import langroid as lr
import langroid.language_models as lm
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.utils.configuration import Settings, set_global

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

    # Look inside Neo4jSettings and explicitly
    # set each param (including database) based on your Neo4j instance
    neo4j_settings = Neo4jSettings(database="neo4j")

    config = Neo4jChatAgentConfig(
        name="TextNeo",
        system_message="""
        You are an information representation expert, and you are especially 
        knowledgeable about representing information in a Knowledge Graph such as Neo4j.        
        When the user gives you a TEXT and the CURRENT SCHEMA (possibly empty), 
        your task is to generate a Cypher query that will add the entities/relationships
        from the TEXT to the Neo4j database, taking the CURRENT SCHEMA into account.
        In particular, SEE IF YOU CAN REUSE EXISTING ENTITIES/RELATIONSHIPS,
        and create NEW ONES ONLY IF NECESSARY.
        
        To present the Cypher query, you can use the `retrieval_query` tool/function        
        """,
        neo4j_settings=neo4j_settings,
        show_stats=False,
        llm=lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4o,
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

    CURRENT_SCHEMA = ""

    task = lr.Task(
        agent,
        interactive=True,
        single_round=False,
    )
    task.run(
        f"""
    TEXT: {TEXT}
    
    CURRENT SCHEMA: {CURRENT_SCHEMA}
    """
    )

    curr_schema = agent.get_schema(None)
    print(f"SCHEMA: {curr_schema}")

    # now feed in the schema to the next run, with new text

    TEXT = """
    Apple was founded as Apple Computer Company on April 1, 1976, to produce and market 
    Steve Wozniak's Apple I personal computer. The company was incorporated by Wozniak 
    and Steve Jobs in 1977. Its second computer, the Apple II, became a best seller as 
    one of the first mass-produced microcomputers. Apple introduced the Lisa in 1983 and 
    the Macintosh in 1984, as some of the first computers to use a graphical user 
    interface and a mouse.
    """

    task.run(
        f"""
        TEXT: {TEXT}

        CURRENT SCHEMA: {curr_schema}
        """
    )
    updated_schema = agent.get_schema(None)
    print(f"UPDATED SCHEMA: {updated_schema}")

    # We can now ask a question that can be answered based on the schema

    config = Neo4jChatAgentConfig(
        name="TextNeoQA",
        system_message="""
        You will get a question about some information that is represented within
        a Neo4j graph database. You will use the `retrieval_query` tool/function to
        generate a Cypher query that will answer the question. Do not explain
        your query, just present it using the `retrieval_query` tool/function.
        """,
        neo4j_settings=neo4j_settings,
        show_stats=False,
        llm=lm.OpenAIGPTConfig(
            chat_model=model or lm.OpenAIChatModel.GPT4o,
        ),
    )

    agent = Neo4jChatAgent(config=config)

    task = lr.Task(agent)

    print("[blue] Now you can ask questions ")

    task.run()


if __name__ == "__main__":
    app()
