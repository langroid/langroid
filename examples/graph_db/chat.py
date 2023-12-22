"""
Single agent to use to chat with a Retrieval-augmented LLM.
User specifies package name -> make inquiry to the KG
                            -> LLM provides details about the packages.

This example relies on neo4j Database. The easiest way to get access to neo4j is by 
creating a cloud account at `https://neo4j.com/cloud/platform/aura-graph-database/`

Upon creating the account successfully, neo4j will create a text file contains 
account settings, please provide the following information (uri, username, password),
while creating the constructor `Neo4jConfig`.

Run like this:
python3 examples/graph_db/chat.py
"""


import typer
from rich import print
from rich.prompt import Prompt
from pydantic import BaseSettings
from dotenv import load_dotenv

from langroid.language_models.openai_gpt import OpenAIGPTConfig
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.graph_database.neo4j import Neo4j, Neo4jConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging

from langroid.agent.task import Task

app = typer.Typer()

setup_colored_logging()


class CLIOptions(BaseSettings):
    model: str = ""

    class Config:
        extra = "forbid"
        env_prefix = ""


def create_db(client, project, version) -> None:
    # with "pypi" as system, "langroid" as name, "0.1.147" as version
    crawl_project = f"""
    with "pypi" as system, "{project}" as name, "{version}" as version

    call apoc.load.json("https://api.deps.dev/v3alpha/systems/"+system+"/packages/"
                        +name+"/versions/"+version+":dependencies")
    yield value as r
    """
    build_project_graph = """
    call { with r
            unwind r.nodes as package
            merge (p:Package:PyPi {name:package.versionKey.name}) on create set 
            p.version = package.versionKey.version
            return collect(p) as packages
    }
    call { with r, packages
        unwind r.edges as edge
        with packages[edge.fromNode] as from, packages[edge.toNode] as to, edge
        merge (from)-[rel:DEPENDS_ON]->(to) ON CREATE SET rel.requirement 
        = edge.requirement
        return count(*) as numRels
    }

    match (root:Package:PyPi) where root.imported is null
    set root.imported = true
    with "pypi" as system, root.name as name, root.version as version
    call apoc.load.json("https://api.deps.dev/v3alpha/systems/"+system+"/packages/"
                        +name+"/versions/"+version+":dependencies")
    yield value as r
    call { with r
            unwind r.nodes as package
            merge (p:Package:PyPi {name:package.versionKey.name}) on create set 
            p.version = package.versionKey.version
            return collect(p) as packages
    }
    call { with r, packages
            unwind r.edges as edge
            with packages[edge.fromNode] as from, packages[edge.toNode] as to, edge
            merge (from)-[rel:DEPENDS_ON]->(to) ON CREATE SET 
            rel.requirement = edge.requirement
            return count(*) as numRels
    }
    return size(packages) as numPackages, numRels
    """

    check_db_exist = "MATCH (n) RETURN n LIMIT 1"
    if client.run_query(check_db_exist):
        return True
    else:
        construct_dependency_graph = crawl_project + build_project_graph
        if client.execute_write_query(construct_dependency_graph):
            print("[green]Database is created!")
            return True
        else:
            print("[red]Database is not created!")
            return False


def inquiry(client, package_name: str):
    query = f"""
            MATCH (p:Package:PyPi)
            WHERE toLower(p.name) CONTAINS '{package_name.lower()}'
            RETURN p.name, p.version
        """
    result = client.run_query(query)
    if result:
        return [(record["p.name"], record["p.version"]) for record in result]
    else:
        return []


def remove_database(client, db_name):
    delete_query = """
            MATCH (n)
            DETACH DELETE n
        """
    if client.execute_write_query(delete_query):
        print("[green]Database is deleted!")
    else:
        print("[red]Database is not deleted!")


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to Dependency Analysis chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    neo4j_cfg = Neo4jConfig(
        uri="",
        username="neo4j",
        password="",
        database="neo4j",
    )

    client = Neo4j(config=neo4j_cfg)

    project_name = Prompt.ask(
        """
    [blue] Tell me the python library or project name that you want to check 
    its dependencies...
    If you are not sure, check this website: https://deps.dev/, and search for 
    pypi projects.
    """
    )

    project_version = Prompt.ask(
        f"""
    [blue] Tell me the version of {project_name}...
    If this project {project_name} exists, check this website:
      https://deps.dev/pypi/{project_name} to find the version.
    """
    )

    create_db(client, project_name, project_version)

    pkg_name = Prompt.ask(
        f"""
    [blue] Tell me the package name that you want to check in the project
      {project_name}...
    """
    )
    result = inquiry(client, pkg_name)

    if len(result) == 0:
        print("[red] No result found!")
        return

    agent = ChatAgent(ChatAgentConfig(llm=OpenAIGPTConfig()))
    inline_result = ", ".join(
        [f"{name} (version {version})" for name, version in result]
    )
    task = Task(
        agent,
        llm_delegate=False,
        single_round=False,
        system_message="Tell me about the following packages: " + inline_result,
    )
    task.run()

    # check if the user wants to delete the database
    if Prompt.ask("[blue] Do you want to delete the database? (y/n)") == "y":
        remove_database(client, "neo4j")


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=not nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    opts = CLIOptions(model=model)
    chat(opts)


if __name__ == "__main__":
    app()
