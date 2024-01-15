"""
Multi agent to use to chat with a Retrieval-augmented LLM.
User specifies package name -> make inquiry to the KG
                            -> LLM provides details about the packages.

This example relies on neo4j Database. The easiest way to get access to neo4j is by 
creating a cloud account at `https://neo4j.com/cloud/platform/aura-graph-database/`

Upon creating the account successfully, neo4j will create a text file contains 
account settings, please provide the following information (uri, username, password),
while creating the constructor `Neo4jConfig`.

Run like this:
python3 examples/kg-chat/dependency_chatbot.py
"""
import typer
from rich import print
from rich.prompt import Prompt
from pydantic import BaseSettings
from dotenv import load_dotenv

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from langroid.utils.constants import NO_ANSWER
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.utils.configuration import set_global, Settings
from langroid.utils.logging import setup_colored_logging
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.google_search_tool import GoogleSearchTool

from langroid.agent.task import Task
from cypher_message import CONSTRUCT_DEPENDENCY_GRAPH


app = typer.Typer()

setup_colored_logging()


class GetPackageInfo(ToolMessage):
    request = "construct_dependency_graph"
    purpose = f"""Get package <package_version>, <package_type>, and <package_name>.
    For the <package_version>, obtain the recent version and should be a number. 
    For the <package_type>, return if the package is PyPI or not.
      Otherwise, return {NO_ANSWER}.
    For the <package_name>, return the package name provided by the user.
    ALL strings are in small letter. 
    """
    package_version: str
    package_type: str
    package_name: str


class DependencyGraphAgent(Neo4jChatAgent):
    package_name: str

    def construct_dependency_graph(self, msg: GetPackageInfo) -> None:
        check_db_exist = (
            "MATCH (n) WHERE n.name = $name AND n.version = $version RETURN n LIMIT 1"
        )
        response = self.read_query(
            check_db_exist, {"name": msg.package_name, "version": msg.package_version}
        )
        if "No records found" not in response:
            self.config.database_created = True
            return "Database Exists"
        else:
            construct_dependency_graph = CONSTRUCT_DEPENDENCY_GRAPH.format(
                package_type=msg.package_type,
                package_name=msg.package_name,
                package_version=msg.package_version,
            )
            if self.write_query(construct_dependency_graph):
                self.config.database_created = True
                return "Database is created!"
            else:
                return f"""
                    Database is not created!
                    Seems the package {msg.package_name} is not found,
                    """


class CLIOptions(BaseSettings):
    model: str = ""
    fn_api: bool = False

    class Config:
        extra = "forbid"
        env_prefix = ""


def chat(opts: CLIOptions) -> None:
    print(
        """
        [blue]Welcome to Dependency Analysis chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    neo4j_settings = Neo4jSettings()

    dependency_agent = DependencyGraphAgent(
        config=Neo4jChatAgentConfig(
            neo4j_settings=neo4j_settings,
            use_tools=not opts.fn_api,
            use_functions_api=opts.fn_api,
            llm=OpenAIGPTConfig(
                chat_model=OpenAIChatModel.GPT4_TURBO,
            ),
        ),
    )

    system_message = f"""You are expert in Dependency graphs and analyzing them using
    Neo4j. FIRST, I'll give you the name of the package that I want to analyze.
    THEN, You can ask me questions about the **package version number** and
    whether the **type** is PYPI. DON'T forget to include the package name
      in your questions. Consult `GoogleSearchAgent` to answer your queries.
    After receiving these infomration, make sure the package version is a number and the
    package type is PYPI.
    THEN ask the user if they want to construct the dependency graph,
    use the tool/function `construct_dependency_graph` to construct
      the dependency graph. Otherwise, say `Couldn't retrieve package type or version`
      and {NO_ANSWER}.
    After constructing the dependency graph successfully, you will have access to Neo4j 
    graph database, which contains dependency graph.
    You will try your best to answer my questions:
    1. You can use the tool `get_schema` to get node leabel and relationships in the
    dependency graph. 
    2. You can use the tool `make_query` to get relevant information from the
      graph database. I will execute this query and send you back the result.
      Make sure your queries comply with the database schema.
    3. Use `GoogleSearchAgent` to answer your queries that you couldn't answer.
    """
    task = Task(
        dependency_agent,
        name="DependencyAgent",
        system_message=system_message,
    )

    config = ChatAgentConfig(
        system_message="You are a helpful assistant",
        llm=OpenAIGPTConfig(
            chat_model=OpenAIChatModel.GPT4_TURBO,
        ),
        vecdb=None,
        use_tools=opts.fn_api,
        use_functions_api=not opts.fn_api,
    )

    google_agent = ChatAgent(config)
    google_agent.enable_message(GoogleSearchTool)
    google_task = Task(
        google_agent,
        name="GoogleSearchAgent",
        system_message="""
        You are a helpful assistant. You will try your best to answer my questions.
        You can use up to 5 results from the `web_search` tool/function-call to
          help you with answering the question.
        Be very concise in your responses, use no more than 1-2 sentences.
        When you answer based on a web search, First show me your answer, 
        and then show me the SOURCE(s) and EXTRACT(s) to justify your answer,
        in this format:
        
        <your answer here>
        SOURCE: https://www.wikihow.com/Be-a-Good-Assistant-Manager
        EXTRACT: Be a Good Assistant ... requires good leadership skills.
        
        SOURCE: ...
        EXTRACT: ...
        
        For the EXTRACT, ONLY show up to first 3 words, and last 3 words.
        SEND **DONE** after providing the answer or if you don't know the answer.
        """,
        interactive=False,
    )

    dependency_agent.enable_message(GetPackageInfo)

    task.add_sub_task(google_task)
    task.run()

    # check if the user wants to delete the database
    if dependency_agent.config.database_created:
        if Prompt.ask("[blue] Do you want to delete the database? (y/n)") == "y":
            dependency_agent.remove_database()


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    no_stream: bool = typer.Option(False, "--nostream", "-ns", help="no streaming"),
    tools: bool = typer.Option(
        False, "--tools", "-t", help="use langroid tools instead of function-calling"
    ),
    nocache: bool = typer.Option(False, "--nocache", "-nc", help="don't use cache"),
    cache_type: str = typer.Option(
        "redis", "--cachetype", "-ct", help="redis or momento"
    ),
) -> None:
    set_global(
        Settings(
            debug=debug,
            cache=nocache,
            stream=not no_stream,
            cache_type=cache_type,
        )
    )
    opts = CLIOptions(model=model, fn_api=not tools)
    chat(opts)


if __name__ == "__main__":
    app()
