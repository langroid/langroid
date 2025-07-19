"""
Single-agent to use to chat with a Neo4j knowledge-graph (KG)
that models a dependency graph of Python packages.

User specifies package name
-> agent gets version number and type of package using google search
-> agent builds dependency graph using Neo4j
-> user asks natural language query about dependencies
-> LLM translates to Cypher query to get info from KG
-> Query results returned to LLM
-> LLM translates to natural language response

This example relies on neo4j. The easiest way to get access to neo4j is by
creating a cloud account at `https://neo4j.com/cloud/platform/aura-graph-database/`

Upon creating the account successfully, neo4j will create a text file that contains
account settings, please provide the following information (uri, username, password) as
described here
`https://github.com/langroid/langroid/tree/main/examples/kg-chat#requirements`

The rest of requirements are described in
 `https://github.com/langroid/langroid/blob/main/examples/kg-chat/README.md`

Run like this:
```
python3 examples/kg-chat/dependency_chatbot.py
```
"""

import webbrowser
from pathlib import Path

import typer
from cypher_message import CONSTRUCT_DEPENDENCY_GRAPH
from dotenv import load_dotenv
from pyvis.network import Network
from rich import print
from rich.prompt import Prompt

from langroid import TaskConfig
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.agent.task import Task
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER, SEND_TO

app = typer.Typer()


class DepGraphTool(ToolMessage):
    request: str = "construct_dependency_graph"
    purpose: str = f"""Get package <package_version>, <package_type>, and <package_name>.
    For the <package_version>, obtain the recent version, it should be a number. 
    For the <package_type>, return if the package is PyPI or not.
      Otherwise, return {NO_ANSWER}.
    For the <package_name>, return the package name provided by the user.
    ALL strings are in lower case.
    """
    package_version: str
    package_type: str
    package_name: str


class VisualizeGraph(ToolMessage):
    request: str = "visualize_dependency_graph"
    purpose: str = """
      Use this tool/function to display the dependency graph.
      """
    package_version: str
    package_type: str
    package_name: str
    query: str


class DependencyGraphAgent(Neo4jChatAgent):
    def construct_dependency_graph(self, msg: DepGraphTool) -> None:
        check_db_exist = (
            "MATCH (n) WHERE n.name = $name AND n.version = $version RETURN n LIMIT 1"
        )
        response = self.read_query(
            check_db_exist, {"name": msg.package_name, "version": msg.package_version}
        )
        if response.success and response.data:
            # self.config.database_created = True
            return "Database Exists"
        else:
            construct_dependency_graph = CONSTRUCT_DEPENDENCY_GRAPH.format(
                package_type=msg.package_type.lower(),
                package_name=msg.package_name,
                package_version=msg.package_version,
            )
            response = self.write_query(construct_dependency_graph)
            if response.success:
                self.config.database_created = True
                return "Database is created!"
            else:
                return f"""
                    Database is not created!
                    Seems the package {msg.package_name} is not found,
                    """

    def visualize_dependency_graph(self, msg: VisualizeGraph) -> str:
        """
        Visualizes the dependency graph based on the provided message.

        Args:
            msg (VisualizeGraph): The message containing the package info.

        Returns:
            str: response indicates whether the graph is displayed.
        """
        # Query to fetch nodes and relationships
        # TODO: make this function more general to return customized graphs
        # i.e, displays paths or subgraphs
        query = """
            MATCH (n)
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
        """

        query_result = self.read_query(query)
        nt = Network(notebook=False, height="750px", width="100%", directed=True)

        node_set = set()  # To keep track of added nodes

        for record in query_result.data:
            # Process node 'n'
            if "n" in record and record["n"] is not None:
                node = record["n"]
                # node_id = node.get("id", None)  # Assuming each node has a unique 'id'
                node_label = node.get("name", "Unknown Node")
                node_title = f"Version: {node.get('version', 'N/A')}"
                node_color = "blue" if node.get("imported", False) else "green"

                # Check if node has been added before
                if node_label not in node_set:
                    nt.add_node(
                        node_label, label=node_label, title=node_title, color=node_color
                    )
                    node_set.add(node_label)

            # Process relationships and node 'm'
            if (
                "r" in record
                and record["r"] is not None
                and "m" in record
                and record["m"] is not None
            ):
                source = record["n"]
                target = record["m"]
                relationship = record["r"]

                source_label = source.get("name", "Unknown Node")
                target_label = target.get("name", "Unknown Node")
                relationship_label = (
                    relationship[1]
                    if isinstance(relationship, tuple) and len(relationship) > 1
                    else "Unknown Relationship"
                )

                # Ensure both source and target nodes are added before adding the edge
                if source_label not in node_set:
                    source_title = f"Version: {source.get('version', 'N/A')}"
                    source_color = "blue" if source.get("imported", False) else "green"
                    nt.add_node(
                        source_label,
                        label=source_label,
                        title=source_title,
                        color=source_color,
                    )
                    node_set.add(source_label)
                if target_label not in node_set:
                    target_title = f"Version: {target.get('version', 'N/A')}"
                    target_color = "blue" if target.get("imported", False) else "green"
                    nt.add_node(
                        target_label,
                        label=target_label,
                        title=target_title,
                        color=target_color,
                    )
                    node_set.add(target_label)

                nt.add_edge(source_label, target_label, title=relationship_label)

        nt.options.edges.font = {"size": 12, "align": "top"}
        nt.options.physics.enabled = True
        nt.show_buttons(filter_=["physics"])

        output_file_path = "neo4j_graph.html"
        nt.write_html(output_file_path)

        # Try to open the HTML file in a browser
        try:
            abs_file_path = str(Path(output_file_path).resolve())
            webbrowser.open("file://" + abs_file_path, new=2)
        except Exception as e:
            print(f"Failed to automatically open the graph in a browser: {e}")


@app.command()
def main(
    debug: bool = typer.Option(False, "--debug", "-d", help="debug mode"),
    model: str = typer.Option("", "--model", "-m", help="model name"),
    tools: bool = typer.Option(
        False, "--tools", "-t", help="use langroid tools instead of function-calling"
    ),
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
        [blue]Welcome to Dependency Analysis chatbot!
        Enter x or q to quit at any point.
        """
    )

    load_dotenv()

    neo4j_settings = Neo4jSettings()

    dependency_agent = DependencyGraphAgent(
        config=Neo4jChatAgentConfig(
            chat_mode=True,
            neo4j_settings=neo4j_settings,
            show_stats=False,
            use_tools=tools,
            use_functions_api=not tools,
            llm=OpenAIGPTConfig(
                chat_model=model or OpenAIChatModel.GPT4o,
            ),
        ),
    )

    system_message = f"""You are an expert in Dependency graphs and analyzing them using
    Neo4j. 
    
    FIRST, I'll give you the name of the package that I want to analyze.
    
    THEN, you can also use the `web_search` tool/function to find out information about a package,
      such as version number and package type (PyPi or not). 
    
    If unable to get this info, you can ask me and I can tell you.
    
    DON'T forget to include the package name in your questions. 
      
    After receiving this information, make sure the package version is a number and the
    package type is PyPi.
    THEN ask the user if they want to construct the dependency graph,
    and if so, use the tool/function `construct_dependency_graph` to construct
      the dependency graph. Otherwise, say `Couldn't retrieve package type or version`
      and {NO_ANSWER}.
    After constructing the dependency graph successfully, you will have access to Neo4j 
    graph database, which contains dependency graph.
    You will try your best to answer my questions. Note that:
    1. You can use the tool `get_schema` to get node label and relationships in the
    dependency graph. 
    2. You can use the tool `retrieval_query` to get relevant information from the
      graph database. I will execute this query and send you back the result.
      Make sure your queries comply with the database schema.
    3. Use the `web_search` tool/function to get information if needed.
    To display the dependency graph use this tool `visualize_dependency_graph`.
    """
    task_config = TaskConfig(addressing_prefix=SEND_TO)
    task = Task(
        dependency_agent,
        name="DependencyAgent",
        system_message=system_message,
        # non-interactive but await user ONLY if addressed or LLM sends a non-tool msg,
        # (see the handle_message_fallback method in the agent)
        interactive=False,
        config=task_config,
    )

    dependency_agent.enable_message(DepGraphTool)
    dependency_agent.enable_message(GoogleSearchTool)
    dependency_agent.enable_message(VisualizeGraph)

    task.run()

    # check if the user wants to delete the database
    if dependency_agent.config.database_created:
        if Prompt.ask("[blue] Do you want to delete the database? (y/n)") == "y":
            dependency_agent.remove_database()


if __name__ == "__main__":
    app()
