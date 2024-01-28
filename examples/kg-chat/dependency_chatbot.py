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
account settings, please provide the following information (uri, username, password),
while creating the constructor `Neo4jConfig`.

Run like this:
```
python3 examples/kg-chat/dependency_chatbot.py
```
"""
import typer
from rich import print
from rich.prompt import Prompt
from dotenv import load_dotenv

import json
import ast
import networkx as nx
import plotly.graph_objects as go


from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from langroid.utils.constants import NO_ANSWER
from langroid.utils.configuration import set_global, Settings
from langroid.agent.tool_message import ToolMessage
from langroid.agent.tools.google_search_tool import GoogleSearchTool

from langroid.agent.task import Task
from cypher_message import CONSTRUCT_DEPENDENCY_GRAPH


app = typer.Typer()


class DepGraphTool(ToolMessage):
    request = "construct_dependency_graph"
    purpose = f"""Get package <package_version>, <package_type>, and <package_name>.
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
    request = "visualize_dependency_graph"
    purpose = """
      Use this tool/function to To display and visualize the dependency graph.
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
        if "No records found" not in response:
            self.config.database_created = True
            return "Database Exists"
        else:
            construct_dependency_graph = CONSTRUCT_DEPENDENCY_GRAPH.format(
                package_type=msg.package_type.lower(),
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
        MATCH (n)-[r]->(m) 
        RETURN n.name, type(r), m.name
        """
        data_str = self.read_query(query)
        # check if data_str contains { and } at the beginning and end
        if data_str[0] == "{" and data_str[-1] == "}":
            # Process the string to make it JSON compatible
            data_str = "[" + data_str + "]"
            data_str = data_str.replace("'", '"')

            # Convert the string to a list of dictionaries
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                # Fallback for any parsing issues
                data = ast.literal_eval(data_str)

            # Create a NetworkX graph
            G = nx.DiGraph()
            for row in data:
                # Access values using dictionary keys
                n_name = row["n.name"]
                r_type = row["type(r)"]
                m_name = row["m.name"]

                G.add_edge(n_name, m_name, relationship=r_type)

            # Position the nodes using one of the NetworkX layout algorithms
            pos = nx.spring_layout(G)

            # Create Edges
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.append(x0)
                edge_x.append(x1)
                edge_x.append(None)  # Prevents a line being drawn between edges
                edge_y.append(y0)
                edge_y.append(y1)
                edge_y.append(None)

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            # Create Nodes
            node_x = []
            node_y = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=True,
                    colorscale="YlGnBu",
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title="Node Connections",
                        xanchor="left",
                        titleside="right",
                    ),
                    line_width=2,
                ),
            )

            # Add node labels
            node_adjacencies = []
            node_text = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
                node_text.append(adjacencies[0])

            node_trace.marker.color = node_adjacencies
            node_trace.text = node_text

            # Create Network Graph
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title="<br>Network graph",
                    titlefont_size=16,
                    showlegend=False,
                    hovermode="closest",
                    margin=dict(b=20, l=5, r=5, t=40),
                    annotations=[
                        dict(
                            text="Python code to visualize an interactive graph using Plotly",
                            showarrow=False,
                            xref="paper",
                            yref="paper",
                            x=0.005,
                            y=-0.002,
                        )
                    ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                ),
            )

            fig.show()

            return "the graph is displayed"
        else:
            return data_str


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
            show_stats=False,
            use_tools=tools,
            use_functions_api=not tools,
            llm=OpenAIGPTConfig(
                chat_model=model or OpenAIChatModel.GPT4_TURBO,
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
      
    After receiving this infomration, make sure the package version is a number and the
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
    2. You can use the tool `make_query` to get relevant information from the
      graph database. I will execute this query and send you back the result.
      Make sure your queries comply with the database schema.
    3. Use the `web_search` tool/function to get information if needed.
    To display the dependency graph use this tool `visualize_dependency_graph`.
    """
    task = Task(
        dependency_agent,
        name="DependencyAgent",
        system_message=system_message,
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
