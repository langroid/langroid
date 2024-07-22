# Reterival over Knowledge Graphs

This folder contains two examples to demonistrate how to use `langroid` to build a chatbot that can answer questions about a knowledge graph.
The first example is a **PyPi Packages Dependency Chatbot** that can answer questions about a dependency graph of a `PyPi` package. 
The second example is a **CSV Chat** that can answer questions about a CSV knowledge graph.

## Requirements:

**1. NEO4j:**

This example relies on the `neo4j` Database. The easiest way to get access to neo4j is
by creating a cloud account at [Neo4j Aura](https://neo4j.com/cloud/platform/aura-graph-database/). OR you
can use Neo4j Docker image using this command:

```bash
docker run --rm \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

Upon creating the account successfully, neo4j will create a text file that contains
account settings, please provide the following information (uri, username,
password, and database), while creating the constructor `Neo4jChatAgentConfig`. 
These settings can be set inside the `.env` file as shown in [`.env-template`](../../.env-template)

**2. Google Custom Search API Credentials** 
needed to enable an Agent to use the `GoogleSearchTool`. 
Follow the [instruction](https://github.com/langroid/langroid?tab=readme-ov-file#gear-installation-and-setup) under `Optional Setup Instructions` to get these API credentials. 

**3. Visualization**
The package `pyvis` is required to enable the visualization tool `VisualizeGraph`. 
Run ``pip install pyvis`` to install this package.

## 1- PyPi Packages Dependency Chatbot

This example uses a `DependencyGraphAgent` 
(derived from [`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py)).
It auto-generates a `neo4j` knowledge-graph based on the dependency
structure of a given `PyPi` package. You can then ask the chatbot questions
about the dependency graph. This agent uses three tools in addition to those 
already available to `Neo4jChatAgent`:

- DepGraphTool to build the dependency graph for a given pkg version, using the API
   at [DepsDev](https://deps.dev/)
- GoogleSearchTool to find package version and type information. It also can answer
other question from the web about other aspects after obtaining the intended information
from the dependency graph. For examples:
  - Is this package/version vulnerable?
  - does the dpendency use latest version for this package verion?
  - Can I upgrade this package in the dependency graph?

The `Neo4jChatAgent` has access to these tools/function-calls:

- `GraphSchemaTool`: get schema of Neo4j knowledge-graph
- `CypherRetrievalTool`: generate cypher queries to get information from
   Neo4j knowledge-graph (Cypher is the query language for Neo4j)
- `VisualizeGraph`: supports only visualizing the the whole dependency graph

### Running the example

Run like this:
```
python3 examples/kg-chat/dependency_chatbot.py
```

`DependencyAgent` then will ask you to provide the name of the `PyPi` package.
It will then the tool `GoogleSearchTool` to get the version of
this package (you can skip this process by providing the intended version).
The `DependencyAgent` agent will ask to confirm the version number before
proceeding with constructing the dependency graph.

Finally, after constructing the dependency graph, you can ask `DependencyAgent`
questions about the dependency graph such as these (specific package names are
used here for illustration purposes, but of course you can use other names):

- what's the depth of the graph?
- what are the direct dependencies?
- any dependency on pytorch? which version?
- Is this package pytorch vunlnerable?
  (Note that in this case the `DependencyAgent` agent will consult the 
  tool `GoogleSearchTool` to get an answer from the internet.)
- tell me 3 interesting things about this package or dependency graph
- what's the path between package-1 and package-2? (provide names of package-1
  and -2)
- Tell me the names of all packages in the dependency graph that use pytorch.

**NOTE:** the dependency graph is constructed based
on [DepsDev API](https://deps.dev/). Therefore, the Chatbot will not be able to
construct the dependency graph if this API doesn't provide dependency metadata
infromation. 

## 2- CSV Chat

This example uses a `CSVGraphAgent` 
(derived from [`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py)).

The `CSVGraphAgent` allows users to ask questions about a CSV file by 
automatically converting it into a Neo4j knowledge graph using Cypher queries. 
This enables capturing complex relationships that cannot be easily
handled by libraries like `pandas`.

If the CSV knowledge graph has not been constructed beforehand, the `CSVGraphAgent`
provides the `pandas_to_kg` tool/function-call to create the necessary nodes and
relationships from the CSV file. Once the CSV knowledge graph is constructed,
the `CSVGraphAgent` can answer questions related to the CSV knowledge graph.
The `CSVGraphAgent` has access to this tool/function-call:

- `PandasToKGTool`: convert a `pandas` DataFrame into a CSV knowledge graph.

### Running the example

Run like this:
```
python3 examples/kg-chat/csv-chat.py
```

The `CSVGraphAgent` will have a dialog with the user to determine if they need to
construct the knowledge graph. If the user chooses to construct the knowledge graph, they
will be prompted to provide the location of the CSV file (URL or local file).

Under the hood, the agent will:

- Attempt to clean the CSV file after parsing it as a `DataFrame`.
- Determine node labels and relationships.
- Create the nodes and relationships in the Neo4j knowledge graph.

After constructing the CSV knowledge graph, you can ask the `CSVGraphAgent` any question
about the CSV knowledge graph. You can use [this IMDB CSV file](https://raw.githubusercontent.com/langroid/langroid-examples/main/examples/docqa/data/movies/IMDB.csv) 
or you can use your own CSV file.

**NOTES:**

- Unlike some other CSV -> Neo4j examples out there, here we are relying on the LLM
  to infer nodes and relationships from the CSV file, and generate the necessary
    Cypher queries to create the CSV knowledge graph. This is more flexible than
    a hard-coded approach.
- The agent will warn you if the CSV file is too large before proceeding with
  constructing the CSV knowledge graph. It will also give you the option to proceed with
  constructing the CSV knowledge graph based on a sample of the CSV file (i.e., a
  specified number of rows).
- The agent uses the function `_preprocess_dataframe_for_neo4j()` to clean the CSV file
  by removing rows that have empty values. However, you can provide your own function to
  clean the CSV file.
