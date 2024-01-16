# PyPi Packages Dependency Chatbot

This example uses a `DependencyGraphAgent` 
(derived from [`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py)).
It auto-generates a `neo4j` knowledge-graph based on the dependency
structure of a given `PyPi` package. You can then ask the chatbot questions
about the dependency graph. This agent uses two tools in addition to those 
already available to `Neo4jChatAgent`:

- GoogleSearchTool to find package version and type information 
- DepGraphTool to build the dependency graph for a given pkg version, using the API
   at [DepsDev](https://deps.dev/)

The `Neo4jChatAgent` has access to these tools/function-calls:

- `GraphDatabaseSchema`: get schema of Neo4j knowledge-graph
- `GenerateCypherQueries`: generate cypher queries to get information from
   Neo4j knowledge-graph (Cypher is the query language for Neo4j)


# Requirements:

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

Upon creating the account successfully, neo4j will create a text file contains
account settings, please provide the following information (uri, username,
password, and database), while creating the constructor `Neo4jChatAgentConfig`. 
These settings can be set inside the `.env` file as shown in [`.env-template`](../../.env-template)

# Running the example

Run like this:
```
python3 examples/kg-chat/dependency_chatbot.py
```

`DependencyAgent` then will ask you to provide the name of the `PyPi` package.
It will then consult the 2nd agent `GoogleSearchAgent` to get the version of
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
  second agent `GoogleSearchAgent` to get an answer from the internet.)
- tell me 3 interesting things about this package
- what's the path between package-1 and package-2? (provide names of package-1
  and -2)
- Tell me the names of all packages in the dependency graph that use pytorch.

**NOTE:** the dependency graph is constructed based
on [DepsDev API](https://deps.dev/). Therefore, the Chatbot will not be able to
construct the dependency graph if this API doesn't provide dependency metadata
infromation. 
