---
title: 'Langroid: Knolwedge Graph RAG powered by Neo4j'
draft: false
date: 2024-01-18
authors: 
  - mohannad
categories:
  - langroid
  - neo4j
  - rag
  - knowledge-graph
comments: true
---

## "Chat" with various sources of information
LLMs are increasingly being used to let users converse in natural language with 
a variety of types of data sources:
<!-- more -->
- unstructured text documents: a user's query is augmented with "relevant" documents or chunks
  (retrieved from an embedding-vector store) and fed to the LLM to generate a response -- 
  this is the idea behind Retrieval Augmented Generation (RAG).
- SQL Databases: An LLM translates a user's natural language question into an SQL query,
  which is then executed by another module, sending results to the LLM, so it can generate
  a natural language response based on the results.
- Tabular datasets: similar to the SQL case, except instead of an SQL Query, the LLM generates 
  a Pandas dataframe expression.

Langroid has had specialized Agents for the above scenarios: `DocChatAgent` for RAG with unstructured
text documents, `SQLChatAgent` for SQL databases, and `TableChatAgent` for tabular datasets.

## Adding support for Neo4j Knowledge Graphs

Analogous to the SQLChatAgent, Langroid now has a 
[`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py) 
to interact with a Neo4j knowledge graph using natural language.
This Agent has access to two key tools that enable it to handle a user's queries:

- `GraphSchemaTool` to get the schema of a Neo4j knowledge graph.
- `CypherRetrievalTool` to generate Cypher queries from a user's query.
Cypher is a specialized query language for Neo4j, and even though it is not as widely known as SQL,
most LLMs today can generate Cypher Queries.

Setting up a basic Neo4j-based RAG chatbot is straightforward. First ensure 
you set these environment variables (or provide them in a `.env` file):
```bash
NEO4J_URI=<uri>
NEO4J_USERNAME=<username>
NEO4J_PASSWORD=<password>
NEO4J_DATABASE=<database>
```

Then you can configure and define a `Neo4jChatAgent` like this:
```python
import langroid as lr
import langroid.language_models as lm

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)

llm_config = lm.OpenAIGPTConfig()

load_dotenv()

neo4j_settings = Neo4jSettings()

kg_rag_agent_config = Neo4jChatAgentConfig(
    neo4j_settings=neo4j_settings,
    llm=llm_config, 
)
kg_rag_agent = Neo4jChatAgent(kg_rag_agent_config)
kg_rag_task = lr.Task(kg_rag_agent, name="kg_RAG")
kg_rag_task.run()
```


## Example: PyPi Package Dependency Chatbot

In the Langroid-examples repository, there is an example python 
[script](https://github.com/langroid/langroid-examples/blob/main/examples/kg-chat/)
showcasing tools/Function-calling + RAG using a `DependencyGraphAgent` derived from [`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py).
This agent uses two tools, in addition to the tools available to `Neo4jChatAgent`:

- `GoogleSearchTool` to find package version and type information, as well as to answer 
 other web-based questions after acquiring the required information from the dependency graph.
- `DepGraphTool` to construct a Neo4j knowledge-graph modeling the dependency structure
   for a specific package, using the API at [DepsDev](https://deps.dev/).

In response to a user's query about dependencies, the Agent decides whether to use a Cypher query
or do a web search. Here is what it looks like in action:

<figure markdown>
  ![dependency-demo](../../assets/demos/dependency_chatbot.gif)
  <figcaption>
Chatting with the `DependencyGraphAgent` (derived from Langroid's `Neo4jChatAgent`).
When a user specifies a Python package name (in this case "chainlit"), the agent searches the web using
`GoogleSearchTool` to find the version of the package, and then uses the `DepGraphTool`
to construct the dependency graph as a neo4j knowledge graph. The agent then answers
questions by generating Cypher queries to the knowledge graph, or by searching the web.
  </figcaption>
</figure>


