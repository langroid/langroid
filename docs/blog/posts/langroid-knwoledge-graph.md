---
title: 'Langroid: Knolwedge Graph RAG powered by Neo4j'
draft: true
date: 2024-01-16
authors: 
  - mohannad
categories:
  - langroid
  - neo4j
  - rag
  - knowledge-graph
comments: true
---

## Knowledge Graph-Based RAG Application with Langroid

RAG (Retrieval-Augmented Generation) is one of the most popular LLM (Language Learning Model) 
applications for answering questions. Recently, the incorporation of knowledge graphs into 
RAG has been gaining more attention. Therefore, Langroid introduces a special agent called 
Neo4jChatAgent, which integrates a suite of tools to enhance interactions with graph databases.

### Built-In Support for Neo4j

Langroid's Neo4jChatAgent leverages the Neo4j graph database. The tools provided by Neo4jChatAgent 
support reading from and writing to the Neo4j graph database to answer users' questions. 
Additionally, these tools can automatically generate necessary Cypher Queries, thereby 
reducing the need to learn a new query language for interacting with graph databases.

Setting up a basic knowledge graph-based RAG chatbot is straightforward (assuming Neo4j 
settings are provided in .env):

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


### PyPi Package Dependency Chatbot

In the Langroid-examples repository, there is an example showcasing Tools/Function-calling + RAG in a single-agent setup [script](https://github.com/langroid/langroid/blob/main/examples/kg-chat/dependency_chatbot.py). A detailed description of this example and instructions for use 
are provided here [description](https://github.com/langroid/langroid/blob/main/examples/kg-chat/README.md).

This example employs a `DependencyGraphAgent`
(derived from [`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py)).
It automatically generates a `neo4j` knowledge graph based on the dependency
structure of a given `PyPi` package. Users can then ask the chatbot questions
about the dependency graph. In addition to the tools available to `Neo4jChatAgent`, this agent utilizes:

- `DepGraphTool` to construct the dependency graph for a specific package version, using the API at [DepsDev](https://deps.dev/).
- `GoogleSearchTool` to find package version and type information, as well as to answer other web-based questions after acquiring the required information from the dependency graph.

Here is what it looks like in action:

<figure markdown>
  ![dependency-demo](../../assets/demos/dependency_chatbot.gif)
  <figcaption>
Constructing the dependency graph for `chainlit` using a single-agent system, with 
Tool/Function-calling and RAG. The DependencyGraphAgent automatically generates a `neo4j` 
knowledge graph based on the dependency structure of a given `PyPi` package
to provide answers using RAG.
  </figcaption>
</figure>


