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

## Knowledge graph-based RAG application with Langroid

RAG is one of the most popular LLM applications to answer questions.   
Recently, knowledge graphs via Retrieval-Augmented Generation (RAG) is getting more 
attention.
Langroid has a built-in a special agent Neo4jChatAgent. This agent incorporates a 
set of tools to facilitate the interactions with the graph database.


### Built-in Support for Neo4j

Langroid's Neo4jChatAgent uses Neo4j graph database. 
Neo4jChatAgent' tools provide support to read and write from/to Neo4j graph database 
to answer ussers' questions.
These tools provide also the functionality to automatically generate necessary Cypher 
Queries to answer the questions. Thus, allevieating the overhead of learning a new query 
language to interact with graph database. 


Setting up a basic knowledge-graph based RAG chatbot is as simple as (assume Neo4j settings provided in .env):

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

### PyPi Packages Dependency Chatbot

In the Langroid-examples repo there is an example showcase Tools/Function-calling + RAG in a single-agent setup [script](https://github.com/langroid/langroid/blob/main/examples/kg-chat/dependency_chatbot.py).

A detailed description about this example and how to use it is provided here [description](https://github.com/langroid/langroid/blob/main/examples/kg-chat/README.md). 

This example uses a `DependencyGraphAgent` 
(derived from [`Neo4jChatAgent`](https://github.com/langroid/langroid/blob/main/langroid/agent/special/neo4j/neo4j_chat_agent.py)).
It auto-generates a `neo4j` knowledge-graph based on the dependency
structure of a given `PyPi` package. You can then ask the chatbot questions
about the dependency graph. This agent uses two tools in addition to those 
already available to `Neo4jChatAgent`:

- DepGraphTool to build the dependency graph for a given pkg version, using the API
   at [DepsDev](https://deps.dev/)
- GoogleSearchTool to find package version and type information. It also can answer
other question from the web about other aspects after obtaining the intended information
from the dependency graph.

Here is what it looks like in action:

<figure markdown>
  ![dependency-demo](../../assets/demos/dependency_chatbot.gif){ width="800" }
  <figcaption>
Constructing dependency graph for `chainlit` using a single-agent system, with 
a Tool/Function-calling and RAG. The DependencyGraphAgent auto-generates a `neo4j` 
knowledge-graph based on the dependency structure of a given `PyPi` package
to answer using RAG.
</figcaption>
</figure>



