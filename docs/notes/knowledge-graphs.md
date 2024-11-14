# Knowledge-graph support

Langroid can be used to set up natural-language conversations with knowledge graphs.
Currently the two most popular knowledge graphs are supported:

## Neo4j

- [implementation](https://github.com/langroid/langroid/tree/main/langroid/agent/special/neo4j)
- test: [test_neo4j_chat_agent.py](https://github.com/langroid/langroid/blob/main/tests/main/test_neo4j_chat_agent.py)
- examples: [chat-neo4j.py](https://github.com/langroid/langroid/blob/main/examples/kg-chat/chat-neo4j.py) 

## ArangoDB

Available with Langroid v0.20.1 and later.

Uses the [python-arangodb](https://github.com/arangodb/python-arango) library.

- [implementation](https://github.com/langroid/langroid/tree/main/langroid/agent/special/arangodb)
- tests: [test_arangodb.py](https://github.com/langroid/langroid/blob/main/tests/main/test_arangodb.py), [test_arangodb_chat_agent.py](https://github.com/langroid/langroid/blob/main/tests/main/test_arangodb_chat_agent.py)
- example: [chat-arangodb.py](https://github.com/langroid/langroid/blob/main/examples/kg-chat/chat-arangodb.py)


