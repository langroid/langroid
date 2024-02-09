import pytest
from dotenv import load_dotenv

from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)

"""
Before running this test case, make sure you run neo4j container using
the following command:
docker run --rm \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest

In this case, Neo4j env variables would look like this 
```
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_URI=neo4j://localhost:7687
NEO4J_DATABASE=neo4j
```

OR 
You can sign-up a free account at Neo4j Aura to create neo4j DB on the cloud.
"""

load_dotenv()
neo4j_settings = Neo4jSettings()


@pytest.fixture
def neo4j_agent(request):
    load_dotenv()
    neo4j_settings = Neo4jSettings()
    agent = Neo4jChatAgent(
        Neo4jChatAgentConfig(
            neo4j_settings=neo4j_settings,
        )
    )

    def teardown():
        # Remove the database
        agent.remove_database()

    request.addfinalizer(teardown)
    return agent


def test_write_then_retrieval(neo4j_agent):
    write_query = """
    CREATE (m:Movie {title: 'Inception', releaseYear: 2010})
    CREATE (a:Actor {name: 'Leonardo DiCaprio'})
    MERGE (a)-[:ACTED_IN]->(m)
    RETURN m, a
    """
    write_result = neo4j_agent.write_query(write_query)
    assert write_result.success is True

    retrieval_query = """
    MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie)
    WHERE a.name = 'Leonardo DiCaprio' AND m.title = 'Inception'
    RETURN a.name, m.title, m.releaseYear, type(r) AS relationship
    """
    read_result = neo4j_agent.read_query(retrieval_query)
    assert read_result.success is True
    assert read_result.data == [
        {
            "a.name": "Leonardo DiCaprio",
            "m.title": "Inception",
            "m.releaseYear": 2010,
            "relationship": "ACTED_IN",
        }
    ]
