import pytest

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

OR 
You can sign-up a free account at Neo4j Aura to create neo4j DB on the cloud.
"""

neo4j_settings = Neo4jSettings(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
)


@pytest.fixture
def agent():
    return Neo4jChatAgent(
        Neo4jChatAgentConfig(
            neo4j_settings=neo4j_settings,
        )
    )


def test_write_then_retrieval(agent):
    try:
        write_query = """
        CREATE (m:Movie {title: 'Inception', releaseYear: 2010})
        CREATE (a:Actor {name: 'Leonardo DiCaprio'})
        MERGE (a)-[:ACTED_IN]->(m)
        RETURN m, a
        """
        write_result = agent.write_query(write_query)
        assert write_result is True
        retrieval_query = """
        MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie)
        WHERE a.name = 'Leonardo DiCaprio' AND m.title = 'Inception'
        RETURN a.name, m.title, m.releaseYear, type(r) AS relationship
        """
        read_result = agent.read_query(retrieval_query)
        name_record = "'a.name': 'Leonardo DiCaprio'"
        title_record = "'m.title': 'Inception'"
        assert name_record in read_result
        assert title_record in read_result

    finally:
        # Cleanup - Remove the created records
        cleanup_query = """
        MATCH (a:Actor {name: 'Leonardo DiCaprio'})-[r:ACTED_IN]->
        (m:Movie {title: 'Inception'})
        DELETE r, a, m
        """
        agent.write_query(cleanup_query)
