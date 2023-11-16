import pytest

from langroid.graph_database.neo4j import Neo4j, Neo4jConfig

"""
Before running this test case, make sure you run neo4j container using
the following command:
docker run --rm \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
"""

neo4j_cfg = Neo4jConfig(
    uri="neo4j://localhost:7687",
    username="neo4j",
    password="password",
    database="neo4j",
)


@pytest.fixture
def neo4j_client():
    client = Neo4j(config=neo4j_cfg)
    yield client
    client.close()


def test_write_query(neo4j_client):
    write_query = """
    CREATE (m:Movie {title: 'Inception', releaseYear: 2010})
    CREATE (a:Actor {name: 'Leonardo DiCaprio'})
    MERGE (a)-[:ACTED_IN]->(m)
    RETURN m, a
    """
    result = neo4j_client.execute_write_query(write_query)
    assert result is True


def test_fetch_query(neo4j_client):
    fetch_query = """
    MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie)
    WHERE a.name = 'Leonardo DiCaprio' AND m.title = 'Inception'
    RETURN a.name, m.title, m.releaseYear, type(r) AS relationship
    """
    result = neo4j_client.run_query(fetch_query)
    assert result is not None
    assert len(result) > 0
    for record in result:
        assert record["a.name"] == "Leonardo DiCaprio"
        assert record["m.title"] == "Inception"
