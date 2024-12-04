import os
import subprocess
import time

import pytest
from neo4j import GraphDatabase

import langroid as lr
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jChatAgent,
    Neo4jChatAgentConfig,
    Neo4jSettings,
)
from langroid.agent.special.neo4j.tools import GraphSchemaTool


def wait_for_neo4j(max_attempts=30, delay=1):
    driver = None
    for attempt in range(max_attempts):
        try:
            driver = GraphDatabase.driver(
                "neo4j://localhost:7687", auth=("neo4j", "password")
            )
            with driver.session() as session:
                session.run("RETURN 1")
            print(f"Neo4j ready after {attempt + 1} attempts")
            return True
        except Exception:
            time.sleep(delay)
        finally:
            if driver:
                driver.close()
    raise TimeoutError("Neo4j failed to start")


COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "docker-compose-neo4j.yml")


def docker_setup_neo4j():
    # More aggressive cleanup
    try:
        # Stop and remove any existing neo4j container
        subprocess.run(
            ["docker", "stop", "neo4j-test"],
            check=False,  # Don't fail if container doesn't exist
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["docker", "rm", "-f", "neo4j-test"], check=False, stderr=subprocess.DEVNULL
        )

        # Clean up using docker-compose
        subprocess.run(
            [
                "docker-compose",
                "-f",
                COMPOSE_FILE,
                "down",
                "--volumes",
                "--remove-orphans",
            ],
            check=True,
        )
    except Exception as e:
        print(f"Cleanup error (non-fatal): {e}")

    # Start fresh container
    subprocess.run(
        ["docker-compose", "-f", COMPOSE_FILE, "up", "-d"],
        check=True,
    )


def docker_teardown_neo4j():
    # Cleanup after tests
    try:
        subprocess.run(
            [
                "docker-compose",
                "-f",
                COMPOSE_FILE,
                "down",
                "--volumes",
                "--remove-orphans",
            ],
            check=True,
        )
        subprocess.run(
            ["docker", "rm", "-f", "neo4j-test"], check=False, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"Cleanup error (non-fatal): {e}")


@pytest.fixture(scope="session", autouse=True)
def setup_neo4j():
    if not os.getenv("CI"):
        docker_setup_neo4j()
    wait_for_neo4j()
    yield
    if not os.getenv("CI"):
        docker_teardown_neo4j()


@pytest.fixture
def neo4j_agent(setup_neo4j):  # add setup_neo4j dependency
    agent = Neo4jChatAgent(
        Neo4jChatAgentConfig(
            neo4j_settings=Neo4jSettings(
                uri="neo4j://localhost:7687",
                username="neo4j",
                password="password",
                database="neo4j",
            )
        )
    )
    # No need to remove/recreate since we're using read-only demo DB
    yield agent


def test_write_then_retrieval(neo4j_agent):
    write_query = """
    CREATE (m:Movie {title: 'Inception', releaseYear: 2010})
    CREATE (a:Actor {name: 'Leonardo DiCaprio'})
    MERGE (a)-[:ACTED_IN]->(m)
    RETURN m, a
    """
    write_result = neo4j_agent.write_query(write_query)
    neo4j_agent.database_created = True
    assert write_result.success is True

    retrieval_query = """
    MATCH (a:Actor)-[r:ACTED_IN]->(m:Movie)
    WHERE a.name = 'Leonardo DiCaprio' AND m.title = 'Inception'
    RETURN a.name, m.title, m.releaseYear, type(r) AS relationship
    """
    read_result = neo4j_agent.read_query(retrieval_query)
    assert read_result.success is True
    assert {
        "a.name": "Leonardo DiCaprio",
        "m.title": "Inception",
        "m.releaseYear": 2010,
        "relationship": "ACTED_IN",
    } in read_result.data

    english_query = """
    What are the movies that Leonardo DiCaprio acted in?
    """
    task = lr.Task(
        neo4j_agent,
        name="Neo",
        interactive=False,
    )
    result = task.run(english_query)
    # english answer
    assert "inception" in result.content.lower()

    # run it as a task for 5 turns
    task = lr.Task(
        neo4j_agent,
        interactive=False,
    )
    result = task.run(english_query)
    assert "inception" in result.content.lower()


def test_delete_node(neo4j_agent):
    # Create and then delete
    create_query = """
    CREATE (p:Person {name: 'John Doe', age: 30})
    RETURN p
    """
    neo4j_agent.write_query(create_query)

    delete_query = """
    MATCH (p:Person {name: 'John Doe'})
    DELETE p
    """
    neo4j_agent.write_query(delete_query)

    # Verify deletion
    verify_query = """
    MATCH (p:Person {name: 'John Doe'})
    RETURN p
    """
    result = neo4j_agent.read_query(verify_query)
    assert len(result.data) == 0


def test_relationship_query(neo4j_agent):
    # Create network of friends
    setup_query = """
    CREATE (a:Person {name: 'Alice'}),
           (b:Person {name: 'Bob'}),
           (c:Person {name: 'Charlie'}),
           (a)-[:FRIENDS_WITH]->(b),
           (b)-[:FRIENDS_WITH]->(c)
    """
    neo4j_agent.write_query(setup_query)

    # Find friends of friends
    query = """
    MATCH (p1:Person {name: 'Alice'})-[:FRIENDS_WITH]->
          ()-[:FRIENDS_WITH]->(fof:Person)
    RETURN fof.name
    """
    result = neo4j_agent.read_query(query)
    assert result.data[0]["fof.name"] == "Charlie"


def test_property_update(neo4j_agent):
    # Create node
    create_query = """
    CREATE (m:Movie {title: 'The Matrix', year: 1999})
    """
    neo4j_agent.write_query(create_query)

    # Update property
    update_query = """
    MATCH (m:Movie {title: 'The Matrix'})
    SET m.rating = 9.5
    RETURN m
    """
    result = neo4j_agent.write_query(update_query)

    # Verify update
    verify_query = """
    MATCH (m:Movie {title: 'The Matrix'})
    RETURN m.rating
    """
    result = neo4j_agent.read_query(verify_query)
    assert result.data[0]["m.rating"] == 9.5


def test_multiple_relationships(neo4j_agent):
    # Create complex relationship network
    setup_query = """
    CREATE (john:Person {name: 'John'}),
           (company:Company {name: 'Tech Corp'}),
           (project:Project {name: 'AI Initiative'}),
           (john)-[:WORKS_AT]->(company),
           (john)-[:MANAGES]->(project),
           (company)-[:OWNS]->(project)
    """
    neo4j_agent.write_query(setup_query)

    # Query to find all relationships
    query = """
    MATCH (p:Person {name: 'John'})-[r]->(x)
    RETURN type(r) as relationship_type, x.name as connected_to
    """
    result = neo4j_agent.read_query(query)

    # Verify both relationships exist
    relationships = [r["relationship_type"] for r in result.data]
    assert "WORKS_AT" in relationships
    assert "MANAGES" in relationships


def test_database_schema(neo4j_agent):
    # First create some data
    setup_query = """
    CREATE (p:Person {name: 'Alice', age: 30}),
           (m:Movie {title: 'Matrix', year: 1999}),
           (g:Genre {name: 'Sci-Fi'}),
           (p)-[:WATCHED]->(m),
           (m)-[:HAS_GENRE]->(g)
    """
    neo4j_agent.write_query(setup_query)

    # Get node labels
    labels_query = """
    CALL db.labels()
    """
    labels_result = neo4j_agent.read_query(labels_query)

    # Get relationship types
    rels_query = """
    CALL db.relationshipTypes()
    """
    rels_result = neo4j_agent.read_query(rels_query)

    # Verify schema
    labels = {item["label"] for item in labels_result.data}
    relationships = {item["relationshipType"] for item in rels_result.data}

    assert {"Person", "Movie", "Genre"}.issubset(labels)
    assert {"WATCHED", "HAS_GENRE"}.issubset(relationships)


def test_graph_schema_visualization(neo4j_agent):
    setup_query = """
    CREATE (p:Person {name: 'Alice', age: 30}),
           (m:Movie {title: 'Matrix', year: 1999}),
           (g:Genre {name: 'Sci-Fi'}),
           (p)-[:WATCHED]->(m),
           (m)-[:HAS_GENRE]->(g)
    """
    neo4j_agent.write_query(setup_query)

    schema_data = neo4j_agent.graph_schema_tool(GraphSchemaTool())

    # Check node labels
    node_labels = {node["name"] for node in schema_data[0]["nodes"]}
    assert {"Person", "Movie", "Genre"}.issubset(node_labels)

    # Check relationships
    relationships = {rel[1] for rel in schema_data[0]["relationships"]}
    assert {"WATCHED", "HAS_GENRE"}.issubset(relationships)
