import os
import subprocess
import time

import pytest
from arango import ArangoClient

COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "docker-compose-arango.yml")


def docker_setup_arango():
    # Start container using docker-compose
    subprocess.run(
        [
            "docker-compose",
            "-f",
            COMPOSE_FILE,
            "up",
            "-d",
        ],
        check=True,
    )


def docker_teardown_arango():
    # Cleanup
    subprocess.run(
        [
            "docker-compose",
            "-f",
            COMPOSE_FILE,
            "down",
        ],
        check=True,
    )


@pytest.fixture(scope="session", autouse=True)
def setup_arango():
    if not os.getenv("CI"):
        docker_setup_arango()
    time.sleep(10)
    yield
    if not os.getenv("CI"):
        docker_teardown_arango()


@pytest.fixture
def arango_client():
    client = ArangoClient(hosts="http://localhost:8529")
    return client


@pytest.fixture
def test_database(arango_client):
    sys_db = arango_client.db("_system", username="root", password="rootpassword")
    # Create test database
    db_name = "test_db"
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
    return arango_client.db(db_name, username="root", password="rootpassword")


@pytest.fixture
def test_collection(test_database):
    # Create test collection: a collection is like a table in a relational database
    coll_name = "test_collection"
    if not test_database.has_collection(coll_name):
        collection = test_database.create_collection(name=coll_name)
    else:
        collection = test_database.collection(coll_name)

    # Clear collection before use
    collection.truncate()

    return collection


def test_create_document(test_collection):
    # Create document: this is like inserting a row in a relational database
    doc = {"name": "test", "value": 123}
    result = test_collection.insert(doc)
    doc_key = result["_key"]

    # Verify document exists
    retrieved_doc = test_collection.get(doc_key)
    assert retrieved_doc["name"] == "test"
    assert retrieved_doc["value"] == 123

    # create document with explicit key
    doc = {"_key": "test_key", "name": "test", "value": 123}
    test_collection.insert(doc)

    # Verify document exists
    retrieved_doc = test_collection.get("test_key")
    # verify that the key is the same
    assert retrieved_doc["_key"] == "test_key"
    assert retrieved_doc["name"] == "test"
    assert retrieved_doc["value"] == 123

    # retrieve document using get, equivalent to above
    retrieved_doc = test_collection.get("test_key")
    assert retrieved_doc["_key"] == "test_key"
    assert retrieved_doc["name"] == "test"
    assert retrieved_doc["value"] == 123


def test_query_documents(test_collection, test_database):
    # Create multiple documents
    for i in range(5):
        doc = {"name": f"test_{i}", "value": i}
        test_collection.insert(doc)

    # Query documents
    aql = "FOR doc IN @@collection FILTER doc.value >= 2 RETURN doc"
    bind_vars = {"@collection": test_collection.name}
    cursor = test_database.aql.execute(aql, bind_vars=bind_vars)
    result = [doc for doc in cursor]

    assert len(result) == 3


def test_knowledge_graph(test_database):
    # Create collections for nodes and edges
    if not test_database.has_collection("nodes"):
        nodes = test_database.create_collection(name="nodes")
    else:
        nodes = test_database.collection("nodes")

    if not test_database.has_collection("relationships"):
        relationships = test_database.create_collection(name="relationships", edge=True)
    else:
        relationships = test_database.collection("relationships")

    nodes.truncate()
    relationships.truncate()

    # Create person nodes
    person1 = nodes.insert({"type": "person", "name": "John"})

    person2 = nodes.insert({"type": "person", "name": "Mary"})

    # Create location node
    location = nodes.insert({"type": "location", "name": "New York"})

    # Create relationships
    relationships.insert(
        {
            "_from": f"nodes/{person1['_key']}",
            "_to": f"nodes/{location['_key']}",
            "type": "LIVES_IN",
        }
    )

    relationships.insert(
        {
            "_from": f"nodes/{person1['_key']}",
            "_to": f"nodes/{person2['_key']}",
            "type": "KNOWS",
        }
    )

    # Query relationships
    aql = """
    FOR p IN nodes
        FILTER p.type == 'person'
        LET lives = (
            FOR v, e IN 1..1 OUTBOUND p relationships
            FILTER e.type == 'LIVES_IN'
            RETURN v.name
        )
        RETURN {person: p.name, livesIn: lives[0]}
    """
    cursor = test_database.aql.execute(aql)
    result = [doc for doc in cursor]

    assert len(result) == 2
    assert result[0]["person"] == "John"
    assert result[0]["livesIn"] == "New York"
    assert result[1]["person"] == "Mary"
    assert result[1]["livesIn"] is None


def test_graph_creation(test_database):
    # Create collections for graph
    if not test_database.has_collection("person_vertices"):
        person_vertices = test_database.create_collection("person_vertices")
    else:
        person_vertices = test_database.collection("person_vertices")

    if not test_database.has_collection("friendship_edges"):
        friendship_edges = test_database.create_collection(
            "friendship_edges", edge=True
        )
    else:
        friendship_edges = test_database.collection("friendship_edges")

    person_vertices.truncate()
    friendship_edges.truncate()

    # Create graph
    graph_name = "social_network"
    if test_database.has_graph(graph_name):
        test_database.delete_graph(graph_name)

    edge_definition = [
        {
            "edge_collection": "friendship_edges",
            "from_vertex_collections": ["person_vertices"],
            "to_vertex_collections": ["person_vertices"],
        }
    ]

    graph = test_database.create_graph(graph_name, edge_definitions=edge_definition)

    # Add vertices
    graph.vertex_collection("person_vertices").insert_many(
        [
            {"_key": "alice", "name": "Alice", "age": 25},
            {"_key": "bob", "name": "Bob", "age": 30},
            {"_key": "charlie", "name": "Charlie", "age": 35},
        ]
    )

    # Add edges
    graph.edge_collection("friendship_edges").insert_many(
        [
            {
                "_from": "person_vertices/alice",
                "_to": "person_vertices/bob",
                "since": 2020,
            },
            {
                "_from": "person_vertices/bob",
                "_to": "person_vertices/charlie",
                "since": 2021,
            },
        ]
    )

    # Test traversal
    result = test_database.aql.execute(
        """
        FOR v, e, p IN 1..2 OUTBOUND 'person_vertices/alice' 
        GRAPH 'social_network'
        RETURN {vertex: v.name, distance: LENGTH(p.edges)}
    """
    )

    friends = [doc for doc in result]
    assert len(friends) == 2
    assert friends[0]["vertex"] == "Bob"
    assert friends[0]["distance"] == 1
    assert friends[1]["vertex"] == "Charlie"
    assert friends[1]["distance"] == 2

    # Test graph properties
    assert graph.name == graph_name
    assert len(graph.edge_definitions()) == 1
    assert graph.has_vertex_collection("person_vertices")
    assert graph.has_edge_definition("friendship_edges")


def test_update_document(test_collection):
    # Create initial document
    doc = {"name": "test", "value": 100}
    result = test_collection.insert(doc)
    doc_key = result["_key"]

    # Update the document
    new_doc = {"_key": doc_key, "value": 200}
    test_collection.update(new_doc)

    # Verify update
    retrieved = test_collection.get(doc_key)
    assert retrieved["value"] == 200


def test_delete_document(test_collection):
    # Create document
    doc = {"name": "to_delete"}
    result = test_collection.insert(doc)
    doc_key = result["_key"]

    # Delete document
    test_collection.delete(doc_key)

    # Verify get returns None
    result = test_collection.get(doc_key)
    assert result is None


def test_batch_insert(test_collection, test_database):
    # Insert multiple documents via AQL
    docs = [
        {"name": "doc1", "value": 1},
        {"name": "doc2", "value": 2},
        {"name": "doc3", "value": 3},
    ]

    aql = "FOR doc IN @docs INSERT doc INTO @@collection"
    bind_vars = {"@collection": test_collection.name, "docs": docs}
    test_database.aql.execute(aql, bind_vars=bind_vars)

    # Verify documents exist
    assert test_collection.count() == 3


def test_aggregate_query(test_collection, test_database):
    # Insert test data
    for i in range(5):
        doc = {"category": "A" if i < 3 else "B", "value": i * 10}
        test_collection.insert(doc)

    # Run aggregation query
    aql = """
    FOR doc IN @@collection
    COLLECT category = doc.category
    AGGREGATE total = SUM(doc.value), avg = AVG(doc.value)
    RETURN {category, total, avg}
    """

    bind_vars = {"@collection": test_collection.name}
    cursor = test_database.aql.execute(aql, bind_vars=bind_vars)
    result = [doc for doc in cursor]

    assert len(result) == 2
    result = sorted(result, key=lambda x: x["category"])

    assert result[0]["category"] == "A"
    assert result[0]["total"] == 30
    assert result[0]["avg"] == 10

    assert result[1]["category"] == "B"
    assert result[1]["total"] == 70
    assert result[1]["avg"] == 35
