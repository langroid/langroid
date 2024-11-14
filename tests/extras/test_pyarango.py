import os
import subprocess
import time

import pytest
from pyArango.connection import Connection
from pyArango.theExceptions import DocumentNotFoundError


@pytest.fixture(scope="session", autouse=True)
def setup_arango():
    test_dir = os.path.dirname(os.path.abspath(__file__))
    compose_file = os.path.join(test_dir, "docker-compose-arango.yml")
    # Start container using docker-compose
    subprocess.run(
        [
            "docker-compose",
            "-f",
            compose_file,
            "up",
            "-d",
        ],
        check=True,
    )
    time.sleep(10)  # Wait for ArangoDB to start
    yield
    # Cleanup
    subprocess.run(
        [
            "docker-compose",
            "-f",
            compose_file,
            "down",
        ],
        check=True,
    )


@pytest.fixture
def arango_connection():
    conn = Connection(username="root", password="", arangoURL="http://localhost:8529")
    return conn


@pytest.fixture
def test_database(arango_connection):
    # Create test database
    db_name = "test_db"
    if not arango_connection.hasDatabase(db_name):
        db = arango_connection.createDatabase(name=db_name)
    else:
        db = arango_connection[db_name]
    return db


@pytest.fixture
def test_collection(test_database):
    # Create test collection: a collection is like a table in a relational database
    coll_name = "test_collection"
    if not test_database.hasCollection(coll_name):
        collection = test_database.createCollection(name=coll_name)
    else:
        collection = test_database[coll_name]

    # Clear collection before use
    collection.truncate()

    return collection


def test_create_document(test_collection):
    # Create document: this is like inserting a row in a relational database
    doc = test_collection.createDocument()
    doc["name"] = "test"
    doc["value"] = 123
    doc.save()

    # Verify document exists
    retrieved_doc = test_collection.fetchDocument(doc._key)
    assert retrieved_doc["name"] == "test"
    assert retrieved_doc["value"] == 123

    # create document with explicit key
    doc = test_collection.createDocument()
    doc._key = "test_key"
    doc["name"] = "test"
    doc["value"] = 123
    doc.save()

    # Verify document exists
    retrieved_doc = test_collection.fetchDocument(doc._key)
    # verify that the key is the same
    assert retrieved_doc._key == "test_key"
    assert retrieved_doc["name"] == "test"
    assert retrieved_doc["value"] == 123

    # retrieve document using key, with dict-like access, equivalent to above
    retrieved_doc = test_collection["test_key"]
    assert retrieved_doc._key == "test_key"
    assert retrieved_doc["name"] == "test"
    assert retrieved_doc["value"] == 123


def test_query_documents(test_collection):
    # Create multiple documents
    for i in range(5):
        doc = test_collection.createDocument()
        doc["name"] = f"test_{i}"
        doc["value"] = i
        doc.save()

    # Query documents
    aql = "FOR doc IN @@collection FILTER doc.value >= 2 RETURN doc"
    bindVars = {"@collection": test_collection.name}
    result = test_collection.database.AQLQuery(aql, bindVars=bindVars, rawResults=True)

    assert len(result) == 3


def test_knowledge_graph(test_database):
    # Create collections for nodes and edges

    # Create collections for nodes and edges
    if not test_database.hasCollection("nodes"):
        nodes = test_database.createCollection(name="nodes")
    else:
        nodes = test_database["nodes"]

    if not test_database.hasCollection("relationships"):
        relationships = test_database.createCollection(
            name="relationships", className="Edges"
        )
    else:
        relationships = test_database["relationships"]

    nodes.truncate()
    relationships.truncate()

    # Create person nodes
    person1 = nodes.createDocument()
    person1["type"] = "person"
    person1["name"] = "John"
    person1.save()

    person2 = nodes.createDocument()
    person2["type"] = "person"
    person2["name"] = "Mary"
    person2.save()

    # Create location node
    location = nodes.createDocument()
    location["type"] = "location"
    location["name"] = "New York"
    location.save()

    # Create relationships
    lives_in = relationships.createDocument()
    lives_in._from = person1._id
    lives_in._to = location._id
    lives_in["type"] = "LIVES_IN"
    lives_in.save()

    knows = relationships.createDocument()
    knows._from = person1._id
    knows._to = person2._id
    knows["type"] = "KNOWS"
    knows.save()

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
    result = test_database.AQLQuery(aql, rawResults=True)

    assert len(result) == 2
    assert result[0]["person"] == "John"
    assert result[0]["livesIn"] == "New York"
    assert result[1]["person"] == "Mary"
    assert result[1]["livesIn"] is None


def test_update_document(test_collection):
    # Create initial document
    doc = test_collection.createDocument()
    doc["name"] = "test"
    doc["value"] = 100
    doc.save()

    # Update the document
    doc["value"] = 200
    doc.save()

    # Verify update
    retrieved = test_collection[doc._key]
    assert retrieved["value"] == 200


def test_delete_document(test_collection):
    # Create document
    doc = test_collection.createDocument()
    doc["name"] = "to_delete"
    doc.save()

    # Store the key before deletion
    doc_key = doc._key

    # Delete document
    doc.delete()

    # Verify deletion using DocumentNotFoundError
    with pytest.raises(DocumentNotFoundError):
        test_collection.fetchDocument(doc_key)


def test_batch_insert(test_collection):
    # Insert multiple documents via AQL
    docs = [
        {"name": "doc1", "value": 1},
        {"name": "doc2", "value": 2},
        {"name": "doc3", "value": 3},
    ]

    aql = "FOR doc IN @docs INSERT doc INTO @@collection"
    bindVars = {"@collection": test_collection.name, "docs": docs}
    test_collection.database.AQLQuery(aql, bindVars=bindVars)

    # Verify documents exist
    assert test_collection.count() == 3


def test_aggregate_query(test_collection):
    # Insert test data
    for i in range(5):
        doc = test_collection.createDocument()
        doc["category"] = "A" if i < 3 else "B"
        doc["value"] = i * 10
        doc.save()

    # Run aggregation query
    aql = """
    FOR doc IN @@collection
    COLLECT category = doc.category
    AGGREGATE total = SUM(doc.value), avg = AVG(doc.value)
    RETURN {category, total, avg}
    """

    result = test_collection.database.AQLQuery(
        aql, bindVars={"@collection": test_collection.name}, rawResults=True
    )

    assert len(result) == 2
    result = sorted(result, key=lambda x: x["category"])

    assert result[0]["category"] == "A"
    assert result[0]["total"] == 30
    assert result[0]["avg"] == 10

    assert result[1]["category"] == "B"
    assert result[1]["total"] == 70
    assert result[1]["avg"] == 35
