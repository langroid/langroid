import os
import subprocess
import time

import pytest
from adb_cloud_connector import get_temp_credentials
from arango.client import ArangoClient
from arango_datasets import Datasets

import langroid as lr
from langroid.agent.special.arangodb.arangodb_agent import (
    ArangoChatAgent,
    ArangoChatAgentConfig,
    ArangoSettings,
)

ARANGO_PASSWORD = "rootpassword"


def wait_for_arango(max_attempts=30, delay=1):
    """Try to connect to ArangoDB until it's ready"""
    client = None
    for attempt in range(max_attempts):
        try:
            client = ArangoClient(hosts="http://localhost:8529")
            sys_db = client.db("_system", username="root", password=ARANGO_PASSWORD)
            sys_db.version()  # test connection
            print(f"ArangoDB ready after {attempt + 1} attempts")
            return True
        except Exception:
            print(f"Waiting for ArangoDB... ({attempt + 1}/{max_attempts})")
            time.sleep(delay)
    raise TimeoutError("ArangoDB failed to start")


COMPOSE_FILE = os.path.join(os.path.dirname(__file__), "docker-compose-arango.yml")


def docker_setup_arango():
    subprocess.run(
        ["docker-compose", "-f", COMPOSE_FILE, "down", "--remove-orphans"],
        check=True,
    )
    subprocess.run(
        ["docker-compose", "-f", COMPOSE_FILE, "up", "-d"],
        check=True,
    )


def docker_teardown_arango():
    subprocess.run(
        ["docker-compose", "-f", COMPOSE_FILE, "down"],
        check=True,
    )


@pytest.fixture(scope="session", autouse=True)
def setup_arango():
    if not os.getenv("CI"):
        docker_setup_arango()
    wait_for_arango()
    yield
    if not os.getenv("CI"):
        docker_teardown_arango()


@pytest.fixture
def arango_client():
    client = ArangoClient(hosts="http://localhost:8529")
    return client


@pytest.fixture
def test_database(arango_client):
    sys_db = arango_client.db("_system", username="root", password=ARANGO_PASSWORD)
    db_name = "test_db"
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
    return arango_client.db(db_name, username="root", password=ARANGO_PASSWORD)


@pytest.fixture
def arango_movie_agent(setup_arango, test_database):

    # Create graph
    graph_name = "MovieGraph"
    ArangoChatAgent.cleanup_graph_db(test_database)

    graph = test_database.create_graph(graph_name)

    # Create collections with the graph
    actors = graph.create_vertex_collection("actors")
    movies = graph.create_vertex_collection("movies")
    acted_in = graph.create_edge_definition(
        edge_collection="acted_in",
        from_vertex_collections=["actors"],
        to_vertex_collections=["movies"],
    )

    # Sample data
    actor_data = [
        {"_key": "meryl", "name": "Meryl Streep", "age": 74, "oscars": 3},
        {"_key": "tom", "name": "Tom Hanks", "age": 67, "oscars": 2},
        {"_key": "leo", "name": "Leonardo DiCaprio", "age": 48, "oscars": 1},
        {"_key": "viola", "name": "Viola Davis", "age": 58, "oscars": 1},
    ]

    movie_data = [
        {
            "_key": "devil",
            "title": "Devil Wears Prada",
            "year": 2006,
            "genre": "Comedy",
            "rating": 7.7,
        },
        {
            "_key": "forrest",
            "title": "Forrest Gump",
            "year": 1994,
            "genre": "Drama",
            "rating": 8.8,
        },
        {
            "_key": "inception",
            "title": "Inception",
            "year": 2010,
            "genre": "Sci-Fi",
            "rating": 8.8,
        },
        {
            "_key": "fences",
            "title": "Fences",
            "year": 2016,
            "genre": "Drama",
            "rating": 7.2,
        },
    ]

    relationship_data = [
        {"_from": "actors/meryl", "_to": "movies/devil"},
        {"_from": "actors/tom", "_to": "movies/forrest"},
        {"_from": "actors/leo", "_to": "movies/inception"},
        {"_from": "actors/viola", "_to": "movies/fences"},
    ]

    try:
        actors.import_bulk(actor_data, on_duplicate="replace")
        movies.import_bulk(movie_data, on_duplicate="replace")
        acted_in.import_bulk(relationship_data, on_duplicate="replace")
    except Exception as e:
        print(f"Error inserting data: {e}")
        raise

    agent = ArangoChatAgent(
        ArangoChatAgentConfig(
            arango_settings=ArangoSettings(
                url="http://localhost:8529",
                username="root",
                password=ARANGO_PASSWORD,
                database="test_db",
            ),
            prepopulate_schema=True,
            use_functions_api=False,
            use_tools=True,
            database_created=True,
        )
    )

    yield agent

    ArangoChatAgent.cleanup_graph_db(test_database)


@pytest.mark.parametrize(
    "english_query,aql_query,expected",
    [
        (
            "What movies has Tom Hanks acted in?",
            """
        FOR actor IN actors
            FILTER actor.name == 'Tom Hanks'
            FOR v, e IN 1..1 OUTBOUND actor acted_in
                RETURN v.title
        """,
            "Forrest Gump",
        ),
        (
            "Who starred in Forrest Gump?",
            """
        FOR movie IN movies
            FILTER movie.title == 'Forrest Gump'
            FOR v, e IN 1..1 INBOUND movie acted_in
                RETURN v.name
        """,
            "Tom Hanks",
        ),
    ],
)
def test_retrieval(arango_movie_agent, english_query, aql_query, expected):
    # Test via direct AQL
    aql_result = arango_movie_agent.read_query(aql_query)
    assert expected.lower() in aql_result.data[0].lower()

    # Test via natural language
    task = lr.Task(arango_movie_agent, interactive=False)
    nl_result = task.run(
        f"""
        Use the `aql_retrieval_tool` to find the answer to this question:
        {english_query}
        """
    )
    assert expected.lower() in nl_result.content.lower()


def test_write_query(arango_movie_agent):
    # Write a new actor
    write_result = arango_movie_agent.write_query(
        """
        INSERT { 
            _key: 'morgan', 
            name: 'Morgan Freeman', 
            age: 86, 
            oscars: 1 
        } INTO actors
        """
    )
    assert write_result.success

    # Verify the write
    read_result = arango_movie_agent.read_query(
        "FOR a IN actors FILTER a._key == 'morgan' RETURN a.name"
    )
    assert "Morgan Freeman" in read_result.data[0]


@pytest.fixture
def number_kg_agent(setup_arango, test_database):
    graph_name = "NumberKG"
    ArangoChatAgent.cleanup_graph_db(test_database)

    graph = test_database.create_graph(graph_name)
    numbers = graph.create_vertex_collection("numbers")
    divides = graph.create_edge_definition(
        edge_collection="divides",
        from_vertex_collections=["numbers"],
        to_vertex_collections=["numbers"],
    )

    # Create numbers
    number_list = [2, 3, 4, 6, 12]
    numbers.import_bulk([{"_key": f"n{i}", "value": i} for i in number_list])

    # Create edges based on divisibility
    edge_data = [
        {"_key": f"{i}_{j}", "_from": f"numbers/n{i}", "_to": f"numbers/n{j}"}
        for i in number_list
        for j in number_list
        if i < j and j % i == 0  # i divides j
    ]
    divides.import_bulk(edge_data)

    plus4 = graph.create_edge_definition(
        edge_collection="plus4",
        from_vertex_collections=["numbers"],
        to_vertex_collections=["numbers"],
    )

    # Add plus4 edges:
    plus4_edges = [
        {"_key": f"plus4_{i}_{i+4}", "_from": f"numbers/n{i}", "_to": f"numbers/n{i+4}"}
        for i in number_list
        if i + 4 in number_list
    ]
    plus4.import_bulk(plus4_edges)

    agent = ArangoChatAgent(
        config=ArangoChatAgentConfig(
            arango_settings=ArangoSettings(
                url="http://localhost:8529",
                username="root",
                password=ARANGO_PASSWORD,
                database="test_db",
            ),
            max_tries=20,
            use_tools=True,
            use_functions_api=False,
            prepopulate_schema=False,
            database_created=True,
        )
    )

    yield agent
    ArangoChatAgent.cleanup_graph_db(test_database)


@pytest.mark.fallback
@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize(
    "english_query,aql_query,expected",
    [
        (
            "What numbers divide 12?",
            """
        FOR v IN 1..1 INBOUND 'numbers/n12' divides
            RETURN v.value
        """,
            [2, 3, 4, 6],
        ),
        (
            "What numbers are divided by 2?",
            """
        FOR v IN 1..1 OUTBOUND 'numbers/n2' divides
            RETURN v.value
        """,
            [4, 6, 12],
        ),
        (
            "what is a number that 2 divides and is plus4 from 2?",
            """
          FOR v IN 1..1 OUTBOUND 'numbers/n2' divides
              FILTER v._id IN (
                  FOR v2 IN 1..1 OUTBOUND 'numbers/n2' plus4
                      RETURN v2._id
              )
              RETURN v.value
          """,
            [6],
        ),
    ],
)
def test_number_relationships(
    number_kg_agent,
    english_query,
    aql_query,
    expected,
):
    # Test via direct AQL
    aql_result = number_kg_agent.read_query(aql_query)
    assert sorted(aql_result.data) == sorted(expected)

    # Test via natural language
    task = lr.Task(number_kg_agent, interactive=False)
    nl_result = task.run(
        f"""
        Answer the following using the graph-db whose schema was provided above,
        using the appropriate AQL tools provided.
        DO NOT use your own knowledge!!
        {english_query}
        """
    )
    assert all(str(num) in nl_result.content for num in expected)


def test_db_schema(number_kg_agent):
    schema_data = number_kg_agent.arango_schema_tool(None)

    # Check schema structure
    assert isinstance(schema_data, dict)
    assert "Graph Schema" in schema_data
    assert "Collection Schema" in schema_data

    # Check graph schema
    graph_schema = schema_data["Graph Schema"]
    assert isinstance(graph_schema, list)
    assert len(graph_schema) == 1
    assert graph_schema[0]["graph_name"] == "NumberKG"

    # Check collection schema
    collection_schema = schema_data["Collection Schema"]
    assert isinstance(collection_schema, list)
    assert len(collection_schema) == 3

    # Get collection info
    numbers_coll = next(
        c for c in collection_schema if c["collection_name"] == "numbers"
    )
    divides_coll = next(
        c for c in collection_schema if c["collection_name"] == "divides"
    )
    plus4_coll = next(c for c in collection_schema if c["collection_name"] == "plus4")

    # Verify numbers collection properties
    number_props = numbers_coll["document_properties"]
    assert any(p["name"] == "_key" for p in number_props)
    assert any(p["name"] == "value" for p in number_props)

    # Verify divides collection properties
    edge_props = divides_coll["edge_properties"]
    assert any(p["name"] == "_from" for p in edge_props)
    assert any(p["name"] == "_to" for p in edge_props)
    assert any(p["name"] == "_key" for p in edge_props)

    # Verify plus4 collection properties
    edge_props = plus4_coll["edge_properties"]
    assert any(p["name"] == "_from" for p in edge_props)
    assert any(p["name"] == "_to" for p in edge_props)
    assert any(p["name"] == "_key" for p in edge_props)


def test_multiple_relationships(number_kg_agent):
    # Query to verify divides relationships
    divides_query = """
    FOR p IN numbers
        FILTER p.value == 2
        FOR v, e IN 1..1 OUTBOUND p divides
        RETURN { 
            relationship_type: 'divides',
            connected_to: v.value 
        }
    """
    divides_result = number_kg_agent.read_query(divides_query)
    divides_values = [r["connected_to"] for r in divides_result.data]
    assert set(divides_values) == {4, 6, 12}

    # Query to verify plus4 relationships
    plus4_query = """
    FOR p IN numbers
        FILTER p.value == 2
        FOR v, e IN 1..1 OUTBOUND p plus4
        RETURN {
            relationship_type: 'plus4',
            connected_to: v.value
        }
    """
    plus4_result = number_kg_agent.read_query(plus4_query)
    plus4_values = [r["connected_to"] for r in plus4_result.data]
    assert set(plus4_values) == {6}


def test_arangodb_cloud_datasets():
    connection = get_temp_credentials(tutorialName="langroid")
    client = ArangoClient(hosts=connection["url"])

    db = client.db(
        connection["dbName"],
        connection["username"],
        connection["password"],
        verify=True,
    )

    datasets = Datasets(db)
    assert len(datasets.list_datasets()) > 0
    DATASET = "IMDB_X"
    info = datasets.dataset_info(DATASET)
    assert info["label"] == DATASET


@pytest.fixture(scope="session")
def arango_agent_from_db():
    """Arango Agent created from a cloud arango dataset"""

    connection = get_temp_credentials(tutorialName="langroid")
    client = ArangoClient(hosts=connection["url"])

    db = client.db(
        connection["dbName"],
        connection["username"],
        connection["password"],
        verify=True,
    )

    ArangoChatAgent.cleanup_graph_db(db)

    datasets = Datasets(db)
    DATASET = "GAME_OF_THRONES"
    info = datasets.dataset_info(DATASET)
    datasets.load(DATASET, batch_size=100, preserve_existing=False)
    print("Info of loaded db: ", info)

    agent = ArangoChatAgent(
        ArangoChatAgentConfig(
            arango_settings=ArangoSettings(
                db=db,
                client=client,
            ),
            prepopulate_schema=True,
            use_functions_api=True,
            use_tools=False,
            database_created=True,
        )
    )

    yield agent
    ArangoChatAgent.cleanup_graph_db(db)


@pytest.mark.parametrize(
    "query,expected",
    [
        ("Who are the two youngest characters?", "Bran Stark, Arya Stark"),
        ("Are Bran Stark and Arya Stark siblings?", "yes"),
        ("Who are Bran Stark's grandparents?", "Rickard, Lyarra"),
        ("What is the age difference between Rickard Stark and Arya Stark?", "49"),
        ("What is the average age of all Stark characters?", "31"),
        ("Does Bran Stark have a dead parent? Say yes or no", "yes"),
    ],
)
def test_GOT_queries(arango_agent_from_db, query, expected):
    # Test natural language query about a popular movie
    task = lr.Task(
        arango_agent_from_db,
        interactive=False,
        restart=True,
    )
    result = task.run(query)

    exp_answers = [r.strip().lower() for r in expected.split(",")]
    assert all(exp in result.content.lower() for exp in exp_answers)
