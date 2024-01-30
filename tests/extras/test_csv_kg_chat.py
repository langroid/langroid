import pandas as pd
import pytest
from dotenv import load_dotenv

from langroid.agent.special.neo4j.csv_kg_chat import (
    CSVGraphAgent,
    CSVGraphAgentConfig,
    PandasToKGTool,
)
from langroid.agent.special.neo4j.neo4j_chat_agent import (
    Neo4jSettings,
)

# Create a dummy DataFrame
data = {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["New York", "London"]}
df = pd.DataFrame(data)


@pytest.fixture
def csv_chat_agent(request):
    load_dotenv()
    neo4j_settings = Neo4jSettings()
    config = CSVGraphAgentConfig(data=df, neo4j_settings=neo4j_settings)
    agent = CSVGraphAgent(config)

    def teardown():
        # Remove the database
        agent.remove_database()

    request.addfinalizer(teardown)
    return agent


def test_pandas_to_kg(csv_chat_agent):
    # Cypher query based on the DataFrame columns
    df_columns = ["name", "age", "city"]
    cypher_query = "CREATE (n:Person {"
    for column in df_columns:
        cypher_query += f"{column}: ${column}, "
    cypher_query = cypher_query.rstrip(", ")
    cypher_query += "})"

    # Create a mock PandasToKGTool object
    msg = PandasToKGTool(cypherQuery=cypher_query, args=df_columns)

    # # Set the DataFrame in the agent
    # csv_chat_agent.df = df

    # Call the method being tested
    result = csv_chat_agent.pandas_to_kg(msg)
    assert result == "Graph database successfully generated"

    # Query to obtain the nodes
    query = "MATCH (n:Person) RETURN n"
    query_result = csv_chat_agent.read_query(query)

    # Extract the inner dictionaries
    data_list = [item["n"] for item in query_result.data]

    # Convert the list of dictionaries to a DataFrame and reorder the columns
    nodes_query_df = pd.DataFrame(data_list).reindex(columns=df.columns)

    # Add assert to check nodes_query_result matches the DataFrame
    assert nodes_query_df.equals(df)
