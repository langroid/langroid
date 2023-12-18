import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import neo4j

from pydantic import BaseSettings

logger = logging.getLogger(__name__)


class Neo4jConfig(BaseSettings):
    uri: str
    username: str
    password: str
    database: str


class Neo4j:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver = None
        self._import_neo4j()
        self._initialize_connection()

    def _import_neo4j(self) -> None:
        global neo4j
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                """
                neo4j not installed. Please install it via:
                pip install neo4j.
                Or when installing langroid, install it with the `neo4j` extra:
                pip install langroid[neo4j]
                """
            )

    def _initialize_connection(self) -> None:
        try:
            self.driver = neo4j.GraphDatabase.driver(
                self.config.uri, auth=(self.config.username, self.config.password)
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Neo4j connection: {e}")

    def close(self) -> None:
        """close the connection"""
        if self.driver:
            self.driver.close()

    # TODO: test under enterprise edition because community edition doesn't allow
    # database creation/deletion

    def run_query(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Executes a given Cypher query on the Neo4j database.

        Args:
            query (str): The Cypher query string to be executed.

        Returns:
            Optional[List[Dict[str, Any]]]: A list of dictionaries representing the
            query results.Each dictionary is a record in the query result.
            Returns None if the query execution fails or if no records are found.
        """
        if not self.driver:
            logging.warning("No database connection is established.")
            return None

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query)
                # Convert Neo4j records to a list of dictionaries, if there are results
                return [record.data() for record in result] if result.peek() else None
        except neo4j.Neo4jError as e:
            logging.warning(f"An error occurred while executing the Cypher query: {e}")
        except Exception as e:
            logging.warning(
                f"An unexpected error occurred while executing the query: {e}"
            )
        return None

    def execute_write_query(self, query: str) -> bool:
        """
        Executes a write transaction using a given Cypher query on the Neo4j database.

        This method should be used for queries that modify the database, such as CREATE,
        UPDATE, or DELETE operations.

        Args:
            query (str): The Cypher query string to be executed.

        Returns:
            bool: True if the query was executed successfully, False otherwise.
        """
        if not self.driver:
            logging.warning("No database connection is established.")
            return False

        try:
            with self.driver.session(database=self.config.database) as session:
                # Execute the query within a write transaction
                session.write_transaction(lambda tx: tx.run(query))
                return True
        except Exception as e:
            logging.warning(
                f"An unexpected error occurred while executing the write query: {e}"
            )
        return False
