import logging
from typing import List, Optional

from neo4j import GraphDatabase, Record
from neo4j.exceptions import (
    AuthError,
    ConfigurationError,
    Neo4jError,
    ServiceUnavailable,
)
from pydantic import BaseSettings

logger = logging.getLogger(__name__)


class Neo4jConfig(BaseSettings):
    uri: str = "bolt://localhost:7687"
    username: str
    password: str
    database: str


class Neo4j:
    def __init__(self, config: Neo4jConfig):
        self.config = config
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri, auth=(self.config.username, self.config.password)
            )
        except ServiceUnavailable:
            logging.warning(
                """Unable to connect to the database. Please check if the 
                            Neo4j service is running."""
            )
        except AuthError:
            logging.warning(
                """Authentication failed. Please check your 
                            username and password."""
            )
        except ConfigurationError as e:
            logging.warning(
                f"""
                There was a configuration error:, {e}
                """
            )
        except Exception as e:
            logging.warning(f"An unexpected error occurred:, {e}")

    def close(self) -> None:
        """close the connection"""
        if self.driver:
            self.driver.close()

    # TODO: test under enterprise edition because community edition doesn't allow
    # database creation/deletion

    def run_query(self, query: str) -> Optional[List[Record]]:
        """
        Executes a read query on the Neo4j database.

        This method is intended for read operations in the Neo4j database. It opens
        a session, executes the provided Cypher query, and then closes the session.
        The method handles any errors that occur during the execution of the query.

        Args:
            query (str): A Cypher query string that represents the read operation
                         to be performed on the database.

        Returns:
            Optional[List[Record]]: A list of neo4j.Record objects containing the
            results of the query.
            Each Record object is a collection of key-value pairs representing
            the fields returned by the query. If the query does not return anything,
            or if an error occurs, the method returns None.

        Note:
            The method returns None if no database connection is established or
            if an error occurs.
        """
        if not self.driver:
            logging.warning("No database connection is established.")
            return None

        try:
            with self.driver.session(database=self.config.database) as session:
                result = session.run(query)
                return [record for record in result] if result.peek() else None
        except Neo4jError as e:
            logging.warning(
                f"""
                An error occurred while executing the Cypher query:, {e}
                """
            )
        except Exception as e:
            logging.warning(
                f"""
                An unexpected error occurred while executing the query:, {e}
                """
            )
        return None

    def execute_write_query(self, query: str) -> bool:
        """
        Executes a write query on the Neo4j database.

        This method is specifically designed for write operations in the Neo4j database.
        It opens a session, executes the provided Cypher query within a write
        transaction, and then closes the session. The method handles any errors that
        occur during the execution of the query.

        Args:
            query (str): A Cypher query string that represents the write operation
                         to be performed on the database.

        Returns:
            bool: True if the write operation was successful, False otherwise.

        Note:
            The method returns False if no database connection is established or if
            an error occurs.
        """
        if not self.driver:
            logger.warning("No database connection is established.")
            return False

        try:
            with self.driver.session(database=self.config.database) as session:
                session.write_transaction(lambda tx: tx.run(query))
                return True
        except Neo4jError as e:
            logging.warning(
                f"""
                An error occurred while executing the write query:,{e}
                """
            )
        except Exception as e:
            logging.warning(
                f"""
                An unexpected error occurred while executing the write query:, {e}
                """
            )
        return False
