import logging
import urllib.parse

from rich import print
from rich.prompt import Prompt

from langroid.parsing.utils import closest_string

logger = logging.getLogger(__name__)


DEFAULT_PORTS = dict(
    postgresql=5432,
    mysql=3306,
    mariadb=3306,
    mssql=1433,
    oracle=1521,
    mongodb=27017,
    redis=6379,
)


def fix_uri(uri: str) -> str:
    """Fixes a URI by percent-encoding the username and password."""

    if "%" in uri:
        return uri  # already %-encoded, so don't do anything
    # Split by '://'
    scheme_part, rest_of_uri = uri.split("://", 1)

    # Get the final '@' (assuming only the last '@' is the separator for user info)
    last_at_index = rest_of_uri.rfind("@")
    userinfo_part = rest_of_uri[:last_at_index]
    rest_of_uri_after_at = rest_of_uri[last_at_index + 1 :]

    if ":" not in userinfo_part:
        return uri
    # Split userinfo by ':' to get username and password
    username, password = userinfo_part.split(":", 1)

    # Percent-encode the username and password
    username = urllib.parse.quote(username)
    password = urllib.parse.quote(password)

    # Construct the fixed URI
    fixed_uri = f"{scheme_part}://{username}:{password}@{rest_of_uri_after_at}"

    return fixed_uri


def _create_database_uri(
    scheme: str,
    username: str,
    password: str,
    hostname: str,
    port: int,
    databasename: str,
) -> str:
    """Generates a database URI based on provided parameters."""
    username = urllib.parse.quote_plus(username)
    password = urllib.parse.quote_plus(password)
    port_str = f":{port}" if port else ""
    return f"{scheme}://{username}:{password}@{hostname}{port_str}/{databasename}"


def get_database_uri() -> str:
    """Main function to gather input and print the database URI."""
    scheme_input = Prompt.ask("Enter the database type (e.g., postgresql, mysql)")
    scheme = closest_string(scheme_input, list(DEFAULT_PORTS.keys()))

    # Handle if no close match is found.
    if scheme == "No match found":
        print(f"No close match found for '{scheme_input}'. Please verify your input.")
        return

    username = Prompt.ask("Enter the database username")
    password = Prompt.ask("Enter the database password", password=True)
    hostname = Prompt.ask("Enter the database hostname")

    # Inform user of default port, and let them choose to override or leave blank
    default_port = DEFAULT_PORTS.get(scheme, "")
    port_msg = (
        f"Enter the database port "
        f"(hit enter to use default: {default_port} or specify another value)"
    )

    port = Prompt.ask(port_msg, default=default_port)
    if not port:  # If user pressed enter without entering anything
        port = default_port
    port = int(port)

    databasename = Prompt.ask("Enter the database name")

    uri = _create_database_uri(scheme, username, password, hostname, port, databasename)
    print(f"Your {scheme.upper()} URI is:\n{uri}")
    return uri
