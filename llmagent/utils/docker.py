from llmagent.utils.configuration import settings
import docker
import os
import logging
logger = logging.getLogger(__name__)

def launch(file_path:str, name:str) -> None:
    '''
    Launch docker-compose file with given name, using python SDK rather than CLI.
    This lets us eliminate the extra step of having to `run docker-compose up` on the
    command line before running the main script.

    Args:
        file_path: relative path to docker-compose file
        name: name of docker-compose project (i.e. container prefix)

    '''
    # Load docker-compose file
    compose_file = os.path.abspath(file_path)

    # Connect to Docker daemon
    client = docker.from_env()

    # Get list of running containers
    containers = client.containers.list()

    # Check if containers defined in docker-compose file are already running
    for container in containers:
        if container.name.startswith(name):
            logging.info(f"Containers are already running, e.g.: {container.name}")
            return

    # Create project from docker-compose file
    project = docker.compose.project.from_filename(
        compose_file,
        project_name=name,
        client=client,
    )

    # Start containers
    project.up()
