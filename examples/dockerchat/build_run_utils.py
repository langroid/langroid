import docker
import os
import time
import datetime
import logging
import subprocess
from typing import Tuple


from rich.console import Console


def _save_dockerfile(
    repo_path: str, dockerfile: str, proposed_dockerfile_name: str
) -> str:
    """
    Save the proposed Dockerfile in the root directory of a repo
    Args:
        dockerfile (str): content of the dockerfile
        proposed_dockerfile_name (str): the name of the Dockerfile,
            better to use a different name to avoid changing existing one (if any).
    Returns:
        str: a string indicates whether the Dockerfile has been saved successfully
    """
    try:
        full_path = os.path.join(repo_path, proposed_dockerfile_name)
        with open(full_path, "w") as f:
            f.write(dockerfile)
        return full_path
    except Exception as e:
        return f"An error occurred while saving the Dockerfile: {e}"


def _build_docker_image(repo_path: str, proposed_doeckerfile_name: str, img_tag: str):
    """
    Build docker image based on the repo_path by using docker SDK
    Args:
        proposed_doeckerfile_name (str): the name of the proposed Dockerfile
        that should be used to build the image
        img_tag (str): the name of the Docker image that will be built based on the
        proposed_doeckerfile_name
    Returns:
        A tuple comprises three items: First, object for the image that was
        built (if succeeded), otherwise, returns None. Second, message indicates
        whetehr the build process succeeded or failed.
        Third, build time or None (if failed)
    """
    console = Console()
    try:
        start = time.time()
        # I noticed the flag ``rm`` isn't used anymore,
        # so I need to do the cleanup myself later on
        with console.status("Verifying the proposed Dockerfile..."):
            image, build_logs = docker.from_env().images.build(
                rm=True,
                path=repo_path,
                tag=img_tag,
                dockerfile=proposed_doeckerfile_name,
            )
        build_time = time.time() - start
        formatted_build_time = "{:.2f}".format(
            datetime.timedelta(seconds=build_time).total_seconds()
        )
    except docker.errors.DockerException as e:
        return (None, f"Image build failed: {e}", None)
    return (image, "Image build successful!", formatted_build_time)


def _cleanup_dockerfile(img_id: str, dockerfile_path: str) -> None:
    """
    Remove Dockefile and built image after performing the verification process
    Args:
        img_id (str): the ID of the Docker image
        dockerfile_path (str): path to the saved Dockerfile
    """
    logger = logging.getLogger(__name__)
    client = docker.from_env()

    try:
        if os.path.exists(dockerfile_path):
            os.remove(dockerfile_path)
            logger.info(f"Dockerfile at path '{dockerfile_path}' has been removed.")
        else:
            logger.error(f"No Dockerfile found at path '{dockerfile_path}'.")
        # Remove Dockerfile_proposed
        client.images.remove(img_id, force=True)
        client.images.get(img_id)
        logger.error("Image removal failed!")
    except docker.errors.ImageNotFound:
        logger.info("Image removed successfully!")


def _execute_command(cmd: str) -> Tuple[bool, str]:
    """
    Executes a command and returns the results
    Args:
        cmd (str): the command to be executed
    Returns:
        A tuple where the 1st element is return code and the 2nd element a log.
        This log will be stdout in case the command executed successfully,
        otherwise, it'll be stderr
    """
    # Split the command into a list of commands
    commands = cmd.split("|")
    commands = [command.strip().split() for command in commands]

    # Set up the first subprocess
    process = subprocess.Popen(
        commands[0], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Set up the remaining subprocesses, if any
    for command in commands[1:]:
        process = subprocess.Popen(
            command,
            stdin=process.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    # Get the output and error (if any)
    stdout, stderr = process.communicate()

    # Check if the command was executed successfully
    if process.returncode == 0:
        return True, stdout.decode()
    else:
        return False, stderr.decode()
