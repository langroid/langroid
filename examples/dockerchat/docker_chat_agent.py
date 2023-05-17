from llmagent.agent.chat_agent import ChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    FileExistsMessage,
    PythonVersionMessage,
    PythonDependencyMessage,
    ValidateDockerfileMessage,
)
from examples.dockerchat.identify_python_version import get_python_version
from examples.dockerchat.identify_python_dependency import (
    identify_dependency_management,
)

import subprocess
import os
import logging

logger = logging.getLogger(__name__)

# Message types that can be handled by the agent;
# each corresponds to a method in the agent.




class DockerChatAgent(ChatAgent):
    repo_path: str = "/nobackup/images_repos/Auto-GPT"

    def python_version(self, PythonVersionMessage) -> str:
        """
        Identifies Python version for a given repo
        Args:
        Returns:
            str: a string indicates the identified python version or indicate the version can't be identified
        """
        python_version = get_python_version(self.repo_path)
        if python_version:
            return python_version
        else:
            logger.error("Could not determine Python version.")
        return "Couldn't identify the python version"

    def file_exists(self, message: FileExistsMessage) -> str:
        # dummy result, fill with actual code.
        if message.filename == "requirements.txt":
            return f"""
            Yes, there is a file named {message.filename} in the repo."""
        else:
            return f"""
            No, there is no file named {message.filename} in the repo."""

    def python_dependency(self, PythonDependencyMessage) -> str:
        """
        Identifies Python dependencies in a given repo by inspecting various artifacts like requirements.txt
        Args:
        Returns:
            str: a string indicates the identified the dependency management approach
        """
        python_dependency = identify_dependency_management(self.repo_path)
        if python_dependency:
            return f"Dependencies in this repo are managed using: {python_dependency}"
        else:
            return "Dependencies are not defined in this repo"

    def _cleanup_dockerfile(self, img_name: str, dockerfile_path: str) -> None:
        """
        Remove Dockefile and built image after performing the verification process
        Args:
            img_name (str): the name of the Docker image
            dockerfile_path (str): path to the saved Dockerfile
        """
        try:
            # Remove Dockerfile
            if os.path.exists(dockerfile_path):
                os.remove(dockerfile_path)
                logger.info(f"Dockerfile at path '{dockerfile_path}' has been removed.")
            else:
                logger.error(f"No Dockerfile found at path '{dockerfile_path}'.")

            # Remove Docker image
            command = f"docker rmi -f {img_name}"
            result = subprocess.run(
                command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # Check if the command was successful
            if result.returncode == 0:
                logger.info(f"Docker image '{img_name}' has been removed.")
            else:
                logger.error(
                    f"Failed to remove Docker image '{img_name}'. Error: {result.stderr.decode()}"
                )

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")

    def _save_dockerfile(self, dockerfile: str) -> str:
        """
        Save the proposed Dockerfile in the root directory of a repo
        Args:
            dockerfile (str): content of the dockerfile
        Returns:
            str: a string indicates whether the Dockerfile has been saved successfully
        """
        try:
            full_path = os.path.join(self.repo_path, "Dockerfile")
            with open(full_path, "w") as f:
                f.write(dockerfile)
            return full_path
        except Exception as e:
            return f"An error occurred while saving the Dockerfile: {e}"

    def validate_dockerfile(self, dockerfile_msg: ValidateDockerfileMessage) -> str:
        """
        validates the proposed Dockerfile by LLM. The validation process involves saving the proposed_dockerfile, building the image, and finally cleanning up
        Args:
            dockerfile_msg (ValidateDockerfileMessage): LLM message contains the definition of the Dockerfile
        Returns:
            str: a string indicates whether the Dockerfile has been built successfully
        """
        proposed_dockerfile = dockerfile_msg.proposed_dockerfile
        try:
            dockerfile_path = self._save_dockerfile(proposed_dockerfile)
            if dockerfile_path.startswith("An error"):
                return dockerfile_path

            original_path = os.getcwd()
            os.chdir(self.repo_path)

            # Build the Docker image
            img_name = "validate_img"
            command = f"docker build -t {img_name} -f {dockerfile_path} ."
            process = subprocess.run(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            os.chdir(original_path)

            # Check the result of the build process
            if process.returncode == 0:
                # do some cleaning: remove the Docker image and the Dockerfile
                self._cleanup_dockerfile(img_name, dockerfile_path)
                return "Docker build was successful"
            else:
                return f"Docker build failed with error message: {process.stderr}"

        except Exception as e:
            return f"An error occurred during the Docker build: {e}"
