from llmagent.agent.chat_agent import ChatAgent
from pydantic import BaseModel, HttpUrl
from typing import Optional
from examples.dockerchat.dockerchat_agent_messages import (
    InformURLMessage,
    FileExistsMessage,
    PythonVersionMessage,
    PythonDependencyMessage,
    ValidateDockerfileMessage,
)
from halo import Halo
from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from examples.dockerchat.identify_python_version import get_python_version
from examples.dockerchat.identify_python_dependency import (
    identify_dependency_management,
    DEPENDENCY_FILES,
)

import os
import json
import logging
import docker
import time
import datetime


logger = logging.getLogger(__name__)

# Message types that can be handled by the agent;
# each corresponds to a method in the agent.


class UrlModel(BaseModel):
    url: HttpUrl


class DockerChatAgent(ChatAgent):
    url: str = "https://github.com/eugeneyan/testing-ml"
    repo_tree: str = None
    repo_path: str = None

    def handle_message_fallback(self, input_str: str = "") -> Optional[str]:
        if self.repo_path is None and "URL" not in input_str:
            return """
            You have not sent me the URL for the repo yet. 
            Please ask me for the URL, and once you receive it, 
            send it to me for confirmation. Once I confirm the URL, 
            you can proceed.
            """

    def inform_url(self, msg: InformURLMessage) -> str:
        try:
            url_model = UrlModel(url=msg.url)
        except ValueError as e:
            return f"""
            A valid URL was not seen: {e}
            Please ask me for the URL before proceeding. 
            And once you receive a URL, 
            please reconfirm it by showing it to me.
            """

        self.url = url_model.url
        self.repo_loader = RepoLoader(self.url, RepoLoaderConfig())
        self.repo_path = self.repo_loader.clone()
        # get the repo tree to depth d, with first k lines of each file
        self.repo_tree = self.repo_loader.get_folder_structure(depth=1, lines=20)
        selected_tree = RepoLoader.select(
            self.repo_tree,
            names=DEPENDENCY_FILES,
        )
        repo_listing = "\n".join(self.repo_loader.ls(self.repo_tree, depth=1))
        repo_contents = json.dumps(selected_tree, indent=2)

        return f"""
        Ok, confirmed, and here is some information about the repo that you can use.
        
        First, here is a listing of the files and directories at the root of the repo:
        {repo_listing}
        
        And here is a JSON representation of the contents of some of the files:
        {repo_contents}
        
        Before proceeding, based on the above information, tell me what 
        information you can gather, to help you with your task. 
        For each piece of information, indicate with "SOURCE:" where you
        got the information from.
        Once you show me the information you are able to infer, 
        you can proceed with your next question or request for information. 

        If you still need further information, you can ask me. 
        """

    def python_version(self, m: PythonVersionMessage) -> str:
        """
        Identifies Python version for a given repo
        Args:
        Returns:
            str: a string indicates the identified python version or indicate the version can't be identified
        """
        if self.repo_path is None:
            return self.handle_message_fallback()
        python_version = get_python_version(self.repo_path)
        if python_version:
            return python_version
        else:
            logger.error("Could not determine Python version.")
        return "Couldn't identify the python version"

    def file_exists(self, message: FileExistsMessage) -> str:
        if self.repo_path is None:
            return self.handle_message_fallback()
        # dummy result, fill with actual code.
        matches = RepoLoader.select(self.repo_tree, names=[message.filename])
        exists = False
        if len(matches) > 0:
            exists = len(matches["files"]) > 0

        if exists:
            return f"""
            Yes, there is a file named {message.filename} in the repo."""
        else:
            return f"""
            No, there is no file named {message.filename} in the repo."""

    def python_dependency(self, m: PythonDependencyMessage) -> str:
        """
        Identifies Python dependencies in a given repo by inspecting various artifacts like requirements.txt
        Args:
        Returns:
            str: a string indicates the identified the dependency management approach
        """
        if self.repo_path is None:
            return self.handle_message_fallback()

        python_dependency = identify_dependency_management(self.repo_path)
        if python_dependency:
            return f"Dependencies in this repo are managed using: {python_dependency}"
        else:
            return "Dependencies are not defined in this repo"

    def _cleanup_dockerfile(self, img_id: str, dockerfile_path: str) -> None:
        """
        Remove Dockefile and built image after performing the verification process
        Args:
            img_id (str): the ID of the Docker image
            dockerfile_path (str): path to the saved Dockerfile
        """
        client = docker.from_env()

        try:
            if os.path.exists(dockerfile_path):
                os.remove(dockerfile_path)
                logger.info(f"Dockerfile at path '{dockerfile_path}' has been removed.")
            else:
                logger.error(f"No Dockerfile found at path '{dockerfile_path}'.")
            # Remove Dockerfile_proposed
            client.images.remove(img_id)
            client.images.get(img_id)
            logger.error("Image removal failed!")
        except docker.errors.ImageNotFound:
            logger.info("Image removed successfully!")

    def _save_dockerfile(self, dockerfile: str, proposed_dockerfile_name: str) -> str:
        """
        Save the proposed Dockerfile in the root directory of a repo
        Args:
            dockerfile (str): content of the dockerfile
            proposed_dockerfile_name (str): the name of the Dockerfile, better to use a different name to avoid changing existing one (if any).
        Returns:
            str: a string indicates whether the Dockerfile has been saved successfully
        """
        try:
            full_path = os.path.join(self.repo_path, proposed_dockerfile_name)
            with open(full_path, "w") as f:
                f.write(dockerfile)
            return full_path
        except Exception as e:
            return f"An error occurred while saving the Dockerfile: {e}"

    def _build_docker_image(self, proposed_doeckerfile_name: str, img_tag: str):
        """
        Build docker image based on the repo_path by using docker SDK
        Args:
            proposed_doeckerfile_name (str): the name of the proposed Dockerfile that should
            be used to build the image
            img_tag (str): the name of the Docker image that will be built based on the
            proposed_doeckerfile_name
        Returns:
            A tuple comprises three items: First, object for the image that was
            built (if succeeded), otherwise, returns None. Second, message indicates
            whetehr the build process succeeded or failed. Third, build time or None (if failed)
        """
        try:
            start = time.time()
            # I noticed the flag ``rm`` isn't used anymore, so I need to do the cleanup myself later on
            with Halo(text="Verifying the proposed Dockerfile...", spinner="dots"):
                image, build_logs = docker.from_env().images.build(
                    rm=True,
                    path=self.repo_path,
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

    def validate_dockerfile(self, dockerfile_msg: ValidateDockerfileMessage) -> str:
        """
        validates the proposed Dockerfile by LLM. The validation process involves saving the proposed_dockerfile, building the image, and finally cleanning up
        Args:
            dockerfile_msg (ValidateDockerfileMessage): LLM message contains the definition of the Dockerfile
        Returns:
            str: a string indicates whether the Dockerfile has been built successfully
        """
        if self.repo_path is None:
            return self.handle_message_fallback()

        proposed_dockerfile_content = dockerfile_msg.proposed_dockerfile
        # It's better to have a different name other than the default name, good for comparison in the future between generated Dockerfile and existing Dockerfile
        proposed_dockerfile_name = "Dockerfile_proposed"
        img_tag = "validate_img"
        dockerfile_path = self._save_dockerfile(
            proposed_dockerfile_content, proposed_dockerfile_name
        )
        if dockerfile_path.startswith("An error"):
            return dockerfile_path

        img, build_log, build_time = self._build_docker_image(
            proposed_dockerfile_name, img_tag
        )

        if img:
            self._cleanup_dockerfile(img.id, dockerfile_path)
            return f"Docker image built successfully and build time took:{build_time} Seconds..."
        else:
            return f"Docker build failed with error message: {build_log}"
