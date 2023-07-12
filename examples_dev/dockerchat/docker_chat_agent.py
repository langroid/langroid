import logging
import time
from typing import Any, List, Optional

import docker
from docker.models.images import Image
from pydantic import BaseModel, HttpUrl
from rich.console import Console
from rich.prompt import Prompt

from examples_dev.codechat.code_chat_agent import CodeChatAgent, CodeChatAgentConfig
from examples_dev.dockerchat.build_run_utils import (
    _build_docker_image,
    _check_docker_daemon_url,
    _cleanup_dockerfile,
    _execute_command,
    _save_dockerfile,
)
from examples_dev.dockerchat.dockerchat_tool_messages import (
    AskURLMessage,
    FileExistsMessage,
    PythonDependencyMessage,
    PythonVersionMessage,
    RunContainerMessage,
    RunPythonMessage,
    ValidateDockerfileMessage,
)
from examples_dev.dockerchat.identify_python_dependency import (
    identify_dependency_management,
)
from examples_dev.dockerchat.identify_python_version import get_python_version
from langroid.agent.chat_agent import ChatAgent, ChatAgentConfig
from langroid.embedding_models.models import OpenAIEmbeddingsConfig
from langroid.language_models.base import LLMMessage
from langroid.language_models.openai_gpt import OpenAIChatModel, OpenAIGPTConfig
from langroid.parsing.parser import ParsingConfig
from langroid.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from langroid.prompts.prompts_config import PromptsConfig
from langroid.utils.constants import NO_ANSWER
from langroid.vector_store.base import VectorStoreConfig
from langroid.vector_store.qdrantdb import QdrantDBConfig

console = Console()
logger = logging.getLogger(__name__)
DEFAULT_URL = "https://github.com/hyperonym/basaran"
# Message types that can be handled by the agent;
# each corresponds to a method in the agent.


PLANNER_SYSTEM_MSG = """
You are a software developer and you want to create a dockerfile to containerize your 
code repository. However: 
(a) you are generally aware of docker, but you're not a docker expert, and
(b) you do not have direct access to the code repository.

"""

PLANNER_USER_MSG = f"""

To accomplish your task, you will be talking to 2 people: DockerExpert, who will manage 
the creation of the dockerfile; and Coder who has access to the code repository and 
will help you with  questions received from DockerExpert. When you are NOT using a 
function_call, any message you write should be formatted as:

"TO[<recipient>]: <message>", 

where <recipient> is either "DockerExpert" or "Coder", and <message> is the message 
you want to send.  
ALWAYS USE THIS FORMAT WHEN YOU ARE NOT USING A FUNCTION_CALL

The DockerExpert will be asking you questions about the 
the code repository in the form "INFO: <question>", or comments or other requests in 
the form "COMMENT: <comment>".  
For each such INFO question Q, you have to think step by step, and break it down into 
small steps, and use the Coder's help to answer these, since you do not 
have direct access to the code repository. Once you figure out the answer to an INFO 
question from DockerExpert, you should send a message to DockerExpert in the form 
"TO[DockerExpert]: <answer>", where <answer> is the answer you have figured out, 
or "{NO_ANSWER}" if you were unable to figure out answer. 
Do not try to make up an answer, if you are unable to 
figure it out from the Coder's responses or other information you have access to.
In particular, since you have no direct access to the repo, do not make up an answer 
unless it is supported by information sent to you by the Coder.
If at first the Coder is unable to answer your question, you can try asking a 
different way. For finding out certain types of information from the Coder, 
you have access to special TOOLS, as described below. When a TOOL is applicable, 
you should make your request in the precise JSON format described for the tool. If a 
tool is not applicable for the information you are seeking, you can make the request 
in  plain English.
  
Once you repond to an INFO question, you may get another INFO question, and so on. 
If the DockerExpert sends a COMMENT or other type of NON-INFO message, then you 
should directly respond to DockerExpert, without involving the Coder.

Remember to be concise in your responses to the DockerExpert, and in particular, 
DO NOT PROVIDE detailed contents of files to the DockerExpert; instead just refer to 
the file, and use those contents. 
"""

CODE_CHAT_INSTRUCTIONS = f"""
You have access to a code repository, and you will receive questions about it, 
to help me create a dockerfile for the repository. 
Along with the question, you may be given extracts from the code repo, and you can 
use those extracts to answer the question. If you cannot answer given the 
information, simply say "{NO_ANSWER}".
"""


class UrlModel(BaseModel):
    url: HttpUrl


class DockerChatAgentConfig(ChatAgentConfig):
    name: str = "DockerExpert"
    gpt4: bool = True
    debug: bool = False
    cache: bool = True
    stream: bool = True
    use_functions_api: bool = False
    use_tools: bool = True
    exclude_file_types: List[str] = ["Dockerfile"]  # file-types to exclude from repo

    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="langroid-dockerchat",
        storage_path=".qdrant/langroid-dockerchat/",
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )
    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
        temperature=0.2,
    )
    parsing: ParsingConfig = ParsingConfig(
        chunk_size=100,
    )

    prompts: PromptsConfig = PromptsConfig()


class DockerChatAgent(ChatAgent):
    url: str = ""
    repo_tree: str = None
    repo_path: str = None
    proposed_dockerfile: str = None
    docker_img: Image = None
    code_chat_agent: CodeChatAgent = None

    def __init__(
        self, config: DockerChatAgentConfig, task: Optional[List[LLMMessage]] = None
    ):
        super().__init__(config, task)
        self.config = config
        code_chat_cfg = CodeChatAgentConfig(
            name="Coder",
            repo_url="",  # this will be set later
            content_includes=[
                "txt",
                "md",
                "yml",
                "yaml",
                "sh",
                "Makefile",
                "py",
                "json",
            ],
            content_excludes=["Dockerfile"],
            # USE same LLM settings as DockerChatAgent, e.g.
            # if DockerChatAgent uses gpt4, then use gpt4 here too
            llm=self.config.llm,
        )
        self.code_chat_agent = CodeChatAgent(code_chat_cfg)

        planner_agent_cfg = ChatAgentConfig(
            name="Planner",
            use_tools=config.use_tools,
            use_functions_api=config.use_functions_api,
            vecdb=None,
            llm=self.config.llm,
        )
        self.planner_agent = ChatAgent(planner_agent_cfg)

    def handle_message_fallback(self, input_str: str = "") -> Optional[str]:
        if self.repo_path is None and "URL" not in input_str:
            return """
            You have not sent me the URL for the repo yet. 
            Please ask me for the URL, and once you receive it, 
            send it to me for confirmation. Once I confirm the URL, 
            you can proceed.
            """

    def run_python(self, msg: RunPythonMessage) -> str:
        # TODO: to be implemented. Return dummy msg for now
        logger.error(
            f"""
        This is a placeholder for the run_python method.
        Here is the code:
        {msg.code}
        """
        )
        return "No results, please continue asking your questions."

    @property
    def url(self):
        return self._url

    @url.setter
    def url(self, value):
        self._url = value
        # clone, chunk, ingest into vector-db of `code_chat_agent`
        self.code_chat_agent.ingest_url(self._url)
        loader_cfg = RepoLoaderConfig()
        for ftype in self.config.exclude_file_types:
            loader_cfg.file_types.remove(ftype)

        self.repo_loader = RepoLoader(
            self._url,
            loader_cfg,
        )
        self.repo_path = self.repo_loader.clone()
        # get the repo tree to depth d, with first k lines of each file
        self.repo_tree, _ = self.repo_loader.load(depth=1, lines=20)

    def ask_url(self, msg: AskURLMessage) -> str:
        while True:
            url = Prompt.ask(
                "[blue]Please enter the URL of the repo, or hit enter to use default"
            )
            if url == "":
                url = DEFAULT_URL
            try:
                url_model = UrlModel(url=url)
            except ValueError as e:
                Prompt.ask(f"[blue]A valid URL was not seen: {e}; Please try again")
            if url_model.url is not None:
                break

        self.url = url_model.url  # uses setter `url` above
        repo_files = [
            f
            for f in RepoLoader.list_files(
                self.repo_path, depth=1, exclude_types=self.config.exclude_file_types
            )
            if "docker" not in f.lower()
        ]
        repo_listing = "\n".join(repo_files)

        repo_listing_message = f"""
        Based on the URL, here is some information about the repo that you can use.  
        
        First, here is a list of ALL the files and directories at the ROOT of the repo: 
        {repo_listing}
        
        In later parts of the conversation, only ask questions that CANNOT 
        be answered by the information above. Do not ask for any info that is already 
        provided above! 
        """
        self.planner_agent.add_user_message(repo_listing_message)
        return repo_listing_message

    def python_version(self, m: PythonVersionMessage) -> str:
        """
        Identifies Python version for a given repo
        Args:
        Returns:
            str: a string indicates the identified python version or indicate the
                version can't be identified
        """
        if self.repo_path is None:
            return self.handle_message_fallback()

        answer = get_python_version(self.repo_path)
        if answer:
            return answer
        else:
            logger.error("Could not determine Python version.")
        return "Couldn't identify the python version"

    def file_exists(self, message: FileExistsMessage) -> str:
        if self.repo_path is None:
            return self.handle_message_fallback()
        # dummy result, fill with actual code.

        matches = RepoLoader.select(self.repo_tree, includes=[message.filename])
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
        Identifies Python dependencies in a given repo by inspecting various
            artifacts like requirements.txt
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

    def validate_dockerfile(
        self,
        dockerfile_msg: ValidateDockerfileMessage,
        confirm: bool = True,
    ) -> str:
        """
        validates the proposed Dockerfile by LLM. The validation process involves
        saving the proposed_dockerfile, building the image, and finally cleanning up
        Args:
            dockerfile_msg (ValidateDockerfileMessage): LLM message contains the
            definition of the Dockerfile
        Returns:
            str: a string indicates whether the Dockerfile has been built successfully
        """
        if self.repo_path is None:
            return self.handle_message_fallback()

        if type(dockerfile_msg.proposed_dockerfile) != str:
            dockerfile_msg.proposed_dockerfile = "\n".join(
                dockerfile_msg.proposed_dockerfile
            )
        if len(dockerfile_msg.proposed_dockerfile) < 20:
            return """
            The `proposed_dockerfile` parameter is invalid;
            Note this parameter should contain the CONTENTS of the 
            proposed dockerfile, NOT the NAME of the Dockerfile
            """

        if confirm:
            user_response = Prompt.ask(
                "Please confirm dockerfile validation",
                choices=["y", "n"],
                default="y",
            )
            if user_response.lower() != "y":
                return """"
                    Not ready for dockerfile validation, please 
                    continue with your next question or request for information.
                    """

        docker_daemon = _check_docker_daemon_url()
        if "exists" in docker_daemon:
            proposed_dockerfile_content = dockerfile_msg.proposed_dockerfile
            # It's better to have a different name other than the default name,
            # good for comparison in the future between
            # generated Dockerfile and existing Dockerfile
            proposed_dockerfile_name = "Dockerfile_proposed"
            img_tag = "validate_img"
            dockerfile_path = _save_dockerfile(
                self.repo_path, proposed_dockerfile_content, proposed_dockerfile_name
            )
            if dockerfile_path.startswith("An error"):
                return dockerfile_path

            img, build_log, build_time = _build_docker_image(
                self.repo_path, proposed_dockerfile_name, img_tag
            )

            if img:
                _cleanup_dockerfile(img.id, dockerfile_path)
                # For future use to run the container
                self.proposed_dockerfile = dockerfile_msg.proposed_dockerfile
                return f"""Docker image built successfully and build time took:
                {build_time} Seconds..."""
            else:
                return f"Docker build failed with error message: {build_log}"
        else:
            return docker_daemon

    def run_container(
        self,
        dockerrun_msg: RunContainerMessage,
        confirm: bool = True,
    ) -> Optional[Any]:
        """
        Runs a container based on the image built using the proposed_dockerfile.
        It then executes test cases inside the running container and reports
        the results.
        Args:
            dockerrun_msg (RunContainerMessage): LLM message contains the
            command and list of test cases
        Returns:
            A raw log and execution code indicate whether the test is
            executed successfully.
        """
        if confirm:
            user_response = Prompt.ask(
                "Please confirm dockerfile validation",
                choices=["y", "n"],
                default="y",
            )
            if user_response.lower() != "y":
                return """"
                    Not ready for dockerfile validation, please 
                    continue with your next question or request for information.
                    """

        docker_daemon = _check_docker_daemon_url()
        if "exists" in docker_daemon:
            client = docker.from_env()

            img_tag = "validate_img"
            img = self.docker_img
            # Save the Dockerfile and build the image
            if img is None:
                proposed_dockerfile_name = "Dockerfile_proposed"
                _ = _save_dockerfile(
                    self.repo_path, self.proposed_dockerfile, proposed_dockerfile_name
                )

                img, _, _ = _build_docker_image(
                    self.repo_path, proposed_dockerfile_name, img_tag
                )
                self.docker_img = img

            test_case = dockerrun_msg.test
            location = dockerrun_msg.location.lower()
            run = dockerrun_msg.run
            test_result = None
            container_id = None
            if img:
                try:
                    if location == "inside":
                        # We use tail to make sure the container keeps running
                        container = client.containers.run(
                            img.id, "tail -f /dev/null", detach=True, auto_remove=False
                        )
                        if container:
                            container_id = container.id
                            # TODO: I need to define some timeout here because
                            # noticed the execution of some commands takes forever
                            test_result = container.exec_run(f"{test_case}")
                            return f"""Test case executed from inside the container:
                            exit code = {test_result.exit_code} {test_result.output}
                            """
                        else:
                            return "Container run failed"

                    if location == "outside":
                        # TODO: we need converter from docker commands to docker SDK
                        cmd_result = _execute_command(run)
                        if cmd_result[0] is True and cmd_result[1]:
                            container_id = cmd_result[1].strip()
                            # delay to allow container finishing its setup
                            time.sleep(60)
                            test_result = _execute_command(test_case)
                            return f"""Test case executed from outside the 
                            container, execution code is: {test_result[0]}"""
                        else:
                            return f"Container run failed: {cmd_result[1]}"
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")

                finally:
                    if container_id:
                        container = client.containers.get(container_id)
                        container.remove(force=True)

            return "Image built failed"
        else:
            return docker_daemon
