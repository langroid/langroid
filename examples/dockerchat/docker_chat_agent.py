from llmagent.agent.chat_agent import ChatAgent, ChatAgentConfig
from pydantic import BaseModel, HttpUrl
from typing import Optional, List, Any
from examples.codechat.code_chat_agent import CodeChatAgentConfig, CodeChatAgent
from examples.codechat.code_chat_tools import (
    ShowFileContentsMessage,
    ShowDirContentsMessage,
)
from examples.dockerchat.dockerchat_agent_messages import (
    RunPythonMessage,
    AskURLMessage,
    FileExistsMessage,
    PythonVersionMessage,
    PythonDependencyMessage,
    ValidateDockerfileMessage,
    EntryPointAndCMDMessage,
    RunContainerMessage,
)
from rich.console import Console
from rich.prompt import Prompt
from llmagent.language_models.base import LLMMessage
from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from examples.dockerchat.identify_python_version import get_python_version
from examples.dockerchat.identify_python_dependency import (
    identify_dependency_management,
)

from examples.dockerchat.build_run_utils import (
    _build_docker_image,
    _cleanup_dockerfile,
    _save_dockerfile,
    _execute_command,
)

import logging
import docker

console = Console()
logger = logging.getLogger(__name__)
DEFAULT_URL = "https://github.com/eugeneyan/testing-ml"
# Message types that can be handled by the agent;
# each corresponds to a method in the agent.

NO_ANSWER = "I don't know"
NONE_ANSWER = "NONE"

PLANNER_INSTRUCTIONS = """
You are a software developer and you want to create a dockerfile to container your 
code repository. However: 
(a) you are generally aware of docker, but you're not a docker expert, and
(b) you do not have direct access to the code repository.
You will be receiving questions from a docker expert about the code repository.
For each MAIN question Q, you have to think step by step, and break it down into 
small steps. For each step (since you cannot access the code repo) you have to ask me 
a question, and I will try to answer. If I cannot, I may say "I don't know" or "NONE", 
in that case, DO NOT MAKE UP AN ANSWER! Instead, you can try asking differently or 
break it down into even smaller steps.  
Only when you are SURE you have the answer to the MAIN question Q, simply say 
"DONE: <whatever the answer is>". Then you may get another MAIN question Q, and so on.
If you are not able to answer the MAIN question Q, simply say "I don't know", 
and DO NOT MAKE UP AN ANSWER!
Your only messages should be 
(a) question for me, (b) DONE: <answer>, or (c) I don't know.
Do you say anything else.
"""

CODE_CHAT_INSTRUCTIONS = """
You have access to a code repository, and you will receive questions about it, 
to help me create a dockerfile for the repository. 
Along with the question, you may be given extracts from the code repo, and you can 
use those extracts to answer the question. If you cannot answer given the 
information, simply say "I don't know", or say "NONE", whichever you prefer.
For some questions, you may be able to use TOOLs to answer them; if there are tools 
available, you will be told what they are, when to use them, and what format to 
request the TOOL.
"""


class UrlModel(BaseModel):
    url: HttpUrl


class DockerChatAgent(ChatAgent):
    url: str = ""
    repo_tree: str = None
    repo_path: str = None
    proposed_dockerfile: str = None
    code_chat_agent: CodeChatAgent = None

    def __init__(
        self, config: ChatAgentConfig, task: Optional[List[LLMMessage]] = None
    ):
        super().__init__(config, task)
        code_chat_cfg = CodeChatAgentConfig(
            name="Coder",
            repo_url="",  # this will be set later
            system_message=CODE_CHAT_INSTRUCTIONS,
            content_includes=["txt", "md", "yml", "yaml", "sh", "Makefile"],
            content_excludes=["Dockerfile"],
            # USE same LLM settings as DockerChatAgent, e.g.
            # if DockerChatAgent uses gpt4, then use gpt4 here too
            llm=self.config.llm,
        )
        self.code_chat_agent = CodeChatAgent(code_chat_cfg)
        self.code_chat_agent.enable_message(ShowDirContentsMessage)
        self.code_chat_agent.enable_message(ShowFileContentsMessage)
        self.code_chat_agent.enable_message(RunPythonMessage)

        planner_agent_cfg = ChatAgentConfig(
            name="Planner",
            system_message=PLANNER_INSTRUCTIONS,
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
        # note this uses the setter of `code_chat_agent`, triggers vector-db
        # creation, repo download, chunking, ingest into vector-db
        self.code_chat_agent.repo_url = self._url

        self.repo_loader = RepoLoader(self._url, RepoLoaderConfig())
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
        repo_listing = "\n".join(self.repo_loader.ls(self.repo_tree, depth=1))

        return f"""
        Based on the URL, here is some information about the repo that you can use.  
        
        First, here is a list of ALL the files and directories at the ROOT of the 
        repo. Any files of interest to you MUST be in this list, therefore you do NOT 
        need to ask in future about whether any file exists.
        {repo_listing}
        In later parts of the conversation, only ask questions that CANNOT 
        be answered by the information above. Do not ask for any info that is already 
        provided above! 
        """

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
        answer = self.ask_agent(
            self.code_chat_agent,
            request="What is the Python version of this repo?",
            no_answer=NO_ANSWER,
            user_confirm=False,
        )
        if answer is not None:
            return answer

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

        answer = self.ask_agent(
            self.code_chat_agent,
            request=f"Does this project contain a file named {message.filename}?",
            no_answer=NO_ANSWER,
            user_confirm=False,
        )
        if answer is not None:
            return answer

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

        answer = self.ask_agent(
            self.code_chat_agent,
            request="Which file is used to manage dependencies in this project?",
            no_answer=NO_ANSWER,
            user_confirm=False,
        )
        if answer is not None:
            return answer

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
            return f"Docker image built successfully and build time took:{build_time} Seconds..."
        else:
            return f"Docker build failed with error message: {build_log}"

    def find_entrypoint(self, m: EntryPointAndCMDMessage) -> str:
        """
        Finds corresponding command to the ENTRYPOINT
        Args:
            m (EntryPointAndCMDMessage): LLM message contains a request to identify
                entrypoints
        Retruns:
            str: description of the main scripts and corresponding argument in the
                repo that are potential candidates to become ENTRYPOINT
        """
        if self.repo_path is None:
            return self.handle_message_fallback()

        answer = self.ask_agent(
            self.code_chat_agent,
            request="""What's the name of main script in this repo and can you SPECIFY 
            the command line and necessary arguments to run the main script? 
            If there are more than one main script, then SPECIFY the commands 
            and necessary arguments corresponding to each one
            """,
            no_answer=NO_ANSWER,
            user_confirm=False,
        )
        if answer is not None:
            return answer

        return "I couldn't identify potentail main scripts for the ENTRYPOINT"

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

        client = docker.from_env()

        # Save the Dockerfile and build the image
        img_tag = "validate_img"
        proposed_dockerfile_name = "Dockerfile_proposed"
        _ = _save_dockerfile(
            self.repo_path, self.proposed_dockerfile, proposed_dockerfile_name
        )

        img, _, _ = _build_docker_image(
            self.repo_path, proposed_dockerfile_name, img_tag
        )

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
                        return f"Test case executed from inside the container: exit code = {test_result.exit_code} {test_result.output}"
                    else:
                        return "Container run failed"

                if location == "outside":
                    # TODO: we need converter from docker commands to docker SDK
                    cmd_result = _execute_command(run)
                    if cmd_result[0] is True:
                        container_id = cmd_result[1]
                        test_result = _execute_command(test_case)
                        return f"Test case executed from outsied the container, execution code is: {test_result[0]}"
                    else:
                        return f"Container run failed: {cmd_result[1]}"
            except Exception as e:
                logger.error(f"An error occurred: {str(e)}")

            finally:
                if container_id:
                    container = client.containers.get(container_id)
                    container.remove(force=True)

        return "Image built failed"
