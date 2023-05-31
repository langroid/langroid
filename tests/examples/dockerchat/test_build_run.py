from llmagent.agent.chat_agent import ChatAgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from examples.dockerchat.docker_chat_agent import DockerChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    AskURLMessage,
    ValidateDockerfileMessage,
    RunContainerMessage,
)

from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from typing import Optional

import os
import tempfile


cfg = ChatAgentConfig(
    debug=False,
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT3_5_TURBO,
        cache_config=RedisCacheConfig(fake=True),
    ),
)


ASK_URL_RESPONSE = """You have not yet sent me the URL. 
            Start by asking for the URL, then confirm the URL with me"""

GOT_URL_RESPONSE = """
Ok, confirming the URL. 
"""


class _TestDockerChatAgent(DockerChatAgent):
    def handle_message_fallback(self, input_str: str = "") -> Optional[str]:
        # if URL not yet known, tell LLM to ask for it, unless this msg
        # contains the word URL
        if self.repo_path is None and "url" not in input_str.lower():
            return ASK_URL_RESPONSE

    def ask_url(self, msg: AskURLMessage) -> str:
        self.repo_path = "dummy"
        return GOT_URL_RESPONSE

    def validate_dockerfile(self, msg: ValidateDockerfileMessage) -> str:
        return super().validate_dockerfile(msg, confirm=False)


PROPOSED_DOCKERFILE_CONTENT = """
    FROM python:3.9
    WORKDIR /app
    # Copy the requirements.txt file to the container
    COPY requirements.txt .
"""


def test_validate_dockerfile():
    agent = _TestDockerChatAgent(cfg)
    temp_folder_path = tempfile.mkdtemp()
    agent.repo_path = temp_folder_path

    try:
        vdm = ValidateDockerfileMessage()
        vdm.proposed_dockerfile = PROPOSED_DOCKERFILE_CONTENT
        result = agent.validate_dockerfile(vdm)

        # should fail because requirements.txt doesn'r exist
        assert "failed" in result

        # We shouldn't test this assetion if there is a problem with docker API
        if "server API" not in result:
            # Create a requirements.txt file in the folder
            temp_file_path = os.path.join(temp_folder_path, "requirements.txt")
            open(temp_file_path, "a").close()
            result = agent.validate_dockerfile(vdm)

            # should succeed after creating the file
            assert "successfully" in result

    finally:
        # Clean up - remove the temporary folder and its contents
        for filename in os.listdir(temp_folder_path):
            file_path = os.path.join(temp_folder_path, filename)
            os.remove(file_path)
        os.rmdir(temp_folder_path)


def test_run_container():
    agent = _TestDockerChatAgent(cfg)
    # create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        agent.repo_path = temp_dir
        agent.proposed_dockerfile = PROPOSED_DOCKERFILE_CONTENT
        # Create a requirements.txt file in the folder
        temp_file_path = os.path.join(temp_dir, "requirements.txt")
        open(temp_file_path, "a").close()

        # write a test case file
        test_case_filename = "test_case.py"
        with open(os.path.join(temp_dir, test_case_filename), "w") as f:
            f.write("import math\nprint(math.sqrt(16))")

        # create the RunContainerMessage object
        run_msg = RunContainerMessage()
        run_msg.cmd = "python"
        run_msg.tests = ["test_case.py"]

        # create the object and run the function with the custom Python 
        # image and the test cases
        run_results = agent.run_container(run_msg, False)

        if run_results:
            for value in run_results.values():
                # check that all test cases exited with code 0 (success)
                # assert all(result[1] == 0 for result in run_results)
                assert value.exit_code == 0

                # check that the logs contain the expected output
                assert "4.0" in value.output.decode("utf-8")
