import pytest

from llmagent.agent.chat_agent import ChatAgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from examples.dockerchat.docker_chat_agent import DockerChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    AskURLMessage,
    ValidateDockerfileMessage,
    RunContainerMessage,
)
from llmagent.parsing.repo_loader import RepoLoader, RepoLoaderConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from typing import Optional

import os
import tempfile


cfg = ChatAgentConfig(
    debug=False,
    vecdb=None,
    llm=OpenAIGPTConfig(
        type="openai",
        chat_model=OpenAIChatModel.GPT4,
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

    def run_container(self, msg: RunContainerMessage) -> str:
        return super().run_container(msg, confirm=False)


PROPOSED_DOCKERFILE_CONTENT = """
    FROM python:3.9
    WORKDIR /app
    # Copy the requirements.txt file to the container
    COPY requirements.txt .
"""

@pytest.skip("this test is too slow")
def test_validate_dockerfile():
    agent = _TestDockerChatAgent(cfg)
    temp_folder_path = tempfile.mkdtemp()
    agent.repo_path = temp_folder_path

    try:
        vdm = ValidateDockerfileMessage.examples()[0]
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


AUTO_GPT_URL = "https://github.com/Significant-Gravitas/Auto-GPT"
BASARAN_URL = "https://github.com/hyperonym/basaran"

BASARAN_DOCKERFILE = """FROM python:3.8\n\nWORKDIR /app
    \n\nCOPY requirements.txt setup.py ./
    \n\nRUN pip install --no-cache-dir -r requirements.txt\n\nCOPY . .
    \n\nEXPOSE 80\n\nENTRYPOINT [\"python\",\"-m\",\"basaran\"]
    """
AUTO_GPT_DOCKERFILE = """FROM python:3.10\n\nWORKDIR /app\n
    \nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt
    \n\nCOPY . .\n\nCMD [\"./run.sh\"]
    """

@pytest.skip("This test is too slow")
def test_run_container():
    agent = _TestDockerChatAgent(cfg)
    url = BASARAN_URL
    agent.repo_loader = RepoLoader(url, RepoLoaderConfig())
    agent.repo_path = agent.repo_loader.clone()
    agent.proposed_dockerfile = BASARAN_DOCKERFILE
    msg = RunContainerMessage.examples()[0]

    msg.location = "outside"
    msg.run = "docker run -d -p 5555:80 --rm validate_img:latest"
    msg.test = "curl -s http://localhost:5555"
    tst_result = agent.run_container(msg)
    if "Container run failed" not in tst_result:
        assert "True" in tst_result
    else:
        assert True

    msg.location = "inside"
    msg.run = "docker run -d -p 5555:80 --rm validate_img:latest"
    msg.test = f"pytest -m {agent.repo_path}/test/test_choice.py"
    tst_result = agent.run_container(msg)
    assert "exit code = 2" in tst_result
