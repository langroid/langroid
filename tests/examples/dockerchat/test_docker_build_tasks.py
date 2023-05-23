from llmagent.agent.base import AgentConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from examples.dockerchat.docker_chat_agent import DockerChatAgent
from examples.dockerchat.dockerchat_agent_messages import (
    AskURLMessage,
    ValidateDockerfileMessage,
)

from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from typing import Optional

import os
import tempfile


cfg = AgentConfig(
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
