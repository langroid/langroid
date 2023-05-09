from examples.dockerchat.docker_chat_agent import DockerChatAgent

import os
import tempfile
import subprocess

import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def mock_functions():
    with patch("os.path.exists") as mock_exists, patch(
        "os.remove"
    ) as mock_remove, patch("subprocess.run") as mock_run:
        mock_exists.return_value = True
        mock_run.return_value = MagicMock()
        yield mock_exists, mock_remove, mock_run


@pytest.mark.parametrize(
    "img_name, dockerfile_path, returncode",
    [
        ("test_image", "test_path", 0),
        ("another_image", "another_path", 1),
    ],
)
def test_remove_dockerfile_and_image(
    mock_functions, img_name, dockerfile_path, returncode
):
    mock_exists, mock_remove, mock_run = mock_functions

    # Set returncode on the subprocess.run mock
    mock_run.return_value.returncode = returncode

    # Call the function
    DockerChatAgent.cleanup_dockerfile(img_name, dockerfile_path)

    # Assert that os.remove and subprocess.run were called with the correct arguments
    mock_remove.assert_called_once_with(dockerfile_path)
    mock_run.assert_called_once_with(
        f"docker rmi -f {img_name}",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_save_dockerfile():
    content = """
    FROM python:3.8-slim

    WORKDIR /app

    COPY requirements.txt requirements.txt
    RUN pip install -r requirements.txt

    COPY . .

    CMD ["python", "app.py"]
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        dockerfile_path = DockerChatAgent.save_dockerfile(content, temp_dir)
        assert os.path.exists(dockerfile_path), "Dockerfile not saved"
        with open(dockerfile_path, "r") as f:
            saved_content = f.read()
        assert content == saved_content, "Dockerfile content mismatch"


def test_build_docker_image():
    content = """
    FROM python:3.8-slim

    WORKDIR /app

    COPY . .

    CMD ["python", "-c", "print('Hello, Docker!')"]
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        result = DockerChatAgent.build_docker_image(content, temp_dir)
        assert "Docker build was successful" == result, "Docker build failed"
