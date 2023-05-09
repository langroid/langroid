from examples.dockerchat.docker_chat_agent import DockerChatAgent

import os
import tempfile


def test_save_dockerfile():
    content = '''
    FROM python:3.8-slim

    WORKDIR /app

    COPY requirements.txt requirements.txt
    RUN pip install -r requirements.txt

    COPY . .

    CMD ["python", "app.py"]
    '''
    with tempfile.TemporaryDirectory() as temp_dir:
        dockerfile_path = DockerChatAgent.save_dockerfile(content, temp_dir)
        assert os.path.exists(dockerfile_path), "Dockerfile not saved"
        with open(dockerfile_path, "r") as f:
            saved_content = f.read()
        assert content == saved_content, "Dockerfile content mismatch"


def test_build_docker_image():
    content = '''
    FROM python:3.8-slim

    WORKDIR /app

    COPY . .

    CMD ["python", "-c", "print('Hello, Docker!')"]
    '''
    with tempfile.TemporaryDirectory() as temp_dir:
        result = DockerChatAgent.build_docker_image(content, temp_dir)
        assert "Docker build was successful" == result, "Docker build failed"
