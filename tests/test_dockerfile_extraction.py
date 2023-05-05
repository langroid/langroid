from examples.dockerchat.docker_agent import response_contain_dockerfile


def test_response_contain_dockerfile() -> None:
    chatgpt_responses = [
        'Sample response 1: ```\nFROM centos:latest\n\nRUN yum -y update && \\\n    yum -y install python38 && \\\n    yum clean all\n\nCOPY . /prm\n\nWORKDIR /prm\n\nCMD ["/nobackup/python", "/nobackup/pca.py"]\n```',
        'Sample response 2: ```\n# hello world\nFROM centos:latest\n\nRUN yum -y update && \\\n    yum -y install python38 && \\\n    yum clean all\n\nCOPY . /prm\n\nWORKDIR /prm\n\nCMD ["/nobackup/python", "/nobackup/pca.py"]\n```',
        "Sample response 3: No dockerfile here",
    ]

    expected_outputs = [
        'FROM centos:latest\n\nRUN yum -y update && \\\n    yum -y install python38 && \\\n    yum clean all\n\nCOPY . /prm\n\nWORKDIR /prm\n\nCMD ["/nobackup/python", "/nobackup/pca.py"]',
        '# hello world\nFROM centos:latest\n\nRUN yum -y update && \\\n    yum -y install python38 && \\\n    yum clean all\n\nCOPY . /prm\n\nWORKDIR /prm\n\nCMD ["/nobackup/python", "/nobackup/pca.py"]',
        None,
    ]

    for i, input_string in enumerate(chatgpt_responses):
        assert (
            response_contain_dockerfile(input_string) == expected_outputs[i]
        ), f"Test Case {i+1} Failed"
