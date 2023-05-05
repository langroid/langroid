import re


def response_contain_dockerfile(response_message):
    """
    Identify if Dockerfile snippet exists inside LLM response
    Args:
        response_message: received LLM response
    Returns:
    """
    dockerfile = None
    pattern = r"```([\s\S]*?FROM[\s\S]*?)```"
    dockerfile_snippet = re.search(pattern, response_message, re.DOTALL)
    if dockerfile_snippet:
        dockerfile = dockerfile_snippet.group(1).strip()

    return dockerfile
