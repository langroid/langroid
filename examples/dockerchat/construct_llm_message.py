from llmagent.language_models.base import LLMMessage, Role


def construct_LLMMEssage(repo_metadata, port, entry_cmd) -> LLMMessage:
    dep_file = None

    if repo_metadata['requirements']:
        dep_file = "requirements.txt"

    msg = LLMMessage(
        role=Role.SYSTEM,
        name="example_user",
        content=f"My my project language is: {repo_metadata['language']}, the depedencies are defined in the file {dep_file}. The application will be exposed via port {port} and the startup command is {entry_cmd}."
    )

    return msg
