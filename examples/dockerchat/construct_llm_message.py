from llmagent.language_models.base import LLMMessage, Role


def construct_LLMMEssage(repo_metadata, port, entry_cmd, env_vars) -> LLMMessage:
    dep_stmt = "My project doesn't have any dependancy requirements,"
    if repo_metadata["requirements"]:
        dep_stmt = f"My project language is: {repo_metadata['language']},"

    port_stmt = " It doesn't expose any ports,"
    if port is not None:
        port_stmt = f" The application will be exposed via port {port},"

    env_vars_stmt = " I don't have any ENV variables."
    if env_vars is not None:
        env_vars_stmt = " Please define the following ENV variables:"
        for var in env_vars:
            env_vars_stmt += var + " "

    entry_cmd_stmt = f". The startup command is {entry_cmd}."
    content_msg = dep_stmt + port_stmt + env_vars_stmt + entry_cmd_stmt

    msg = LLMMessage(role=Role.SYSTEM, name="example_user", content=content_msg)

    return msg
