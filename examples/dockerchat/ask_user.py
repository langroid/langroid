from rich import print


def get_entry_startup_cmd():
    print("\n[blue]Please specify the file and command for running your app: ", end="")
    entry_cmd = input("")
    return entry_cmd


def get_env_vars():
    env_vars = []
    print("\n[blue]Please specify ENV variables for running your app: ", end="")
    query = input("")
    return env_vars


def get_expose_port():
    print("\n[blue]Please specify port number to expose your app, otherwise type No: ", end="")
    query = input("")
    port = None
    if query not in ["No"]:
        port = query
    return port
