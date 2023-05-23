from typing import Optional, Tuple, List
import os
import json


def _identify_main_script(directory) -> List[str]:
    """
    Place holder to return the main scrip/s in the repo.
    This main scrip will be used to define the ENTRYPOINT and CMD. The main idea here this function will consult code-chat to identify main script candidate files
    Args:

    Returns:
        A list of main script candidate files
    """
    candidate_main_scripts = []
    # Iterate over all files in the given directory
    for filename in os.listdir(directory):
        # Check only Python files
        if filename.endswith(".py"):
            # Open the file and read its content
            with open(os.path.join(directory, filename), "r") as file:
                file_content = file.read()

                # If the file includes the main clause, return it as the main script
                if 'if __name__ == "__main__":' in file_content:
                    candidate_main_scripts.append(filename)

    # If no main script was found, return None
    return candidate_main_scripts


def _identify_cmd(main_scripts: List, cmd_args_only: bool = False) -> List[List[str]]:
    """
    Args:
        cmd_args_only (bool): when the flags for CMD and ENTRYPOINT are enabled. It's presummebly the user wants to use CMD as args to override default values of the arguemnts of ENTRYPOINT command
    Returns:
        list of string lists that represents possible instructions for CMD
    """
    # TODO I need to iterate over elements of main_scripts
    cmd = []
    if cmd_args_only:
        # In this case code-chat might be able to help to extract/identify arguments of the main_script
        cmd.append(["arg1", "arg2"])
    else:
        cmd.append(["python", main_scripts[0]])
    return cmd


def _identify_entrypoint(main_scripts: List, entrypoint_only_exe: bool = False) -> List[List[str]]:
    """
    Args:
        entrypoint_only_exe (bool): when the flags for CMD and ENTRYPOINT are enabled. It's presummebly the user wants to use CMD as args to override default values of the arguemnts of ENTRYPOINT command. So the ENTRYPOINT here defines only the executable.
    Returns:
        list of string lists that represents possible instructions for CMD
    """
    # TODO I need to iterate over elements of main_scripts
    entrypoints = []
    if entrypoint_only_exe:
        entrypoints.append(["python", main_scripts[0]])
    else:
        entrypoints.append(["python", main_scripts[0], "arg1", "arg2"])


def identify_entrypoint_CMD(
    directory: str, cmd: bool = False, entrypoint: bool = False
) -> Tuple[Optional[List[List[str]]], Optional[List[List[str]]]]:
    """
    This placeholder returns the commands for ENTRYPOINT and CMD based on the main script identifed in the repo.
    Args:
        cmd (bool): flag indicates the user wants to set the command for CMD
        entrypoint (bool): flag indicates the user wants to set the command for ENTRYPOINT
    Returns:
        str: tuple contains list of commands for ENTRYPOINT and CMD. List of string lists because there is a possibility that there are more than one entry point to the application (i.e., more than one main script), though it's not a good practice
    """
    main_scripts = None
    entrypoint_candidates = None
    cmd_candidates = None
    # No need to get main scripts if there is no interest in setting cmd and entrypoint
    if cmd or entrypoint:
        main_scripts = _identify_main_script(directory)

    if main_scripts:
        if cmd and entrypoint:
            # this means CMD provides arguments that are fed to the ENTRYPOINT. So the user wants the arguments of the ENTRYPOINT command to be overwritten. However, this looks advanced-level if the assumption here users of this tool aren't familar with Docker. But I'll leave it in the meantime.
            cmd_candidates = _identify_cmd(main_scripts, True)  # only args
            entrypoint_candidates = _identify_entrypoint(main_scripts, True)  # only executable 
        elif cmd:
            cmd_candidates = _identify_cmd(main_scripts)
        elif entrypoint:
            entrypoint_candidates = _identify_entrypoint(main_scripts)
        else:
            cmd_candidates = None
            # Fall back to a default,
            entrypoint_candidates = ["/bin/sh", "-c"]     
    else:
        # there is still a possibility the user wants to set the command, but at this stage it's difficult to infer that automatically and the user needs to elaborate
        return None, None

    cmd_entry_candidates = {
        "entrypoint": entrypoint_candidates,
        "cmd": cmd_candidates
    }
    return json.dumps(cmd_entry_candidates)

