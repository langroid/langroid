from langroid.pydantic_v1 import BaseModel


# Define the ANSI escape sequences for various colors and reset
class Colors(BaseModel):
    RED: str = "\033[31m"
    BLUE: str = "\033[34m"
    GREEN: str = "\033[32m"
    ORANGE: str = "\033[33m"  # no standard ANSI color for orange; using yellow
    CYAN: str = "\033[36m"
    MAGENTA: str = "\033[35m"
    YELLOW: str = "\033[33m"
    RESET: str = "\033[0m"


NO_ANSWER = "DO-NOT-KNOW"
DONE = "DONE"
USER_QUIT_STRINGS = ["q", "x", "quit", "exit", "bye", DONE]
PASS = "__PASS__"
PASS_TO = PASS + ":"
SEND_TO = "__SEND__:"
TOOL = "TOOL"
# This is a recommended setting for TaskConfig.addressing_prefix if using it at all;
# prefer to use `RecipientTool` to allow agents addressing others.
# Caution the AT string should NOT contain any 'word' characters, i.e.
# it no letters, digits or underscores.
# See tests/main/test_msg_routing for example usage
AT = "|@|"
TOOL_BEGIN = "TOOL_BEGIN"
TOOL_END = "TOOL_END"
