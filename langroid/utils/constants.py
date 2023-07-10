from pydantic import BaseModel


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


USER_QUIT = ["q", "x", "quit", "exit", "bye"]
NO_ANSWER = "DO-NOT-KNOW"
DONE = "DONE:"
