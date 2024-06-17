import re
from typing import Optional, Tuple


def parse_addressed_message(
    content: str, addressing: str = "@"
) -> Tuple[Optional[str], str]:
    """In a message-string containing possibly multiple @<recipient> occurrences,
    find the last addressee and extract their name,
    and the message content following it.

    E.g. "thank you @bob, now I will ask @alice again. @alice, where is the mirror?" =>
    ("alice", "where is the mirror?")

    Args:
        content (str): The message content.
        addressing (str, optional): The addressing character. Defaults to "@".

    Returns:
        Tuple[Optional[str], str]:
        A tuple containing the last addressee and the subsequent message content.
    """
    # Regex to find all occurrences of the pattern
    pattern = re.compile(rf"{re.escape(addressing)}(\w+)[^\w]")
    matches = list(pattern.finditer(content))

    if not matches:
        return None, content  # No addressee found, return None and original content

    # Get the last match
    last_match = matches[-1]
    last_addressee = last_match.group(1)
    # Extract content after the last addressee
    content_after = content[last_match.end() :].strip()

    return last_addressee, content_after
