import re
from typing import Optional, Tuple


def parse_addressed_message(
    content: str, addressing: str = "@"
) -> Tuple[Optional[str], str]:
    # escape special characters in addressing prefix for regex use
    addressing_escaped = re.escape(addressing)
    pattern = rf"{addressing_escaped}(\w+)[,:\s]?"
    # Regular expression to find a username prefixed by addressing character or string
    match = re.findall(pattern, content)

    addressee = None
    if match:
        # select the last match as the addressee
        addressee = match[-1]

        # Remove the last occurrence of the addressing prefix followed by the
        # username and optional punctuation or whitespace
        # To remove only the last occurrence, we'll construct a new pattern that
        # specifically matches the last addressee
        last_occurrence_pattern = rf"{addressing_escaped}{addressee}[,:\\s]?"
        # Replace the last occurrence found in the content
        content = re.sub(last_occurrence_pattern, "", content, count=1).strip()

    return addressee, content
