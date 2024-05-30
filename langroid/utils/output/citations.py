def extract_markdown_references(md_string: str) -> list[int]:
    """
    Extracts markdown references (e.g., [^1], [^2]) from a string and returns
    them as a sorted list of integers.

    Args:
        md_string (str): The markdown string containing references.

    Returns:
        list[int]: A sorted list of unique integers from the markdown references.
    """
    import re

    # Regex to find all occurrences of [^<number>]
    matches = re.findall(r"\[\^(\d+)\]", md_string)
    # Convert matches to integers, remove duplicates with set, and sort
    return sorted(set(int(match) for match in matches))


def format_footnote_text(content: str, width: int = 80) -> str:
    """
    Formats the content part of a footnote (i.e. not the first line that
    appears right after the reference [^4])
    It wraps the text so that no line is longer than the specified width and indents
    lines as necessary for markdown footnotes.

    Args:
        content (str): The text of the footnote to be formatted.
        width (int): Maximum width of the text lines.

    Returns:
        str: Properly formatted markdown footnote text.
    """
    import textwrap

    # Wrap the text to the specified width
    wrapped_lines = textwrap.wrap(content, width)
    if len(wrapped_lines) == 0:
        return ""
    indent = "    "  # Indentation for markdown footnotes
    return indent + ("\n" + indent).join(wrapped_lines)
