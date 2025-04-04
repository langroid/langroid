import logging
from typing import List, Tuple

from langroid.mytypes import Document

logger = logging.getLogger(__name__)


def extract_markdown_references(md_string: str) -> List[int]:
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


def invalid_markdown_citations(md_string: str) -> List[str]:
    """
    Finds non-numeric markdown citations (e.g., [^a], [^xyz]) in a string.

    Args:
        md_string (str): The markdown string to search for invalid citations.

    Returns:
        List[str]: List of invalid citation strings (without brackets/caret).
    """
    import re

    # Find all citation references first
    matches = re.findall(r"\[\^([^\]\s]+)\]", md_string)

    # Filter out purely numeric citations
    invalid_citations = [match for match in matches if not match.isdigit()]

    return sorted(set(invalid_citations))


def format_footnote_text(content: str, width: int = 0) -> str:
    """
    Formats the content so that each original line is individually processed.
    - If width=0, no wrapping is done (lines remain as is).
    - If width>0, lines are wrapped to that width.
    - Blank lines remain blank (with indentation).
    - Everything is indented by 4 spaces (for markdown footnotes).

    Args:
        content (str): The text of the footnote to be formatted.
        width (int): Maximum width of the text lines. If 0, lines are not wrapped.

    Returns:
        str: Properly formatted markdown footnote text.
    """
    import textwrap

    indent = "    "  # 4 spaces for markdown footnotes
    lines = content.split("\n")  # keep original line structure

    output_lines = []
    for line in lines:
        # If the line is empty (or just spaces), keep it blank (but indented)
        if not line.strip():
            output_lines.append(indent)
            continue

        if width > 0:
            # Wrap each non-empty line to the specified width
            wrapped = textwrap.wrap(line, width=width)
            if not wrapped:
                # If textwrap gives nothing, add a blank (indented) line
                output_lines.append(indent)
            else:
                for subline in wrapped:
                    output_lines.append(indent + subline)
        else:
            # No wrapping: just indent the original line
            output_lines.append(indent + line)

    # Join them with newline so we preserve the paragraph/blank line structure
    return "\n".join(output_lines)


def format_cited_references(
    citations: List[int], passages: list[Document]
) -> Tuple[str, str]:
    """
    Given a list of (integer) citations, and a list of passages, return a string
    that can be added as a footer to the main text, to show sources cited.

    Args:
        citations (list[int]): list of citations, presumably from main text
        passages (list[Document]): list of passages (Document objects)

    Returns:
        str: formatted string of FULL citations (i.e. reference AND content)
            for footnote in markdown;
        str: formatted string of BRIEF citations (i.e. reference only)
            for footnote in markdown.
    """
    citations_str = ""
    full_citations_str = ""
    if len(citations) > 0:
        # append [i] source, content for each citation
        good_citations = [c for c in citations if c > 0 and c <= len(passages)]
        if len(good_citations) < len(citations):
            logger.warning(f"Invalid citations: {set(citations) - set(good_citations)}")

        # source and content for each citation
        full_citations_str = "\n".join(
            [
                f"[^{c}] {str(passages[c-1].metadata)}"
                f"\n{format_footnote_text(passages[c-1].content)}"
                for c in good_citations
            ]
        )

        # source for each citation
        citations_str = "\n".join(
            [f"[^{c}] {str(passages[c-1].metadata)}" for c in good_citations]
        )
    return full_citations_str, citations_str
