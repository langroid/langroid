from typing import List

from langroid.mytypes import Document


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


def format_cited_references(citations: List[int], passages: list[Document]) -> str:
    """
    Given a list of (integer) citations, and a list of passages, return a string
    that can be added as a footer to the main text, to show sources cited.

    Args:
        citations (list[int]): list of citations, presumably from main text
        passages (list[Document]): list of passages (Document objects)

    Returns:
        str: formatted string of citations for footnote in markdown
    """
    citations_str = ""
    if len(citations) > 0:
        # append [i] source, content for each citation
        citations_str = "\n".join(
            [
                f"[^{c}] {passages[c-1].metadata.source}"
                f"\n{format_footnote_text(passages[c-1].content)}"
                for c in citations
            ]
        )
    return citations_str
