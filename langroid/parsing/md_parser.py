import re
from typing import List

from langroid.pydantic_v1 import BaseModel, Field

HEADER_CONTEXT_SEP = "\n...\n"


# Pydantic model definition for a node in the markdown hierarchy
class Node(BaseModel):
    content: str  # The text of the header or content block
    path: List[str]  # List of header texts from root to this node
    children: List["Node"] = Field(default_factory=list)
    # Nested children nodes

    def __repr__(self) -> str:
        # for debug printing
        return (
            f"Node(content={self.content!r}, path={self.path!r}, "
            f"children={len(self.children)})"
        )

    # Pydantic v1 requires forward references for self-referencing models
    # Forward references will be resolved with the update_forward_refs call below.


# Resolve forward references for Node (required for recursive models in Pydantic v1)
Node.update_forward_refs()


def _cleanup_text(text: str) -> str:
    # 1) Convert alternative newline representations (any CRLF or CR) to a single '\n'
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2) Replace 3 or more consecutive newlines with exactly 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown_headings(md_text: str) -> List[Node]:
    """
    Parse `md_text` to extract a heading-based hierarchy, skipping lines
    that look like headings inside fenced code blocks. Each heading node
    will have a child node for the text that appears between this heading
    and the next heading.

    Returns a list of top-level Node objects.

    Example structure:
        Node(content='# Chapter 1', path=['# Chapter 1'], children=[
            Node(content='Intro paragraph...', path=['# Chapter 1'], children=[]),
            Node(content='## Section 1.1', path=['# Chapter 1', '## Section 1.1'],
                 children=[
                  Node(content='Some text in Section 1.1.', path=[...], children=[])
            ]),
            ...
        ])
    """
    # If doc is empty or only whitespace, return []
    if not md_text.strip():
        return []

    lines = md_text.splitlines(True)  # keep the newline characters

    # We'll scan line-by-line, track code-fence status, collect headings
    headings = []  # list of (level, heading_line, start_line_idx)
    in_code_fence = False
    fence_marker = None  # track which triple-backtick or ~~~ opened

    for i, line in enumerate(lines):
        # Check if we're toggling in/out of a fenced code block
        # Typically triple backtick or triple tilde: ``` or ~~~
        # We do a *loose* check: a line that starts with at least 3 backticks or tildes
        # ignoring trailing text. You can refine as needed.
        fence_match = re.match(r"^(```+|~~~+)", line.strip())
        if fence_match:
            # If we are not in a fence, we enter one;
            # If we are in a fence, we exit if the marker matches
            marker = fence_match.group(1)  # e.g. "```" or "~~~~"
            if not in_code_fence:
                in_code_fence = True
                fence_marker = marker[:3]  # store triple backtick or triple tilde
            else:
                # only close if the fence_marker matches
                # E.g. if we opened with ```, we close only on ```
                if fence_marker and marker.startswith(fence_marker):
                    in_code_fence = False
                    fence_marker = None

        if not in_code_fence:
            # Check if the line is a heading
            m = HEADING_RE.match(line)
            if m:
                hashes = m.group(1)  # e.g. "##"
                heading_text = line.rstrip("\n")  # entire line, exact
                level = len(hashes)
                headings.append((level, heading_text, i))

    # If no headings found, return a single root node with the entire text
    if not headings:
        return [Node(content=md_text.strip(), path=[], children=[])]

    # Add a sentinel heading at the end-of-file, so we can slice the last block
    # after the final real heading. We'll use level=0 so it doesn't form a real node.
    headings.append((0, "", len(lines)))

    # Now we build "heading blocks" with
    # (level, heading_text, start_line, end_line, content)
    heading_blocks = []
    for idx in range(len(headings) - 1):
        level, heading_line, start_i = headings[idx]
        next_level, _, next_start_i = headings[idx + 1]

        # Content is everything after the heading line until the next heading
        # i.e. lines[start_i+1 : next_start_i]
        block_content_lines = lines[start_i + 1 : next_start_i]
        block_content = "".join(block_content_lines).rstrip("\n")

        heading_blocks.append(
            {"level": level, "heading_text": heading_line, "content": block_content}
        )
    # (We skip the sentinel heading in the final result.)

    # We'll now convert heading_blocks into a tree using a stack-based approach
    root_nodes: List[Node] = []
    stack: List[Node] = []
    header_path: List[str] = []

    for hb in heading_blocks:
        level = hb["level"]  # type: ignore
        heading_txt = hb["heading_text"]
        content_txt = hb["content"]

        # --- Pop stack first! ---
        while stack and len(stack[-1].path) >= level:
            stack.pop()
            header_path.pop()

        # build new path, create a node for the heading
        new_path = header_path + [heading_txt]
        heading_node = Node(
            content=heading_txt, path=new_path, children=[]  # type: ignore
        )

        # Possibly create a content child for whatever lines were below the heading
        if content_txt.strip():  # type: ignore
            content_node = Node(
                content=content_txt, path=new_path, children=[]  # type: ignore
            )
            heading_node.children.append(content_node)

        # Attach heading_node to the stack top or as a root
        if stack:
            stack[-1].children.append(heading_node)
        else:
            root_nodes.append(heading_node)

        stack.append(heading_node)
        header_path.append(heading_txt)  # type: ignore

    return root_nodes


# The Chunk model for the final enriched chunks.
class Chunk(BaseModel):
    text: str  # The chunk text (which includes header context)
    path: List[str]  # The header path (list of header strings)
    token_count: int


# Configuration for chunking
class MarkdownChunkConfig(BaseModel):
    chunk_size: int = 200  # desired chunk size in tokens
    overlap_tokens: int = 30  # number of tokens to overlap between chunks
    variation_percent: float = 0.3  # allowed variation
    rollup: bool = True  # whether to roll up chunks
    header_context_sep: str = HEADER_CONTEXT_SEP  # separator for header context


# A simple tokenizer that counts tokens as whitespace-separated words.
def count_words(text: str) -> int:
    return len(text.split())


def recursive_chunk(text: str, config: MarkdownChunkConfig) -> List[str]:
    """
    Enhanced chunker that:
      1. Splits by paragraph (top-level).
      2. Splits paragraphs by sentences if needed (never mid-sentence unless huge).
      3. Allows going over the upper bound rather than splitting a single sentence.
      4. Overlaps only once between consecutive chunks.
      5. Looks ahead to avoid a "dangling" final chunk below the lower bound.
      6. Preserves \n\n (and other original spacing) as best as possible.
    """

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def count_words(text_block: str) -> int:
        return len(text_block.split())

    lower_bound = int(config.chunk_size * (1 - config.variation_percent))
    upper_bound = int(config.chunk_size * (1 + config.variation_percent))

    # Quick check: if the entire text is short enough, return as-is.
    if count_words(text) <= upper_bound:
        return [text.strip()]

    # Split into paragraphs, preserving \n\n if it's there.
    raw_paragraphs = text.split("\n\n")
    paragraphs = []
    for i, p in enumerate(raw_paragraphs):
        if p.strip():
            # Re-append the double-newline if not the last piece
            if i < len(raw_paragraphs) - 1:
                paragraphs.append(p + "\n\n")
            else:
                paragraphs.append(p)

    # Split paragraphs into "segments": each segment is either
    # a full short paragraph or (if too big) a list of sentences.
    sentence_regex = r"(?<=[.!?])\s+"

    def split_paragraph_into_sentences(paragraph: str) -> List[str]:
        """
        Return a list of sentence-sized segments. If a single sentence
        is bigger than upper_bound, do a word-level fallback.
        """
        if count_words(paragraph) <= upper_bound:
            return [paragraph]

        sentences = re.split(sentence_regex, paragraph)
        # Clean up stray whitespace
        sentences = [s.strip() for s in sentences if s.strip()]

        expanded = []
        for s in sentences:
            if count_words(s) > upper_bound:
                expanded.extend(_fallback_word_split(s, config))
            else:
                expanded.append(s)
        return expanded

    def _fallback_word_split(long_text: str, cfg: MarkdownChunkConfig) -> List[str]:
        """
        As a last resort, split extremely large 'sentence' by words.
        """
        words = long_text.split()
        pieces = []
        start = 0
        while start < len(words):
            end = start + cfg.chunk_size
            chunk_words = words[start:end]
            pieces.append(" ".join(chunk_words))
            start = end
        return pieces

    # Build a list of segments
    segments = []
    for para in paragraphs:
        if count_words(para) > upper_bound:
            # split into sentences
            segs = split_paragraph_into_sentences(para)
            segments.extend(segs)
        else:
            segments.append(para)

    # -------------------------------------------------
    # Accumulate segments into final chunks
    # -------------------------------------------------
    chunks = []
    current_chunk = ""
    current_count = 0

    def flush_chunk() -> None:
        nonlocal current_chunk, current_count
        trimmed = current_chunk.strip()
        if trimmed:
            chunks.append(trimmed)
        current_chunk = ""
        current_count = 0

    def remaining_tokens_in_future(all_segments: List[str], current_index: int) -> int:
        """Sum of word counts from current_index onward."""
        return sum(count_words(s) for s in all_segments[current_index:])

    for i, seg in enumerate(segments):
        seg_count = count_words(seg)

        # If this single segment alone exceeds upper_bound, we accept it as a big chunk.
        if seg_count > upper_bound:
            # If we have something in the current chunk, flush it first
            flush_chunk()
            # Then store this large segment as its own chunk
            chunks.append(seg.strip())
            continue

        # Attempt to add seg to the current chunk
        if (current_count + seg_count) > upper_bound and (current_count >= lower_bound):
            # We would normally flush here, but let's see if we are nearing the end:
            # If the remaining tokens (including this one) is < lower_bound,
            # we just add it anyway to avoid creating a tiny final chunk.
            future_tokens = remaining_tokens_in_future(segments, i)
            if future_tokens < lower_bound:
                # Just add it (allowing to exceed upper bound)
                if current_chunk:
                    # Add space or preserve newline carefully
                    # We'll do a basic approach here:
                    if seg.startswith("\n\n"):
                        current_chunk += seg  # preserve double new line
                    else:
                        current_chunk += " " + seg
                    current_count = count_words(current_chunk)
                else:
                    current_chunk = seg
                    current_count = seg_count
            else:
                # Normal flush
                old_chunk = current_chunk
                flush_chunk()
                # Overlap from old_chunk
                overlap_tokens_list = (
                    old_chunk.split()[-config.overlap_tokens :] if old_chunk else []
                )
                overlap_str = (
                    " ".join(overlap_tokens_list) if overlap_tokens_list else ""
                )
                if overlap_str:
                    current_chunk = overlap_str + " " + seg
                else:
                    current_chunk = seg
                current_count = count_words(current_chunk)
        else:
            # Just accumulate
            if current_chunk:
                if seg.startswith("\n\n"):
                    current_chunk += seg
                else:
                    current_chunk += " " + seg
            else:
                current_chunk = seg
            current_count = count_words(current_chunk)

    # Flush leftover
    flush_chunk()

    # Return non-empty
    return [c for c in chunks if c.strip()]


# Function to process a Node and produce enriched chunks.
def chunk_node(node: Node, config: MarkdownChunkConfig) -> List[Chunk]:
    chunks: List[Chunk] = []

    # Check if this is a header-only node.
    is_header_only = node.path and node.content.strip() == node.path[-1]

    # Only generate a chunk for the node if it has non-header content,
    # or if it’s header-only AND has no children (i.e., it's a leaf header).
    if node.content.strip() and (not is_header_only or not node.children):
        header_prefix = (
            config.header_context_sep.join(node.path) + "\n\n" if node.path else ""
        )
        content_chunks = recursive_chunk(node.content, config)
        for chunk_text in content_chunks:
            full_text = header_prefix + chunk_text
            chunks.append(
                Chunk(
                    text=full_text, path=node.path, token_count=count_words(full_text)
                )
            )

    # Process children nodes recursively.
    for child in node.children:
        child_chunks = chunk_node(child, config)
        chunks.extend(child_chunks)

    return chunks


# Function to process an entire tree of Nodes.
def chunk_tree(root_nodes: List[Node], config: MarkdownChunkConfig) -> List[Chunk]:
    all_chunks: List[Chunk] = []
    for node in root_nodes:
        all_chunks.extend(chunk_node(node, config))
    return all_chunks


def aggregate_content(node: Node) -> str:
    """
    Recursively aggregate the content from a node and all its descendants,
    excluding header-only nodes to avoid duplication.
    """
    parts = []

    # Skip header-only nodes in content aggregation
    is_header_only = node.path and node.content.strip() == node.path[-1].strip()
    if not is_header_only and node.content.strip():
        parts.append(node.content.strip())

    # Recurse on children
    for child in node.children:
        child_text = aggregate_content(child)
        if child_text.strip():
            parts.append(child_text.strip())

    return "\n\n".join(parts)


def flatten_tree(node: Node, level: int = 0) -> str:
    """
    Flatten a node and its children back into proper markdown text.

    Args:
        node: The node to flatten
        level: The current heading level (depth in the tree)

    Returns:
        str: Properly formatted markdown text
    """
    result = ""

    # Check if this is a header node (content matches last item in path)
    is_header = node.path and node.content.strip().startswith("#")

    # For header nodes, don't duplicate the hash marks
    if is_header:
        result = node.content.strip() + "\n\n"
    elif node.content.strip():
        result = node.content.strip() + "\n\n"

    # Process all children
    for child in node.children:
        result += flatten_tree(child, level + 1)

    return result


def rollup_chunk_node(
    node: Node, config: MarkdownChunkConfig, prefix: str = ""
) -> List[Chunk]:
    """
    Recursively produce rollup chunks from `node`, passing down a `prefix`
    (e.g., parent heading(s)).

    - If a node is heading-only (content == last path item) and has children,
      we skip creating a chunk for that node alone and instead add that heading
      to the `prefix` for child nodes.
    - If a node is NOT heading-only OR has no children, we try to fit all of its
      flattened content into a single chunk. If it's too large, we chunk it.
    - We pass the (possibly updated) prefix down to children, so each child's
      chunk is enriched exactly once with all ancestor headings.
    """

    chunks: List[Chunk] = []

    # Check if the node is "heading-only" and has children
    # e.g. node.content=="# Chapter 1" and node.path[-1]=="# Chapter 1"
    is_heading_only_with_children = (
        node.path
        and node.content.strip() == node.path[-1].strip()
        and len(node.children) > 0
    )

    if is_heading_only_with_children:
        # We do NOT create a chunk for this node alone.
        # Instead, we add its heading to the prefix for child chunks.
        new_prefix = prefix + node.content.strip()
        for i, child in enumerate(node.children):
            sep = "\n\n" if i == 0 else config.header_context_sep
            chunks.extend(rollup_chunk_node(child, config, prefix=new_prefix + sep))
        return chunks

    # If not heading-only-with-children, we handle this node's own content:
    # Flatten the entire node (including sub-children) in standard Markdown form.
    flattened = flatten_tree(node, level=len(node.path))
    flattened_with_prefix = prefix + flattened
    total_tokens = count_words(flattened_with_prefix)

    # Check if we can roll up everything (node + children) in a single chunk
    if total_tokens <= config.chunk_size * (1 + config.variation_percent):
        # One single chunk for the entire subtree
        chunks.append(
            Chunk(text=flattened_with_prefix, path=node.path, token_count=total_tokens)
        )
    else:
        # It's too large overall. We'll chunk the node's own content first (if any),
        # then recurse on children.
        node_content = node.content.strip()

        # If we have actual content that is not just a heading, chunk it with the prefix
        # (like "preamble" text).
        # Note: if this node is heading-only but has NO children,
        # it will still land here
        # (because is_heading_only_with_children was False due to zero children).
        if node_content and (not node.path or node_content != node.path[-1].strip()):
            # The node is actual content (not purely heading).
            # We'll chunk it in paragraphs/sentences with the prefix.
            content_chunks = recursive_chunk(node_content, config)
            for text_block in content_chunks:
                block_with_prefix = prefix + text_block
                chunks.append(
                    Chunk(
                        text=block_with_prefix,
                        path=node.path,
                        token_count=count_words(block_with_prefix),
                    )
                )

        # Now recurse on children, passing the same prefix so they get it too
        for child in node.children:
            chunks.extend(rollup_chunk_node(child, config, prefix=prefix))

    return chunks


def rollup_chunk_tree(
    root_nodes: List[Node],
    config: MarkdownChunkConfig,
) -> List[Chunk]:
    # Create a dummy root node that contains everything.
    dummy_root = Node(content="", path=[], children=root_nodes)

    # Now process just the dummy root node with an empty prefix.
    chunks = rollup_chunk_node(dummy_root, config, prefix="")
    return chunks


def chunk_markdown(markdown_text: str, config: MarkdownChunkConfig) -> List[str]:
    tree = parse_markdown_headings(markdown_text)
    if len(tree) == 1 and len(tree[0].children) == 0:
        # Pure text, no hierarchy, so just use recursive_chunk
        text_chunks = recursive_chunk(markdown_text, config)
        return [_cleanup_text(chunk) for chunk in text_chunks]
    if config.rollup:
        chunks = rollup_chunk_tree(tree, config)
    else:
        chunks = chunk_tree(tree, config)
    return [_cleanup_text(chunk.text) for chunk in chunks]


if __name__ == "__main__":
    # Example usage:
    markdown_text = """# Title
Intro para. Hope this is not
getting split.
## SubTitle
- Item1
- Item2
"""
    # Set up chunking config with very large chunk size.
    # (you can adjust chunk_size, overlap_tokens, variation_percent)
    config = MarkdownChunkConfig(
        chunk_size=200, overlap_tokens=5, variation_percent=0.2
    )
    chunks = chunk_markdown(markdown_text, config)

    for idx, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {idx} --- ")
        print(chunk)
        print()

    config.rollup = True
    # with rollup_chunk_tree we get entire doc as 1 chunk
    chunks = chunk_markdown(markdown_text, config)
    assert len(chunks) == 1
    for idx, chunk in enumerate(chunks, 1):
        print(f"--- Chunk {idx} ---")
        print(chunk)
        print()
