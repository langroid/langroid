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
    Improved chunker that:
      1. Splits by paragraph as the top boundary if possible.
      2. Otherwise, splits the paragraph by sentences (never mid-sentence).
      3. Accepts going over the upper bound if a single sentence is that large.
      4. Overlaps only once between consecutive chunks (the overlap
         from chunk1 is not duplicated again in chunk3).
      5. Preserves all original formatting (including \n\n).
      6. Avoids placing "PARA1 PARA2" or similar leftover tokens in the same chunk.
    """

    def count_words(text_block: str) -> int:
        return len(text_block.split())

    lower_bound = int(config.chunk_size * (1 - config.variation_percent))
    upper_bound = int(config.chunk_size * (1 + config.variation_percent))

    # Quick check: if the entire text is short enough, return as-is
    if count_words(text) <= upper_bound:
        return [text.strip()]

    # --- STEP 1: Split into paragraphs (with their trailing newlines if present).
    # We'll preserve the exact \n\n so that we maintain formatting in the final chunks.
    # A simple approach is to do split on '\n\n', then re-append '\n\n' to each piece.
    raw_paragraphs = text.split("\n\n")

    paragraphs = []
    for i, p in enumerate(raw_paragraphs):
        if p.strip():
            # Re-append the double-newline if it existed in the original text
            # (unless it's the very last paragraph without trailing \n\n).
            # This helps keep the formatting intact.
            if i < len(raw_paragraphs) - 1:
                paragraphs.append(p + "\n\n")
            else:
                paragraphs.append(p)

    # A helper to split a paragraph into sentences (and preserve spacing/periods).
    sentence_regex = r"(?<=[.!?])\s+"

    def split_paragraph_into_sentences(paragraph: str) -> List[str]:
        # If paragraph is short enough, return it as a single "sentence-chunk".
        if count_words(paragraph) <= upper_bound:
            return [paragraph]

        # Otherwise, do a sentence split:
        # keep the original spacing after each period by capturing the delimiter.
        sentences = re.split(sentence_regex, paragraph)
        # Re-insert the needed whitespace, if any was stripped by the split
        # so that each sentence retains punctuation. We'll handle the trailing
        # space by just storing them as separate items.
        # This might cause slight changes in whitespace, but you can refine if needed.
        sentences = [s.strip() for s in sentences if s.strip()]

        # Now each item in `sentences` is presumably 1 or more sentences if
        # original text had multiple punctuation marks.
        # We'll check each chunk; if it's bigger than upper_bound, we may do word-split.
        expanded_sentences = []
        for s in sentences:
            if count_words(s) > upper_bound:
                # As a last resort, do a word-based split:
                expanded_sentences.extend(_fallback_word_split(s, config))
            else:
                expanded_sentences.append(s)
        return expanded_sentences

    def _fallback_word_split(long_text: str, cfg: MarkdownChunkConfig) -> List[str]:
        """Split extremely large 'sentence' by words if we must."""
        words = long_text.split()
        pieces = []
        start = 0
        while start < len(words):
            # We'll take up to chunk_size words
            # (over-bound is allowed if there's a single chunk).
            end = start + cfg.chunk_size
            chunk_words = words[start:end]
            pieces.append(" ".join(chunk_words))
            start = end
        return pieces

    # We'll transform the paragraphs into a list of "segments":
    # either an entire paragraph (if short) or a list of sentence-sized strings.
    # That way we can unify the chunking logic: accumulate these segments until
    # we exceed the limit.
    segments = []
    for para in paragraphs:
        if count_words(para) > upper_bound:
            # Split by sentences
            segs = split_paragraph_into_sentences(para)
            segments.extend(segs)
        else:
            # The entire paragraph is one segment
            segments.append(para)

    # --- STEP 2: Accumulate segments into final chunks
    chunks = []
    current_chunk = ""
    current_count = 0

    def flush_chunk() -> None:
        """Flush the current chunk into chunks list (if not empty) and reset."""
        nonlocal current_chunk, current_count
        trimmed = current_chunk.rstrip()
        if trimmed:
            chunks.append(trimmed)
        current_chunk = ""
        current_count = 0

    def add_with_overlap(segment: str) -> None:
        """
        Attempt to add `segment` to current_chunk. If it exceeds the upper bound,
        and current_chunk >= lower_bound, flush the chunk. Then create a new chunk
        that begins with the overlap from the old chunk + segment.

        Note: Because we do NOT want to chop a sentence mid-way, if the segment
        alone is bigger than upper_bound, we accept it in one piece anyway
        (the user said "I’m okay with exceeding the upper bound for a single sentence").
        """
        nonlocal current_chunk, current_count

        seg_count = count_words(segment)
        # If adding this segment would exceed upper_bound,
        # and the current chunk is >= lower_bound, then we flush.
        # But if the current chunk is still < lower_bound, we keep going.
        if (current_count + seg_count > upper_bound) and (current_count >= lower_bound):
            # finalize chunk with overlap for next
            old_chunk = current_chunk
            flush_chunk()
            # Overlap: only take the last N tokens from old_chunk
            if old_chunk.strip():
                overlap_tokens_list = old_chunk.split()[-config.overlap_tokens :]
                overlap_str = " ".join(overlap_tokens_list)
                # Carefully preserve the exact trailing format from those tokens if
                # desired. You could do something more advanced here if you want
                # perfect whitespace preservation. For simplicity, we'll just re-join.
                current_chunk = overlap_str + " " + segment
            else:
                current_chunk = segment
            current_count = count_words(current_chunk)
        else:
            # Just accumulate
            if current_chunk:
                current_chunk += (
                    segment if segment.startswith("\n\n") else (" " + segment)
                )
            else:
                current_chunk = segment
            current_count = count_words(current_chunk)

    for seg in segments:
        seg_count = count_words(seg)
        # If this single segment alone exceeds the upper_bound,
        # but we said we are OK with big single segments, just add it fresh as a chunk.
        if seg_count > upper_bound:
            # If there's anything in current_chunk, flush it first
            flush_chunk()
            # This segment alone is a chunk
            chunks.append(seg.strip())
            continue

        # Otherwise, try to add it
        add_with_overlap(seg)

    # Flush any leftover
    flush_chunk()

    # --- STEP 3: Return non-empty chunks
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
        if node_content and node_content != node.path[-1].strip():
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
