import re
from dataclasses import dataclass
from typing import List

import pytest

from langroid.parsing.md_parser import (
    MarkdownChunkConfig,
    Node,
    chunk_markdown,
    count_words,
    parse_markdown_headings,
    recursive_chunk,
)


@dataclass
class SectionData:
    header: str
    content: str

    def to_markdown(self) -> str:
        return f"{self.header}\n{self.content}\n\n"


CH1_DATA = SectionData(
    header="# Chapter 1",
    content="""Intro paragraph under Chapter 1.
This is a somewhat longer paragraph that might require splitting
if token limits are low.
```java
# Fake Chapter in Code Block - just a comment!
This is not a real chapter.
## Comment in code!
```
""",
)

SEC1_1_DATA = SectionData(
    header="## Section 1.1",
    content="""Some text in Section 1.1. 
It might include multiple sentences. Here is another sentence.
```python
# Throw in some comments just to mix things up.
def some_function():
    return None
## end of function definition
```
""",
)

SEC1_2_DATA = SectionData(
    header="## Section 1.2",
    content="""- Bullet A
- Bullet B""",
)

CH2_DATA = SectionData(
    header="# Chapter 2", content="""Final paragraph in Chapter 2."""
)


# Combined fixture with all the data
@pytest.fixture
def markdown_sections() -> List[SectionData]:
    return [CH1_DATA, SEC1_1_DATA, SEC1_2_DATA, CH2_DATA]


@pytest.fixture
def sample_markdown(markdown_sections) -> str:
    return "".join(section.to_markdown() for section in markdown_sections)


def test_parse_markdown_headings(sample_markdown, markdown_sections):
    """
    Test the parse_markdown_headings_only function using a sample Markdown document.
    We verify that the resulting hierarchy of Nodes matches our expectations.
    """
    # Parse the sample markdown to a list of top-level Node objects
    tree = parse_markdown_headings(sample_markdown)
    ch1, sec1_1, sec1_2, ch2 = markdown_sections

    # We expect two top-level headings: Chapter 1 and Chapter 2
    assert len(tree) == 2

    # Check Chapter 1 node
    ch1_node = tree[0]
    assert isinstance(ch1_node, Node)
    assert ch1_node.content == ch1.header
    assert ch1_node.path == [ch1.header]

    # Under Chapter 1, we expect:
    #   1. A paragraph (intro)
    #   2. Heading "Section 1.1"
    #   3. Heading "Section 1.2"
    assert len(ch1_node.children) == 3

    intro_para = ch1_node.children[0]
    assert intro_para.content.strip() == ch1.content.strip()
    assert intro_para.path == [ch1.header]
    assert len(intro_para.children) == 0  # Paragraph has no sub-children

    section_11 = ch1_node.children[1]
    assert section_11.content == sec1_1.header
    assert section_11.path == [ch1.header, sec1_1.header]

    # Under Section 1.1, we expect a paragraph node
    assert len(section_11.children) == 1
    sec_11_para = section_11.children[0]
    assert sec_11_para.content.strip() == sec1_1.content.strip()
    assert sec_11_para.path == [ch1.header, sec1_1.header]

    section_12 = ch1_node.children[2]
    assert section_12.content == sec1_2.header
    assert section_12.path == [ch1.header, sec1_2.header]

    # Under Section 1.2, we expect a single content node containing the bullet points
    assert len(section_12.children) == 1
    bullets_node = section_12.children[0]
    # The bullet items are joined with newlines (per the extract_text logic)
    assert bullets_node.content.strip() == sec1_2.content.strip()
    assert bullets_node.path == [ch1.header, sec1_2.header]

    # Check Chapter 2 node
    ch2_node = tree[1]
    assert ch2_node.content == ch2.header
    assert ch2_node.path == [ch2.header]
    # Under Chapter 2, we expect a single paragraph
    assert len(ch2_node.children) == 1
    ch2_para = ch2_node.children[0]
    assert ch2_para.content.strip() == ch2.content.strip()
    assert ch2_para.path == [ch2.header]
    assert len(ch2_para.children) == 0


def test_empty_document():
    tree = parse_markdown_headings("")
    assert tree == []


def test_no_headers_only_paragraphs():
    md = """This is just a paragraph.
    
And another one.
"""
    tree = parse_markdown_headings(md)
    assert len(tree) == 1
    assert all(node.path == [] for node in tree)
    assert tree[0].content.strip() == md.strip()


def test_headers_with_no_content():
    md = """# Title
## Subsection
### Subsubsection
"""
    tree = parse_markdown_headings(md)
    assert len(tree) == 1
    assert tree[0].content == "# Title"
    assert len(tree[0].children) == 1
    assert tree[0].children[0].content == "## Subsection"
    assert len(tree[0].children[0].children) == 1
    assert tree[0].children[0].children[0].content == "### Subsubsection"
    assert tree[0].children[0].children[0].children == []


def test_header_with_inline_formatting():
    md = """# Header with **bold** and *italic* text
Some _content_.
"""
    tree = parse_markdown_headings(md)
    assert tree[0].content == "# Header with **bold** and *italic* text"
    assert tree[0].children[0].content.strip() == "Some _content_."


def test_lists_and_code_blocks():
    md = """# List and Code

## List Section
- Item 1
- Item 2

## Code Section

print("Hello, world!")
    
"""
    tree = parse_markdown_headings(md)

    list_section = tree[0].children[0]
    assert list_section.content == "## List Section"
    list_content = list_section.children[0]
    assert list_content.content == "- Item 1\n- Item 2"

    code_section = tree[0].children[1]
    assert code_section.content == "## Code Section"
    code_block = code_section.children[0]
    assert 'print("Hello, world!")' in code_block.content


def test_multiple_same_level_headers():
    md = """# Header A
Paragraph A.

# Header B
Paragraph B.
"""
    tree = parse_markdown_headings(md)
    assert len(tree) == 2
    assert tree[0].content == "# Header A"
    assert tree[0].children[0].content.strip() == "Paragraph A."
    assert tree[1].content == "# Header B"
    assert tree[1].children[0].content.strip() == "Paragraph B."


def test_header_skipping_levels():
    md = """# H1
### H3
Some text.
"""
    tree = parse_markdown_headings(md)
    h1 = tree[0]
    assert h1.content == "# H1"
    # H3 should be treated as a direct child of H1
    h3 = h1.children[0]
    assert h3.content == "### H3"
    assert h3.path == ["# H1", "### H3"]
    assert h3.children[0].content.strip() == "Some text."


@pytest.mark.parametrize("chunk_size_factor", [1.2, 100])
@pytest.mark.parametrize("rollup", [True, False])
def test_markdown_chunking(
    sample_markdown,
    markdown_sections,
    chunk_size_factor: int,
    rollup: bool,
):
    """
    Given a Markdown document with sections and sub-sections, this test verifies that:
      - The tree is built correctly from the document.
      - The chunking process produces distinct chunks with enriched header context.
      - A header-only node does not duplicate the header in its own chunk.

    The sample document has:
      - Chapter 1 with a preamble.
      - Section 1.1 with content.
      - Section 1.2 with bullet content.
      - Chapter 2 with its own content.
    """

    ch1, sec1_1, sec1_2, ch2 = markdown_sections
    chunk_size = chunk_size_factor * count_words(ch1.content)
    config = MarkdownChunkConfig(
        chunk_size=chunk_size,
        overlap_tokens=5,
        variation_percent=0.2,
        rollup=rollup,
    )

    # Structure-aware chunking of the text into enriched chunks.
    chunks: List[str] = chunk_markdown(sample_markdown, config)

    if rollup and chunk_size > count_words(sample_markdown):
        assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
        assert (
            chunks[0].split() == sample_markdown.split()
        ), "Chunk does not match original Markdown"
        # check that line-breaks in each section content are preserved
        for section in markdown_sections:
            assert (
                section.content in chunks[0]
            ), f"Section content not found in the chunk: {section.content}"

    if not rollup or chunk_size < count_words(sample_markdown):
        # Based on our document structure, we expect four chunks:
        # 1. Chapter 1's preamble content (enriched with prefix "# Chapter 1")
        # 2. Section 1.1 content (enriched with prefix "# Chapter 1 \n\n # Section 1.1")
        # 3. Section 1.2 content (enriched with prefix "# Chapter 1 \n\n # Section 1.2")
        # 4. Chapter 2 content (enriched with prefix "# Chapter 2")
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"

        assert (
            chunks[0].split() == ch1.to_markdown().split()
        ), "Chunk 1 does not match Chapter 1 preamble"
        assert (
            ch1.content.strip() in chunks[0]
        ), "Chapter 1 content not preserved in Chunk 1"

        assert chunks[1].split() == (
            (ch1.header + config.header_context_sep + sec1_1.to_markdown()).split()
        ), "Chunk 2 does not match Section 1.1"
        assert (
            sec1_1.content.strip() in chunks[1]
        ), "Section 1.1 content not preserved in Chunk 2"

        assert chunks[2].split() == (
            (ch1.header + config.header_context_sep + sec1_2.to_markdown()).split()
        ), "Chunk 3 does not match Section 1.2"
        assert (
            sec1_2.content.strip() in chunks[2]
        ), "Section 1.2 content not preserved in Chunk 3"

        assert (
            chunks[3].split() == ch2.to_markdown().split()
        ), "Chunk 4 does not match Chapter 2"
        assert (
            ch2.content.strip() in chunks[3]
        ), "Chapter 2 content not preserved in Chunk 4"


@pytest.mark.parametrize("rollup", [False, True])
@pytest.mark.parametrize("chunk_size", [20, 500])
def test_chunking_sizes(
    chunk_size: int,
    rollup: bool,
):
    """
    Test that the chunking logic produces chunks that:
      - Have token counts between the lower and upper bounds
        (except possibly the final chunk)
      - Include the header enrichment in each chunk's text
      - Include the expected overlap between consecutive chunks
    """
    # Create a long text consisting of 200 repeated tokens ("word")
    long_text = " ".join(["word"] * 200)  # 200 tokens
    md_text = f"""# Chapter 1
{long_text}
"""

    # Set chunking configuration.
    # Here chunk_size=50 means that (with variation_percent=0.2)
    # we expect chunks to have between 40 and 60 tokens.
    config = MarkdownChunkConfig(
        chunk_size=chunk_size, rollup=rollup, overlap_tokens=5, variation_percent=0.2
    )

    # Produce the enriched chunks from the tree.
    chunks = chunk_markdown(md_text, config)

    # Compute the allowed bounds.
    lower_bound = config.chunk_size * (1 - config.variation_percent)
    upper_bound = config.chunk_size * (1 + config.variation_percent)

    # Verify each chunk's token count.
    # For all chunks except possibly the final one,
    # we expect at least lower_bound tokens.
    for i, chunk in enumerate(chunks):
        tokens = count_words(chunk)
        if i < len(chunks) - 1:
            assert (
                tokens >= lower_bound
            ), f"Chunk {i} has {tokens} tokens, expected at least {lower_bound}"
        assert (
            tokens <= 2 * upper_bound
        ), (  # relaxed check
            f"Chunk {i} has {tokens} tokens, expected at most {upper_bound}"
        )

    # Check that each chunk is enriched with the header context.
    # Each chunk's text should contain "Chapter 1" since that is the header path.
    for i, chunk in enumerate(chunks):
        assert "Chapter 1" in chunk, f"Chunk {i} is missing header enrichment"

    # Verify that consecutive chunks share the expected overlap.
    # For each consecutive pair of chunks, the last `overlap_tokens`
    # tokens of the previous chunk
    # should appear among the first tokens of the next chunk.
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            prev_tokens = chunks[i].split()
            next_tokens = chunks[i + 1].split()
            # Get the last few tokens from the previous chunk:
            expected_overlap = prev_tokens[-config.overlap_tokens :]
            # Look at the beginning of the next chunk
            # (allowing some room for header enrichment).
            next_head = next_tokens[:15]
            for word in expected_overlap:
                assert (
                    word in next_head
                ), f"Overlap word '{word}' from chunk {i} not found in chunk {i+1}"


@pytest.mark.parametrize("rollup", [False, True])
def test_chunking_word_set_consistency(rollup: bool):
    """
    Test that when converting Markdown text to chunks, the union of distinct words
    from the original content is the same as the union of distinct words from the
    chunks (ignoring header enrichment).
    """
    # Define a header and content.
    header = "# Chapter 1"
    # Create content text with 100 unique words.
    content_tokens = [f"word{i}" for i in range(1, 101)]
    content_text = " ".join(content_tokens)

    # Create a sample Markdown document with a header and content.
    md_text = f"{header}\n\n{content_text}\n"

    # Set up the chunking configuration.
    config = MarkdownChunkConfig(
        chunk_size=20,  # small chunk size for testing
        overlap_tokens=5,  # intended overlap tokens
        variation_percent=0.2,  # chunks between 16 and 24 tokens
        rollup=rollup,
    )

    # Produce enriched chunks from the tree.
    chunks = chunk_markdown(md_text, config)

    # Remove header enrichment ("Chapter 1\n\n") from each chunk and collect all words.
    chunk_word_set = set()
    for chunk in chunks:
        assert chunk.startswith(header)
        # Split into words and update the set.
        chunk_word_set.update(chunk[len(header) :].split())

    # Compute the distinct set of words in the original content.
    original_word_set = set(content_text.split())

    # Verify that the union of words from the chunks
    # matches the original content's words.
    assert (
        chunk_word_set == original_word_set
    ), f"Word sets do not match.\nExpected: {original_word_set}\nGot: {chunk_word_set}"


def smart_tokenize(text: str) -> list:
    """
    Tokenize text by first inserting a space after a period if it's immediately
    followed by an uppercase letter (a common side-effect of line joining),
    then splitting on whitespace.
    """
    fixed = re.sub(r"(\.)([A-Z])", r"\1 \2", text)
    return fixed.split()


def generate_sentence(word_count: int, sentence_id: int) -> str:
    """
    Generate a dummy sentence with `word_count` words and a trailing period.
    Uses "wordX" to identify each word, plus a "sentence{ID}" marker at the end.
    """
    words = [f"word{i}" for i in range(1, word_count + 1)]
    # Put a sentinel to identify the sentence number
    # and close with a period for the splitting logic.
    sentence_str = " ".join(words) + f" sentence{sentence_id}."
    return sentence_str


def generate_paragraph(
    sentence_count: int,
    words_per_sentence: int,
    paragraph_id: int,
) -> str:
    """
    Generate a dummy paragraph with `sentence_count` sentences,
    each with `words_per_sentence` words.
    """
    sentences = [
        generate_sentence(words_per_sentence, s_id + 1)
        for s_id in range(sentence_count)
    ]
    # Add a sentinel "PARA{ID}" at the end to visually check paragraph boundaries.
    para_str = " ".join(sentences) + f" PARA{paragraph_id}"
    return para_str


@pytest.mark.parametrize("chunk_size_factor", [0.5, 1, 1.5])
@pytest.mark.parametrize("rollup", [False, True])
def test_degenerate_markdown_parsing_and_chunking(
    chunk_size_factor: float,
    rollup: bool,
):
    # A degenerate Markdown document: plain text without any Markdown formatting.

    paragraph1 = generate_paragraph(
        sentence_count=10, words_per_sentence=50, paragraph_id=1
    )
    paragraph2 = generate_paragraph(
        sentence_count=10, words_per_sentence=50, paragraph_id=2
    )

    # Combine paragraphs with a double-newline
    plain_text = paragraph1 + "\n\n" + paragraph2
    plain_text = plain_text.strip()

    # Parse the plain text using our Markdown parser.
    tree = parse_markdown_headings(plain_text)

    # For plain text, we expect a single node.
    assert len(tree) == 1, "Expected one node for plain text"
    node = tree[0]

    # Use smart_tokenize to account for missing spaces at line joins.
    expected_tokens = set(smart_tokenize(plain_text))
    actual_tokens = set(smart_tokenize(node.content))
    assert (
        expected_tokens == actual_tokens
    ), "Distinct word sets from node content and original plain text do not match"

    # Plain text should not have header enrichment.
    assert node.path == [] or node.path == [""], "Plain text should have no header path"
    assert node.children == [], "Plain text should not produce any children nodes"

    # Set up a chunking configuration.
    config = MarkdownChunkConfig(
        desired_chunk_tokens=50,  # high enough to avoid splitting for this test
        overlap_tokens=5,
        variation_percent=0.2,
        rollup=rollup,
    )

    # Generate chunks from the parsed tree.
    chunks = chunk_markdown(plain_text, config)

    # Collect distinct words from the chunks.
    chunk_word_set = set()
    for chunk in chunks:
        # Since there is no header enrichment (no headers in plain text),
        # we can tokenize directly.
        chunk_word_set.update(smart_tokenize(chunk))

    original_word_set = set(smart_tokenize(plain_text))
    assert chunk_word_set == original_word_set, (
        f"Word sets do not match between chunks and original text.\n"
        f"Expected: {original_word_set}\nGot: {chunk_word_set}"
    )


def condensed_chunk_view(chunks: List[str], max_words: int = 5) -> str:
    """
    Return a compact string showing each chunk's first/last few words and total length.
    """
    lines = []
    for i, c in enumerate(chunks):
        words = c.split()
        total = len(words)
        if total <= 2 * max_words:
            preview = " ".join(words)
        else:
            preview = (
                " ".join(words[:max_words]) + " ... " + " ".join(words[-max_words:])
            )
        lines.append(f"Chunk {i+1} (total {total} words): {preview}")
    return "\n".join(lines)


# ----------------------------------------------------------------
# THE TEST
# ----------------------------------------------------------------


@pytest.mark.parametrize(
    "chunk_size, overlap_tokens, variation_percent",
    [
        (50, 5, 0.3),  # Scenario 1
        (20, 5, 0.3),  # Scenario 2
        (8, 3, 0.3),  # Scenario 3 (forces word-level splits)
    ],
)
def test_recursive_chunk(chunk_size, overlap_tokens, variation_percent):
    """
    Tests that the chunker respects paragraph boundaries when possible,
    then sentence boundaries, and only splits sentences when no other option
    is possible under the given config.
    """
    # Generate some text with 2 paragraphs, each having 3 sentences of 10 words.
    # ~ Each paragraph => 3 sentences =>
    #    each sentence has ~10 words => ~30 words per paragraph.
    # So total words ~60. This helps us see chunking behavior across boundaries.
    paragraph1 = generate_paragraph(
        sentence_count=3, words_per_sentence=10, paragraph_id=1
    )
    paragraph2 = generate_paragraph(
        sentence_count=3, words_per_sentence=10, paragraph_id=2
    )

    # Combine paragraphs with a double-newline
    text = paragraph1 + "\n\n" + paragraph2

    config = MarkdownChunkConfig(
        chunk_size=chunk_size,
        overlap_tokens=overlap_tokens,
        variation_percent=variation_percent,
    )

    chunks = recursive_chunk(text, config)

    # Print a condensed view for manual inspection
    print("\n===================================")
    print(
        f"Config: chunk_size={chunk_size}, "
        f"overlap_tokens={overlap_tokens}, "
        f"variation_percent={variation_percent}\n"
    )
    print("Generated Text (first 30 words):")
    print(" ".join(text.split()[:30]), "...")
    print("\nChunks:")
    print(condensed_chunk_view(chunks, max_words=5))
    print("===================================\n")

    # Basic asserts:
    # 1. No chunk should exceed the upper bound in terms of word count
    upper_bound = chunk_size * (1 + variation_percent)
    for i, chunk in enumerate(chunks):
        word_count_in_chunk = len(chunk.split())
        assert word_count_in_chunk <= upper_bound + 5, (
            f"Chunk {i+1} has {word_count_in_chunk} words, "
            f"exceeds upper bound (~{upper_bound:.1f})."
        )

    # 2. Check that chunking doesn't produce empty chunks
    for i, chunk in enumerate(chunks):
        assert chunk.strip(), f"Chunk {i+1} is empty!"

    # 3. (Optional) If chunk_size is >= total words, we expect exactly 1 chunk
    total_words = len(text.split())
    if total_words <= chunk_size * (1 + variation_percent):
        assert len(chunks) == 1, (
            "Expected a single chunk since the text is short enough, "
            f"but got {len(chunks)} chunks."
        )


def test_recursive_chunk_enhanced():
    config = MarkdownChunkConfig(
        chunk_size=8,
        overlap_tokens=2,
        variation_percent=0.3,
    )

    # Construct a text with 2 paragraphs, each containing 2 sentences,
    # plus paragraph markers
    paragraph1 = (
        "word1 word2 word3 word4 sentence1.\n"
        "word5 word6 word7 word8 sentence2. PARA1"
    )
    paragraph2 = (
        "cat1 cat2 cat3 cat4 sentence1.\n" "cat5 cat6 cat7 cat8 sentence2. PARA2"
    )

    text = paragraph1 + "\n\n" + paragraph2

    # Now chunk it
    chunks = recursive_chunk(text, config)

    print("\n------------------ ENHANCED CHUNK TEST ------------------")
    for i, c in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(c.split())} words):\n{c}\n")

    # A. Check no chunk splits mid-sentence
    for i, chunk in enumerate(chunks, 1):
        # We expect every sentence boundary to remain intact:
        # "sentence1." or "sentence2." should not be truncated in the middle
        assert (
            "sentence1." in chunk or "sentence2." in chunk or "PARA" in chunk
        ), f"Chunk {i} might have truncated a sentence or lost markers: {chunk}"

    # B. Check paragraph markers do not get merged.
    # We expect that "PARA1" and "PARA2" never appear in the same chunk.
    for i, chunk in enumerate(chunks, 1):
        assert not (
            "PARA1" in chunk and "PARA2" in chunk
        ), "Found both PARA1 and PARA2 in the same chunk!"

    # C. If there's overlap, ensure it's only from chunk (i) to chunk (i+1).
    # A naive check: the last 2 tokens of chunk i = the first 2 tokens of chunk i+1,
    # but chunk i+2 does not contain that same overlap at the start.
    for i in range(len(chunks) - 1):
        chunk_i_tokens = chunks[i].split()
        chunk_i_plus_1_tokens = chunks[i + 1].split()

        overlap_i = chunk_i_tokens[-2:]  # last 2 tokens of chunk i
        start_of_chunk_i_plus_1 = chunk_i_plus_1_tokens[
            :2
        ]  # first 2 tokens of chunk i+1
        assert overlap_i == start_of_chunk_i_plus_1, (
            f"Expected chunk {i+1} to start with overlap tokens from chunk {i}.\n"
            f"Overlap {overlap_i}, got {start_of_chunk_i_plus_1}"
        )

        # Now check chunk (i+2) if it exists
        if i + 2 < len(chunks):
            chunk_i_plus_2_tokens = chunks[i + 2].split()
            # The first 2 tokens of chunk i+2 should NOT match overlap_i
            start_of_chunk_i_plus_2 = chunk_i_plus_2_tokens[:2]
            assert (
                start_of_chunk_i_plus_2 != overlap_i
            ), f"Found repeated overlap in chunk {i+2} that belonged to chunk {i}!"

    # D. For formatting, ensure that if a chunk boundary occurs right before
    # a paragraph break, the next chunk still preserves '\n\n'
    # if it was originally there.
    # e.g. if chunk i ends in "sentence2. PARA1" plus "\n\n", the next chunk
    # should start with something that includes the next paragraph. We
    # can do a simple check that either chunk i ends with \n\n or chunk i+1
    # starts with it.
    for i in range(len(chunks) - 1):
        # If the original text had \n\n between paragraphs, we expect
        # either chunk i ends with \n\n or chunk i+1 starts with \n\n.
        if "PARA1" in chunks[i]:  # likely the end of paragraph 1
            # then chunk i+1 should contain the next paragraph's text,
            # ideally starting with \n\n + "cat1" or an overlap snippet.
            next_chunk = chunks[i + 1]
            # Checking minimal condition: that next_chunk includes "cat1" or "cat5"...
            assert "cat1" in next_chunk or "cat5" in next_chunk, (
                f"Paragraph formatting might have been lost: chunk {i+1} "
                f"does not contain cat1/cat5"
            )
