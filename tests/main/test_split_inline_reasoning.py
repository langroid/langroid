"""
Tests for OpenAIGPT._split_inline_reasoning, which separates inline
thought-delimiter tags (e.g. <think>...</think>) from text content
during streaming.
"""

from langroid.language_models.openai_gpt import OpenAIGPT

split = OpenAIGPT._split_inline_reasoning
DELIMS = ("<think>", "</think>")


class TestSplitInlineReasoning:
    """Unit tests for the streaming inline-reasoning splitter."""

    # --- no-op cases: nothing to parse ---

    def test_empty_text(self):
        """Empty event_text should pass through unchanged."""
        text, reasoning, in_r = split("", "", False, DELIMS)
        assert text == ""
        assert reasoning == ""
        assert in_r is False

    def test_plain_text_no_delimiters(self):
        """Text without any delimiters should pass through as-is."""
        text, reasoning, in_r = split("hello world", "", False, DELIMS)
        assert text == "hello world"
        assert reasoning == ""
        assert in_r is False

    def test_already_has_reasoning_field(self):
        """When the API already provides separate reasoning, skip parsing."""
        text, reasoning, in_r = split("some text", "api reasoning", False, DELIMS)
        assert text == "some text"
        assert reasoning == "api reasoning"
        assert in_r is False

    # --- single chunk contains full <think>...</think> ---

    def test_full_think_block_in_one_chunk(self):
        """Complete <think>reasoning</think> in a single chunk."""
        text, reasoning, in_r = split("<think>step 1</think>", "", False, DELIMS)
        assert text == ""
        assert reasoning == "step 1"
        assert in_r is False

    def test_text_before_and_after_think(self):
        """Text surrounding a complete think block."""
        text, reasoning, in_r = split(
            "before<think>middle</think>after", "", False, DELIMS
        )
        assert text == "beforeafter"
        assert reasoning == "middle"
        assert in_r is False

    def test_text_before_think_only(self):
        """Text before think block, nothing after."""
        text, reasoning, in_r = split(
            "prefix<think>reasoning</think>", "", False, DELIMS
        )
        assert text == "prefix"
        assert reasoning == "reasoning"
        assert in_r is False

    def test_text_after_think_only(self):
        """Think block followed by text, no prefix."""
        text, reasoning, in_r = split(
            "<think>reasoning</think>suffix", "", False, DELIMS
        )
        assert text == "suffix"
        assert reasoning == "reasoning"
        assert in_r is False

    # --- multi-chunk: start delimiter in one chunk, end in another ---

    def test_start_delimiter_only(self):
        """Chunk has <think> but no </think> — enters reasoning state."""
        text, reasoning, in_r = split(
            "hello<think>partial reasoning", "", False, DELIMS
        )
        assert text == "hello"
        assert reasoning == "partial reasoning"
        assert in_r is True

    def test_continuation_mid_reasoning(self):
        """Chunk arrives while already in reasoning (no delimiters)."""
        text, reasoning, in_r = split("more reasoning stuff", "", True, DELIMS)
        assert text == ""
        assert reasoning == "more reasoning stuff"
        assert in_r is True

    def test_end_delimiter_while_in_reasoning(self):
        """Chunk has </think> while in reasoning state."""
        text, reasoning, in_r = split("final bit</think>answer", "", True, DELIMS)
        assert text == "answer"
        assert reasoning == "final bit"
        assert in_r is False

    def test_end_delimiter_no_trailing_text(self):
        """End delimiter with nothing after it."""
        text, reasoning, in_r = split("last thought</think>", "", True, DELIMS)
        assert text == ""
        assert reasoning == "last thought"
        assert in_r is False

    # --- multi-chunk sequence simulating a real stream ---

    def test_three_chunk_sequence(self):
        """Simulate: chunk1 opens thinking, chunk2 continues, chunk3 closes."""
        # chunk 1: start of thinking
        text1, reason1, in_r = split("<think>step 1", "", False, DELIMS)
        assert text1 == ""
        assert reason1 == "step 1"
        assert in_r is True

        # chunk 2: still thinking
        text2, reason2, in_r = split(" step 2", "", in_r, DELIMS)
        assert text2 == ""
        assert reason2 == " step 2"
        assert in_r is True

        # chunk 3: done thinking, answer follows
        text3, reason3, in_r = split(" step 3</think>The answer", "", in_r, DELIMS)
        assert text3 == "The answer"
        assert reason3 == " step 3"
        assert in_r is False

    # --- custom delimiters ---

    def test_custom_delimiters(self):
        """Works with non-default delimiters like <thinking>...</thinking>."""
        custom = ("<thinking>", "</thinking>")
        text, reasoning, in_r = split(
            "<thinking>hmm</thinking>result", "", False, custom
        )
        assert text == "result"
        assert reasoning == "hmm"
        assert in_r is False

    # --- edge cases ---

    def test_empty_think_block(self):
        """<think></think> with nothing inside."""
        text, reasoning, in_r = split("<think></think>answer", "", False, DELIMS)
        assert text == "answer"
        assert reasoning == ""
        assert in_r is False

    def test_only_start_delimiter(self):
        """Chunk is exactly the start delimiter, nothing else."""
        text, reasoning, in_r = split("<think>", "", False, DELIMS)
        assert text == ""
        assert reasoning == ""
        assert in_r is True

    def test_only_end_delimiter_while_reasoning(self):
        """Chunk is exactly the end delimiter."""
        text, reasoning, in_r = split("</think>", "", True, DELIMS)
        assert text == ""
        assert reasoning == ""
        assert in_r is False
