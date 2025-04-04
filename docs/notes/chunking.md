# Document Chunking/Splitting in Langroid

Langroid's [`ParsingConfig`][langroid.parsing.parser.ParsingConfig]
provides several document chunking strategies through the `Splitter` enum:

## 1. MARKDOWN (`Splitter.MARKDOWN`) (The default)

**Purpose**: Structure-aware splitting that preserves markdown formatting.

**How it works**:

- Preserves document hierarchy (headers and sections)
- Enriches chunks with header information
- Uses word count instead of token count (with adjustment factor)
- Supports "rollup" to maintain document structure
- Ideal for markdown documents where preserving formatting is important

## 2. TOKENS (`Splitter.TOKENS`)

**Purpose**: Creates chunks of approximately equal token size.

**How it works**:

- Tokenizes the text using tiktoken
- Aims for chunks of size `chunk_size` tokens (default: 200)
- Looks for natural breakpoints like punctuation or newlines
- Prefers splitting at sentence/paragraph boundaries
- Ensures chunks are at least `min_chunk_chars` long (default: 350)

## 3. PARA_SENTENCE (`Splitter.PARA_SENTENCE`)

**Purpose**: Splits documents respecting paragraph and sentence boundaries.

**How it works**:

- Recursively splits documents until chunks are below 1.3× the target size
- Maintains document structure by preserving natural paragraph breaks
- Adjusts chunk boundaries to avoid cutting in the middle of sentences
- Stops when it can't split chunks further without breaking coherence

## 4. SIMPLE (`Splitter.SIMPLE`)

**Purpose**: Basic splitting using predefined separators.

**How it works**:

- Uses a list of separators to split text (default: `["\n\n", "\n", " ", ""]`)
- Splits on the first separator in the list
- Doesn't attempt to balance chunk sizes
- Simplest and fastest splitting method


## Basic Configuration

```python
from langroid.parsing.parser import ParsingConfig, Splitter

config = ParsingConfig(
    splitter=Splitter.MARKDOWN,  # Most feature-rich option
    chunk_size=200,              # Target tokens per chunk
    chunk_size_variation=0.30,   # Allowed variation from target
    overlap=50,                  # Token overlap between chunks
    token_encoding_model="text-embedding-3-small"
)
```

## Format-Specific Configuration

```python
# Customize PDF parsing
config = ParsingConfig(
    splitter=Splitter.PARA_SENTENCE,
    pdf=PdfParsingConfig(
        library="pymupdf4llm"  # Default PDF parser
    )
)

# Use Gemini for PDF parsing
config = ParsingConfig(
    pdf=PdfParsingConfig(
        library="gemini",
        gemini_config=GeminiConfig(
            model_name="gemini-2.0-flash",
            requests_per_minute=5
        )
    )
)
```

# Setting Up Parsing Config in DocChatAgentConfig

You can configure document parsing when creating a `DocChatAgent` by customizing the `parsing` field within the `DocChatAgentConfig`. Here's how to do it:

```python
from langroid.agent.special.doc_chat_agent import DocChatAgentConfig  
from langroid.parsing.parser import ParsingConfig, Splitter, PdfParsingConfig

# Create a DocChatAgent with custom parsing configuration
agent_config = DocChatAgentConfig(
    parsing=ParsingConfig(
        # Choose the splitting strategy
        splitter=Splitter.MARKDOWN,  # Structure-aware splitting with header context
        
        # Configure chunk sizes
        chunk_size=800,              # Target tokens per chunk
        overlap=150,                 # Overlap between chunks
        
        # Configure chunk behavior
        max_chunks=5000,             # Maximum number of chunks to create
        min_chunk_chars=250,         # Minimum characters when truncating at punctuation
        discard_chunk_chars=10,      # Discard chunks smaller than this
        
        # Configure context window
        n_neighbor_ids=3,            # Store 3 chunk IDs on either side
        
        # Configure PDF parsing specifically
        pdf=PdfParsingConfig(
            library="pymupdf4llm",   # Choose PDF parsing library
        )
    )
)
```