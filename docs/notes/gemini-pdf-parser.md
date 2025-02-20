# Using the Gemini PDF Parser

## Overview
The Gemini PDF parser allows extracting structured text and images from PDF documents efficiently. This guide explains how to configure and use the parser in a Python environment.

## Features
- Converts PDF content into Markdown format.
- Uses Gemini multimodal models to describe images within PDFs.
- Supports page-wise or chunk-based processing for optimized performance.

## Setup and Configuration

### Initializing the Gemini PDF Parser

You can initialize the Gemini PDF parser as follows:

```python
parsing_config = ParsingConfig(
    n_neighbor_ids=2,
    pdf=PdfParsingConfig(
        library="gemini",
        gemini_config=GeminiConfig(
            model_name="gemini-2.0-flash",
            split_on_page=True,
            output_filename=output_file_name,
            max_tokens=7000,
            requests_per_minute=5,
        ),
    ),
)
```

## Parameters

### `model_name`
Specifies the Gemini model to use for PDF conversion. Default: `gemini-2.0-flash`.

### `max_tokens`
Limits the number of tokens in the input. The model's maximum context length is **8192 tokens**.
- Default: **7000 tokens** (leaving room for generated captions).
- Optional parameter.

### `split_on_page`
Determines whether to process the document **page by page**.
- **Default: `True`**
- If set to `False`, the parser will create chunks based on `max_tokens` while respecting page boundaries.
- When `False`, the parser will send chunks containing multiple pages (e.g., `[11,12,13,14,15]`).
- **Advantages of `False`:**
  - Reduces API calls to the LLM.
  - Lowers token usage since system prompts are not repeated per page.
- **Disadvantages of `False`:**
  - Delegates the task of page boundary management to the LLM by adding page boundary marker while splitting pages.
  - Hence output might not maintain strict per-page formatting.
- If your use case does **not** require strict page-by-page parsing, consider setting this to `False`.

### `output_filename`
Defines where to save the parsed document.
- When directly passing the bytes, ensure an output filename is specified.
- By default, everything is saved in the directory `.gemini-pdfparser`.

### `requests_per_minute`
Limits API request frequency to avoid rate limits.
- If you encounter rate limits, set this to **1 or 2**.
