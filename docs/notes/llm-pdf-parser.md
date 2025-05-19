# Using the LLM-based PDF Parser

- Converts PDF content into Markdown format using Multimodal models.
- Uses multimodal models to describe images within PDFs.
- Supports page-wise or chunk-based processing for optimized performance.

---

### Initializing the LLM-based PDF Parser

Make sure you have set up your API key for whichever model you specify in `model_name` below.

You can initialize the LLM PDF parser as follows:

```python
parsing_config = ParsingConfig(
    n_neighbor_ids=2,
    pdf=PdfParsingConfig(
        library="llm-pdf-parser",
        llm_parser_config=LLMPdfParserConfig(
            model_name="gemini-2.0-flash",
            split_on_page=True,
            max_tokens=7000,
            requests_per_minute=5,
            timeout=60,  # increase this for large documents
        ),
    ),
)
```

---

## Parameters

### `model_name`

Specifies the model to use for PDF conversion.
**Default:** `gemini/gemini-2.0-flash`

---

### `max_tokens`

Limits the number of tokens in the input. The model's output limit is **8192 tokens**.

- **Default:** 7000 tokens (leaving room for generated captions)
- _Optional parameter_

---

### `split_on_page`

Determines whether to process the document **page by page**.

- **Default:** `True`
- If set to `False`, the parser will create chunks based on `max_tokens` while respecting page boundaries.
- When `False`, the parser will send chunks containing multiple pages (e.g., `[11,12,13,14,15]`).

**Advantages of `False`:**

- Reduces API calls to the LLM.
- Lowers token usage since system prompts are not repeated per page.

**Disadvantages of `False`:**

- You will not get per-page splitting but groups of pages as a single unit.

> If your use case does **not** require strict page-by-page parsing, consider setting this to `False`.

---

### `requests_per_minute`

Limits API request frequency to avoid rate limits.

- If you encounter rate limits, set this to **1 or 2**.

---
