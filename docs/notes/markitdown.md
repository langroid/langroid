# Markitdown Document Parsers

Langroid integrates with Microsoft's Markitdown library to provide 
conversion of Microsoft Office documents to markdown format. 
Three specialized parsers are available, for `docx`, `xlsx`, and `pptx` files.



## Prerequisites

To use these parsers, install Langroid with the required extras:

```bash
pip install "langroid[markitdown]"    # Just Markitdown parsers
# or
pip install "langroid[doc-parsers]"   # All document parsers
```

## Available Parsers


Once you set up a `parser` for the appropriate document-type, you  
can get the entire document with `parser.get_doc()`,
or get automatically chunked content with `parser.get_doc_chunks()`.


### 1. `MarkitdownDocxParser`

Converts Word documents (`*.docx`) to markdown, preserving structure, 
formatting, and tables.

See the tests

- [`test_docx_parser.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_docx_parser.py)
- [`test_markitdown_parser.py`](https://github.com/langroid/langroid/blob/main/tests/main/test_markitdown_parser.py)

for examples of how to use these parsers.


```python
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import DocxParsingConfig, ParsingConfig

parser = DocumentParser.create(
    "path/to/document.docx",
    ParsingConfig(
        docx=DocxParsingConfig(library="markitdown-docx"),
        # ... other parsing config options
    ),
)

```


### 2. `MarkitdownXLSXParser`

Converts Excel spreadsheets (*.xlsx/*.xls) to markdown tables, preserving data and sheet structure.

```python
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, MarkitdownXLSParsingConfig

parser = DocumentParser.create(
    "path/to/spreadsheet.xlsx",
    ParsingConfig(xls=MarkitdownXLSParsingConfig())
)
```


### 3. `MarkitdownPPTXParser`

Converts PowerPoint presentations (*.pptx) to markdown, preserving slide content and structure.

```python
from langroid.parsing.document_parser import DocumentParser
from langroid.parsing.parser import ParsingConfig, MarkitdownPPTXParsingConfig

parser = DocumentParser.create(
    "path/to/presentation.pptx",
    ParsingConfig(pptx=MarkitdownPPTXParsingConfig())
)
```

