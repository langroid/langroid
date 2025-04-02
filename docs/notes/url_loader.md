# Firecrawl and Trafilatura Crawlers Documentation

`URLLoader` uses `Trafilatura` if not explicitly specified

## Overview
*   **`FirecrawlCrawler`**:  Leverages the Firecrawl API for efficient web scraping and crawling. 
It offers built-in document processing capabilities, and 
**produces non-chunked markdown output** from web-page content.
Requires `FIRECRAWL_API_KEY` environment variable to be set in `.env` file or environment.
*   **`TrafilaturaCrawler`**: Utilizes the Trafilatura library and Langroid's parsing tools 
for extracting and processing web content - this is the default crawler, and 
does not require setting up an external API key. Also produces 
**chuked markdown output** from web-page content.
*  **`ExaCrawler`**: Integrates with the Exa API for high-quality content extraction.
  Requires `EXA_API_KEY` environment variable to be set in `.env` file or environment.
This crawler also produces **chunked markdown output** from web-page content.


## Installation

`TrafilaturaCrawler` comes with Langroid

To use `FirecrawlCrawler`, install the `firecrawl` extra:

```bash
pip install langroid[firecrawl]
```

## Exa Crawler Documentation

### Overview

`ExaCrawler` integrates with Exa API to extract high-quality content from web pages. 
It provides efficient content extraction with the simplicity of API-based processing.

### Parameters

Obtain an Exa API key from [Exa](https://exa.ai/) and set it in your environment variables, 
e.g. in your `.env` file as:

```env
EXA_API_KEY=your_api_key_here
```

* **config (ExaCrawlerConfig)**: An `ExaCrawlerConfig` object.
    * **api_key (str)**: Your Exa API key.

### Usage

```python
from langroid.parsing.url_loader import URLLoader, ExaCrawlerConfig

# Create an ExaCrawlerConfig object
exa_config = ExaCrawlerConfig(
    # Typically omitted here as it's loaded from EXA_API_KEY environment variable
    api_key="your-exa-api-key" 
)

loader = URLLoader(
    urls=[
        "https://pytorch.org",
        "https://www.tensorflow.org"
    ],
    crawler_config=exa_config
)

docs = loader.load()
print(docs)
```

### Benefits

* Simple API integration requiring minimal configuration
* Efficient handling of complex web pages
* For plain html content, the `exa` api produces high-quality content extraction with 
clean text output with html tags, which we then convert to markdown using the `markdownify` library.
* For "document" content (e.g., `pdf`, `doc`, `docx`), 
the content is downloaded via the `exa` API and langroid's document-processing 
tools are used to produce **chunked output** in a format controlled by the `Parser` configuration
  (defaults to markdown in most cases).


## Trafilatura Crawler Documentation

### Overview

`TrafilaturaCrawler` is a web crawler that uses the Trafilatura library for content extraction 
and Langroid's parsing capabilities for further processing. 


### Parameters

*   **config (TrafilaturaConfig)**: A `TrafilaturaConfig` object that specifies
    parameters related to scraping or output format.
    * `threads` (int): The number of threads to use for downloading web pages.
    * `format` (str): one of `"markdown"` (default), `"xml"` or `"txt"`; in case of `xml`, 
    the output is in html format.

Similar to the `ExaCrawler`, the `TrafilaturaCrawler` works differently depending on 
the type of web-page content:
- for "document" content (e.g., `pdf`, `doc`, `docx`), the content is downloaded
  and parsed with Langroid's document-processing tools are used to produce **chunked output** 
  in a format controlled by the `Parser` configuration (defaults to markdown in most cases).
- for plain-html content, the output format is based on the `format` parameter; 
  - if this parameter is `markdown` (default), the library extracts content in 
    markdown format, and the final output is a list of chunked markdown documents.
  - if this parameter is `xml`, content is extracted in `html` format, which 
    langroid then converts to markdown using the `markdownify` library, and the final
    output is a list of chunked markdown documents.
  - if this parameter is `txt`, the content is extracted in plain text format, and the final
    output is a list of plain text documents.

### Usage

```python
from langroid.parsing.url_loader import URLLoader, TrafilaturaConfig

# Create a TrafilaturaConfig instance
trafilatura_config = TrafilaturaConfig(threads=4)


loader = URLLoader(
    urls=[
        "https://pytorch.org",
        "https://www.tensorflow.org",
        "https://ai.google.dev/gemini-api/docs",
        "https://books.toscrape.com/"
    ],
    crawler_config=trafilatura_config,
)

docs = loader.load()
print(docs)
```

### Langroid Parser Integration

`TrafilaturaCrawler` relies on a Langroid `Parser` to handle document processing. 
The `Parser` uses the default parsing methods or with a configuration that 
can be adjusted to suit the current use case.

## Firecrawl Crawler Documentation

### Overview

`FirecrawlCrawler` is a web crawling utility class that uses the Firecrawl API 
to scrape or crawl web pages efficiently. It offers two modes:

*   **Scrape Mode (default)**: Extracts content from a list of specified URLs.
*   **Crawl Mode**: Recursively follows links from a starting URL, 
gathering content from multiple pages, including subdomains, while bypassing blockers.  
**Note:** `crawl` mode accepts only ONE URL as a list.

### Parameters

Obtain a Firecrawl API key from [Firecrawl](https://firecrawl.dev/) and set it in 
your environment variables, e.g. in your `.env` file as
```env
FIRECRAWL_API_KEY=your_api_key_here
```

*   **config (FirecrawlConfig)**:  A `FirecrawlConfig` object.

    *   **timeout (int, optional)**: Time in milliseconds (ms) to wait for a response. 
        Default is `30000ms` (30 seconds). In crawl mode, this applies per URL.
    *   **limit (int, optional)**: Maximum number of pages to scrape in crawl mode. Helps control API usage.
    *   **params (dict, optional)**: Additional parameters to customize the request. 
        See the [scrape API](https://docs.firecrawl.dev/api-reference/endpoint/scrape) and 
        [crawl API](https://docs.firecrawl.dev/api-reference/endpoint/crawl-post) for details.

### Usage

#### Scrape Mode (Default)

Fetch content from multiple URLs:

```python
from langroid.parsing.url_loader import URLLoader, FirecrawlConfig
from langroid.parsing.document_parser import 

# create a FirecrawlConfig object
firecrawl_config = FirecrawlConfig(
    # typical/best practice is to omit the api_key, and 
    # we leverage Pydantic BaseSettings to load it from the environment variable
    # FIRECRAWL_API_KEY in your .env file
    api_key="your-firecrawl-api-key", 
    timeout=15000,  # Timeout per request (15 sec)
    mode="scrape",
)

loader = URLLoader(
    urls=[
        "https://pytorch.org",
        "https://www.tensorflow.org",
        "https://ai.google.dev/gemini-api/docs",
        "https://books.toscrape.com/"
    ],
    crawler_config=firecrawl_config
)

docs = loader.load()
print(docs)
```

#### Crawl Mode

Fetch content from multiple pages starting from a single URL:

```python
from langroid.parsing.url_loader import URLLoader, FirecrawlConfig

# create a FirecrawlConfig object
firecrawl_config = FirecrawlConfig(
    timeout=30000,  # 10 sec per page
    mode="crawl",
    params={
        "limit": 5,
    }
)


loader = URLLoader(
    urls=["https://books.toscrape.com/"],
    crawler_config=firecrawl_config
)

docs = loader.load()
print(docs)
```

### Output

Results are stored in the `firecrawl_output` directory.

### Best Practices

*   Set `limit` in crawl mode to avoid excessive API usage.
*   Adjust `timeout` based on network conditions and website responsiveness.
*   Use `params` to customize scraping behavior based on Firecrawl API capabilities.

### Firecrawl's Built-In Document Processing

`FirecrawlCrawler` benefits from Firecrawl's built-in document processing, 
which automatically extracts and structures content from web pages (including pdf,doc,docx). 
This reduces the need for complex parsing logic within Langroid.
Unlike the `Exa` and `Trafilatura` crawlers, the resulting documents are 
*non-chunked* markdown documents. 

## Choosing a Crawler

*   Use `FirecrawlCrawler` when you need efficient, API-driven scraping with built-in document processing. 
This is often the simplest and most effective choice, but incurs a cost due to 
the paid API. 
*   Use `TrafilaturaCrawler` when you want local non API based scraping (less accurate ).
*   Use `ExaCrawlwer` as a sort of middle-ground between the two, 
    with high-quality content extraction for plain html content, but rely on 
    Langroid's document processing tools for document content. This will cost
    significantly less than Firecrawl.

## Example script

See the script [`examples/docqa/chat_search.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/chat_search.py) 
which shows how to use a Langroid agent to search the web and scrape URLs to answer questions.