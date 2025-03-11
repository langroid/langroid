# Firecrawl and Trafilatura Crawlers Documentation

`URLLoader` uses `Trafilatura` if not explicitly specified

## Overview
*   **`FirecrawlCrawler`**:  Leverages the Firecrawl API for efficient web scraping and crawling. It offers built-in document processing capabilities.
*   **`TrafilaturaCrawler`**: Utilizes the Trafilatura library and Langroid's parsing tools for extracting and processing web content.

## Installation

`TrafilaturaCrawler` comes with Langroid

To use `FirecrawlCrawler`, install the `firecrawl` extra:

```bash
pip install langroid[firecrawl]
```

## Trafilatura Crawler Documentation

### Overview

`TrafilaturaCrawler` is a web crawler that uses the Trafilatura library for content extraction and Langroid's parsing capabilities for further processing. This crawler is useful when you need more control over the parsing process and want to leverage Langroid's document processing tools.

### Parameters

*   **parser (Parser)**: A Langroid `Parser` object that defines how to process the extracted text. See Langroid's documentation on parsing for details.
*   **threads (int)**: The number of threads to use for downloading web pages.

### Usage

```python
from langroid.parsing.url_loader import URLLoader
from langroid.crawlers import TrafilaturaCrawler
from langroid.parsing.parser import Parser, ParsingConfig

# Define a parsing configuration
parsing_config = ParsingConfig()

# Create a Parser instance
parser = Parser(parsing_config)

loader = URLLoader(
    urls=[
        "https://pytorch.org",
        "https://www.tensorflow.org",
        "https://ai.google.dev/gemini-api/docs",
        "https://books.toscrape.com/"
    ],
    crawler=TrafilaturaCrawler(parser=parser, threads=4),
)

docs = loader.load()
print(docs)
```

### Langroid Parser Integration

`TrafilaturaCrawler` relies on a Langroid `Parser` to handle document processing. The `Parser` uses the default parsing methods or with a configuration that can be adjust to more suit the current use case.

## Firecrawl Crawler Documentation

### Overview

`FirecrawlCrawler` is a web crawling utility class that uses the Firecrawl API to scrape or crawl web pages efficiently. It offers two modes:

*   **Scrape Mode (default)**: Extracts content from a list of specified URLs.
*   **Crawl Mode**: Recursively follows links from a starting URL, gathering content from multiple pages, including subdomains, while bypassing blockers.  **Note:** `crawl` mode accepts only ONE URL as a list.

### Parameters

*   **timeout (int, optional)**: Time in milliseconds (ms) to wait for a response. Default is `30000ms` (30 seconds). In crawl mode, this applies per URL.
*   **limit (int, optional)**: Maximum number of pages to scrape in crawl mode. Helps control API usage.
*   **params (dict, optional)**: Additional parameters to customize the request. See the [scrape API](https://docs.firecrawl.dev/api-reference/endpoint/scrape) and [crawl API](https://docs.firecrawl.dev/api-reference/endpoint/crawl-post) for details.

### Usage

#### Scrape Mode (Default)

Fetch content from multiple URLs:

```python
from langroid.parsing.url_loader import URLLoader
from langroid.crawlers import FirecrawlCrawler

loader = URLLoader(
    urls=[
        "https://pytorch.org",
        "https://www.tensorflow.org",
        "https://ai.google.dev/gemini-api/docs",
        "https://books.toscrape.com/"
    ],
    crawler=FirecrawlCrawler(
        timeout=15000,  # Timeout per request (15 sec)
        mode="scrape",
    )
)

docs = loader.load()
print(docs)
```

#### Crawl Mode

Fetch content from multiple pages starting from a single URL:

```python
from langroid.parsing.url_loader import URLLoader
from langroid.crawlers import FirecrawlCrawler

loader = URLLoader(
    urls=["https://books.toscrape.com/"],
    crawler=FirecrawlCrawler(
        timeout=10000,  # 10 sec per page
        mode="crawl",
        params={
            "limit": 5,
        }
    ),
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

`FirecrawlCrawler` benefits from Firecrawl's built-in document processing, which automatically extracts and structures content from web pages. This reduces the need for complex parsing logic within Langroid.

## Choosing a Crawler

*   Use `FirecrawlCrawler` when you need efficient, API-driven scraping with built-in document processing. This is often the simplest and most effective choice.
*   Use `TrafilaturaCrawler` when you want local non API based scraping (less accurate ) .
