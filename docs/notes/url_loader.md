# Firecrawl and Trafilatura Crawlers Documentation

`URLLoader` uses `Trafilatura` if not explicitly specified

## Overview
*   **`FirecrawlCrawler`**:  Leverages the Firecrawl API for efficient web scraping and crawling. 
It offers built-in document processing capabilities.
Requires `FIRECRAWL_API_KEY` environment variable to be set in `.env` file or environment.
*   **`TrafilaturaCrawler`**: Utilizes the Trafilatura library and Langroid's parsing tools 
for extracting and processing web content - this is the default crawler, and 
does not require setting up an external API key.
*  **`ExaCrawler`**: Integrates with the Exa API for high-quality content extraction.
  Requires `EXA_API_KEY` environment variable to be set in `.env` file or environment.

## Installation

`TrafilaturaCrawler` comes with Langroid

To use `FirecrawlCrawler`, install the `firecrawl` extra:

```bash
pip install langroid[firecrawl]
```

## Exa Crawler Documentation

### Overview

`ExaCrawler` integrates with Exa API to extract high-quality content from web pages. It provides efficient content extraction with the simplicity of API-based processing.

### Parameters

Obtain an Exa API key from [Exa](https://exa.ai/) and set it in your environment variables, e.g. in your `.env` file as:

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
* High-quality content extraction with clean text output -- this is, however, 
  plain text and not in markdown format like Firecrawl provides.
* Efficient handling of complex web pages
* No need for additional parsing as Exa handles document processing

## Trafilatura Crawler Documentation

### Overview

`TrafilaturaCrawler` is a web crawler that uses the Trafilatura library for content extraction 
and Langroid's parsing capabilities for further processing. 
This crawler is useful when you need more control over the parsing process and 
want to leverage Langroid's document processing tools.

### Parameters

*   **config (TrafilaturaConfig)**: A `TrafilaturaConfig` object that defines how to process the extracted text. 
    *   `threads` (int): The number of threads to use for downloading web pages.

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
The `Parser` uses the default parsing methods or with a configuration that can be adjust to more suit the current use case.

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
    timeout=10000,  # 10 sec per page
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

`FirecrawlCrawler` benefits from Firecrawl's built-in document processing, which automatically extracts and structures content from web pages (including pdf,doc,docx). 
This reduces the need for complex parsing logic within Langroid.

## Choosing a Crawler

*   Use `FirecrawlCrawler` when you need efficient, API-driven scraping with built-in document processing. 
This is often the simplest and most effective choice.
*   Use `TrafilaturaCrawler` when you want local non API based scraping (less accurate ) .

## Example script

See the script [`examples/docqa/chat_search.py`](https://github.com/langroid/langroid/blob/main/examples/docqa/chat_search.py) 
which shows how to use a Langroid agent to search the web and scrape URLs to answer questions.