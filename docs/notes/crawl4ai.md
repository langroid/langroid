# Crawl4ai Crawler Documentation

## Overview

The `Crawl4aiCrawler` is a highly advanced and flexible web crawler integrated into Langroid, built on the powerful `crawl4ai` library. It uses a real browser engine (Playwright) to render web pages, making it exceptionally effective at handling modern, JavaScript-heavy websites. This crawler provides a rich set of features for simple page scraping, deep-site crawling, and sophisticated data extraction, making it the most powerful crawling option available in Langroid.

It is local crawler, so no need for API keys.

## Installation

To use `Crawl4aiCrawler`, you must install the `crawl-4-ai` extra dependencies.

This installs and tests for crawl4ai setup readiness.

```bash
# Install langroid with crawl4ai support
pip install "langroid[crawl-4-ai]"
crawl4ai setup
crawl4ai doctor

```

## Key Features

- **Real Browser Rendering**: Accurately processes dynamic content, single-page applications (SPAs), and sites that require JavaScript execution.

- **Simple and Deep Crawling**: Can scrape a list of individual URLs (`simple` mode) or perform a recursive, deep crawl of a website starting from a seed URL (`deep` mode).

- **Powerful Extraction Strategies**:

  - **Structured JSON (No LLM)**: Extract data into a predefined JSON structure using CSS selectors, XPath, or Regex patterns. This is extremely fast, reliable, and cost-effective.

  - **LLM-Based Extraction**: Leverage Large Language Models (like GPT or Gemini) to extract data from unstructured content based on natural language instructions and a Pydantic schema.

- **Advanced Markdown Generation**: Go beyond basic HTML-to-markdown conversion. Apply content filters to prune irrelevant sections (sidebars, ads, footers) or use an LLM to intelligently reformat content for maximum relevance, perfect for RAG pipelines.

- **High-Performance Scraping**: Optionally use an LXML-based scraping strategy for a significant speed boost on large HTML documents.

- **Fine-Grained Configuration**: Offers detailed control over browser behavior (`BrowserConfig`) and individual crawl runs (`CrawlerRunConfig`) for advanced use cases.

## Configuration (`Crawl4aiConfig`)

The `Crawl4aiCrawler` is configured via the `Crawl4aiConfig` object. This class acts as a high-level interface to the underlying `crawl4ai` library's settings.

All of the strategies are optional.
Learn more about these strategies , browser_config and run_config at [Crawl4AI docs](https://docs.crawl4ai.com/)

```python
from langroid.parsing.url_loader import Crawl4aiConfig

# All parameters are optional and have sensible defaults
config = Crawl4aiConfig(
    crawl_mode="simple",  # or "deep"
    extraction_strategy=...,
    markdown_strategy=...,
    deep_crawl_strategy=...,
    scraping_strategy=...,
    browser_config=...,  # For advanced browser settings
    run_config=...,      # For advanced crawl-run settings
)
```

**Main Parameters:**

- `crawl_mode` (str):

  - `"simple"` (default): Crawls each URL in the provided list individually.

  - `"deep"`: Starts from the first URL in the list and recursively crawls linked pages based on the `deep_crawl_strategy`.

  - Make sure you are setting `"crawl_mode=deep"` whenever you are deep crawling this is crucial for smooth functioning.

- `extraction_strategy` (`ExtractionStrategy`): Defines how to extract structured data from a page. If set, the `Document.content` will be a **JSON string** containing the extracted data.

- `markdown_strategy` (`MarkdownGenerationStrategy`): Defines how to convert HTML to markdown. This is used when `extraction_strategy` is not set. The `Document.content` will be a **markdown string**.

- `deep_crawl_strategy` (`DeepCrawlStrategy`): Configuration for deep crawling, such as `max_depth`, `max_pages`, and URL filters. Only used when `crawl_mode` is `"deep"`.

- `scraping_strategy` (`ContentScrapingStrategy`): Specifies the underlying HTML parsing engine. Useful for performance tuning.

- `browser_config` & `run_config`: For advanced users to pass detailed `BrowserConfig` and `CrawlerRunConfig` objects directly from the `crawl-4-ai` library.

---

## Usage Examples

These are representative example for runnable examples check the script [`examples/url_loader/cralw4ai_examples.py`](https://github.com/langroid/langroid/blob/main/examples/url_loader/cralw4ai_examples.py)

### 1. Simple Crawling (Default Markdown)

This is the most basic usage. It will fetch the content of each URL and convert it to clean markdown.

```python
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig

urls = [
    "https://pytorch.org/blog/introducing-torchrecipes/",
    "https://techcrunch.com/2024/03/04/anthropic-releases-claude-3/",
]

# Use default settings
crawler_config = Crawl4aiConfig()
loader = URLLoader(urls=urls, crawler_config=crawler_config)

docs = loader.load()
for doc in docs:
    print(f"URL: {doc.metadata.source}")
    print(f"Content (first 200 chars): {doc.content[:200]}")
```

### 2. Structured JSON Extraction (No LLM)

When you need to extract specific, repeated data fields from a page, schema-based extraction is the best choice. It's fast, precise, and free of LLM costs. The result in `Document.content` is a JSON string.

#### a. Using CSS Selectors (`JsonCssExtractionStrategy`)

This example scrapes titles and links from the Hacker News front page.

```python
import json
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

HACKER_NEWS_URL = "https://news.ycombinator.com"
HACKER_NEWS_SCHEMA = {
    "name": "HackerNewsArticles",
    "baseSelector": "tr.athing",
    "fields": [
        {"name": "title", "selector": "span.titleline > a", "type": "text"},
        {"name": "link", "selector": "span.titleline > a", "type": "attribute", "attribute": "href"},
    ],
}

# Create the strategy and pass it to the config
css_strategy = JsonCssExtractionStrategy(schema=HACKER_NEWS_SCHEMA)
crawler_config = Crawl4aiConfig(extraction_strategy=css_strategy)

loader = URLLoader(urls=[HACKER_NEWS_URL], crawler_config=crawler_config)
documents = loader.load()

# The Document.content will contain the JSON string
extracted_data = json.loads(documents[0].content)
print(json.dumps(extracted_data[:3], indent=2))
```

#### b. Using Regex (`RegexExtractionStrategy`)

This is ideal for finding common patterns like emails, URLs, or phone numbers.

```python
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.extraction_strategy import RegexExtractionStrategy

url = "https://www.scrapethissite.com/pages/forms/"

# Combine multiple built-in patterns
regex_strategy = RegexExtractionStrategy(
    pattern=(
        RegexExtractionStrategy.Email
        | RegexExtractionStrategy.Url
        | RegexExtractionStrategy.PhoneUS
    )
)

crawler_config = Crawl4aiConfig(extraction_strategy=regex_strategy)
loader = URLLoader(urls=[url], crawler_config=crawler_config)
documents = loader.load()

print(documents[0].content)
```

### 3. Advanced Markdown Generation

For RAG applications, the quality of the markdown is crucial. These strategies produce highly relevant, clean text. The result in `Document.content` is the filtered markdown (`fit_markdown`).

#### a. Pruning Filter (`PruningContentFilter`)

This filter heuristically removes boilerplate content based on text density, link density, and common noisy tags.

```python
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

prune_filter = PruningContentFilter(threshold=0.6, min_word_threshold=10)
md_generator = DefaultMarkdownGenerator(
    content_filter=prune_filter,
    options={"ignore_links": True}
)

crawler_config = Crawl4aiConfig(markdown_strategy=md_generator)
loader = URLLoader(urls=["https://news.ycombinator.com"], crawler_config=crawler_config)
docs = loader.load()

print(docs[0].content[:500])
```

#### b. LLM Filter (`LLMContentFilter`)

Use an LLM to semantically understand the content and extract only the relevant parts based on your instructions. This is extremely powerful for creating topic-focused documents.

```python
import os
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.async_configs import LLMConfig
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import LLMContentFilter

# Requires an API key, e.g., OPENAI_API_KEY
llm_filter = LLMContentFilter(
    llm_config=LLMConfig(
        provider="openai/gpt-4o-mini",
        api_token=os.getenv("OPENAI_API_KEY"),
    ),
    instruction="""
    Extract only the main article content.
    Exclude all navigation, sidebars, comments, and footer content.
    Format the output as clean, readable markdown.
    """,
    chunk_token_threshold=4096,
)

md_generator = DefaultMarkdownGenerator(content_filter=llm_filter)
crawler_config = Crawl4aiConfig(markdown_strategy=md_generator)
loader = URLLoader(urls=["https://www.theverge.com/tech"], crawler_config=crawler_config)
docs = loader.load()

print(docs[0].content)
```

### 4. Deep Crawling

To crawl an entire website or a specific section, use `deep` mode.

```python
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter

# Crawl up to 2 levels deep, for a max of 5 pages, containing 'core' in the URL
deep_crawl_strategy = BFSDeepCrawlStrategy(
    max_depth=2,
    max_pages=5,
    filter_chain=FilterChain([URLPatternFilter(patterns=["*core*"])])
)

crawler_config = Crawl4aiConfig(
    crawl_mode="deep",
    deep_crawl_strategy=deep_crawl_strategy
)

loader = URLLoader(urls=["https://docs.crawl4ai.com/"], crawler_config=crawler_config)
docs = loader.load()

print(f"Crawled {len(docs)} pages.")
for doc in docs:
    print(f"- {doc.metadata.source}")
```

### 5. High-Performance Scraping (`LXMLWebScrapingStrategy`)

For a performance boost, especially on very large, static HTML pages, switch the scraping strategy to LXML.

```python
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

crawler_config = Crawl4aiConfig(
    scraping_strategy=LXMLWebScrapingStrategy()
)

loader = URLLoader(urls=["https://www.nbcnews.com/business"], crawler_config=crawler_config)
docs = loader.load()
print(f"Content Length: {len(docs[0].content)}")
```

### 6. LLM-Based JSON Extraction (`LLMExtractionStrategy`)

When data is unstructured or requires semantic interpretation, use an LLM for extraction. This is slower and more expensive but incredibly flexible. The result in `Document.content` is a JSON string.

```python
import os
import json
from langroid.pydantic_v1 import BaseModel, Field
from typing import Optional
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig
from crawl4ai.async_configs import LLMConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

# Define the data structure you want to extract
class ArticleData(BaseModel):
    headline: str
    summary: str = Field(description="A short summary of the article")
    author: Optional[str] = None

# Configure the LLM strategy
llm_strategy = LLMExtractionStrategy(
    llm_config=LLMConfig(
        provider="openai/gpt-4o-mini",
        api_token=os.getenv("OPENAI_API_KEY"),
    ),
    schema=ArticleData.schema_json(),
    extraction_type="schema",
    instruction="Extract the headline, summary, and author of the main article.",
)

crawler_config = Crawl4aiConfig(extraction_strategy=llm_strategy)
loader = URLLoader(urls=["https://news.ycombinator.com"], crawler_config=crawler_config)
docs = loader.load()

extracted_data = json.loads(docs[0].content)
print(json.dumps(extracted_data, indent=2))
```

## How It Handles Different Content Types

The `Crawl4aiCrawler` is smart about handling different types of URLs:

- **Web Pages** (e.g., `http://...`, `https://...`): These are processed by the `crawl4ai` browser engine. The output format (`markdown` or `JSON`) depends on the strategy you configure in `Crawl4aiConfig`.
- **Local and Remote Documents** (e.g., URLs ending in `.pdf`, `.docx`): These are automatically detected and delegated to Langroid's internal `DocumentParser`. This ensures that documents are properly parsed and chunked according to your `ParsingConfig`, just like with other Langroid tools.

## Conclusion

The `Crawl4aiCrawler` is a feature-rich, powerful tool for any web-based data extraction task.

- For **simple, clean text**, use the default `Crawl4aiConfig`.

- For **structured data from consistent sites**, use `JsonCssExtractionStrategy` or `RegexExtractionStrategy` for unbeatable speed and reliability.

- To create **high-quality, focused content for RAG**, use `PruningContentFilter` or the `LLMContentFilter` with the `DefaultMarkdownGenerator`.

- To scrape an **entire website**, use `deep_crawl_strategy` with `crawl_mode="deep"`.

- For **complex or unstructured data** that needs AI interpretation, `LLMExtractionStrategy` provides a flexible solution.
