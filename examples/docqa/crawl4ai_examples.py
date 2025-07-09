import json
import os
from typing import Optional

from langroid.parsing.url_loader import Crawl4aiConfig, URLLoader
from crawl4ai.async_configs import LLMConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import (
    PruningContentFilter,
    LLMContentFilter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# Helper for pydantic models if LLMExtractionStrategy is used with schema
from langroid.pydantic_v1 import BaseModel, Field
from typing import List
from langroid.mytypes import Document


from langroid.parsing.url_loader import Crawl4aiConfig, URLLoader
from crawl4ai.async_configs import LLMConfig
from crawl4ai.extraction_strategy import (
    JsonCssExtractionStrategy,
    LLMExtractionStrategy,
)
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import (
    PruningContentFilter,
    LLMContentFilter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy

# Helper for pydantic models if LLMExtractionStrategy is used with schema
from langroid.pydantic_v1 import BaseModel, Field
from typing import List
from langroid.mytypes import Document


from rich.console import Console
from rich.prompt import IntPrompt

console = Console()

import sys
import os

sys.path.append(os.path.dirname(__file__))

from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig


def simple_crawler_example():
    """
    Demonstrates a basic crawl using Crawl4aiConfig with default settings.
    It will fetch the markdown content of the given URLs.
    """
    print("\n--- Running simple_crawler_example ---")
    urls = [
        "https://pytorch.org",
        "https://arxiv.org/pdf/1706.03762",  # This will be handled by DocumentParser
    ]
    crawler_config = Crawl4aiConfig()  # Uses default BrowserConfig and CrawlerRunConfig
    loader = URLLoader(urls=urls, crawler_config=crawler_config)

    docs = loader.load()
    for doc in docs:
        print(
            f"URL: {doc.metadata.source}, Content Length: {len(doc.content)} (first 200 chars: {doc.content[:200]})"
        )
    print("--- simple_crawler_example finished ---")


def extract_to_json_example():
    """
    Demonstrates how to use `JsonCssExtractionStrategy` to extract structured JSON
    from a webpage, configured via `Crawl4aiConfig`.
    """
    print("\n--- Running extract_to_json_example ---")
    HACKER_NEWS_URL = "https://news.ycombinator.com"
    HACKER_NEWS_SCHEMA = {
        "name": "HackerNewsArticles",
        "baseSelector": "tr.athing",  # Each article is in a <tr> with class 'athing'
        "fields": [
            {"name": "title", "selector": "span.titleline > a", "type": "text"},
            {
                "name": "link",
                "selector": "span.titleline > a",
                "type": "attribute",
                "attribute": "href",
            },
        ],
    }

    css_strategy = JsonCssExtractionStrategy(schema=HACKER_NEWS_SCHEMA)

    hn_crawler_config = Crawl4aiConfig(extraction_strategy=css_strategy)

    print(f"Starting scrape of {HACKER_NEWS_URL}...")
    loader = URLLoader(urls=[HACKER_NEWS_URL], crawler_config=hn_crawler_config)
    documents = loader.load()

    if documents:
        print("\nScrape successful! Processing extracted data...")
        extracted_json_string = documents[0].content
        try:
            extracted_data = json.loads(extracted_json_string)
            print("\n--- Top 3 Articles from Hacker News ---")
            for i, item in enumerate(extracted_data[:3], 1):
                print(f"{i}. Title: {item.get('title')}")
                print(f"   Link: {item.get('link')}")
            print(f"\nTotal items extracted: {len(extracted_data)}")
        except json.JSONDecodeError:
            print("Error: Failed to parse the extracted content as JSON.")
            print("Received content:", extracted_json_string)
    else:
        print("\nScrape failed. No documents were returned.")
    print("--- extract_to_json_example finished ---")


def markdown_generation_example():
    """
    Demonstrates customizing markdown generation using `markdown_strategy` in Crawl4aiConfig.
    Uses PruningContentFilter for focused content.
    """
    print("\n--- Running markdown_generation_example ---")
    url = "https://news.ycombinator.com"

    # Define a content filter to prune irrelevant sections
    prune_filter = PruningContentFilter(
        threshold=0.6,  # More aggressive pruning
        threshold_type="dynamic",
        min_word_threshold=10,
    )

    # Configure the markdown generator to use the filter and ignore links
    md_generator = DefaultMarkdownGenerator(
        content_filter=prune_filter,
        options={
            "ignore_links": True,
            "body_width": 100,  # Wrap text at 100 characters
            "citations": False,  # Disable citations
        },
    )

    crawler_config = Crawl4aiConfig(markdown_strategy=md_generator)

    loader = URLLoader(urls=[url], crawler_config=crawler_config)
    docs = loader.load()

    if docs:
        print(f"Markdown Content (first 500 chars) for {url}:")
        # In this setup, the 'content' of the Document will be the fit_markdown
        print(docs[0].content[:500])
        print(f"Original URL: {docs[0].metadata.source}")
    else:
        print(f"Failed to crawl {url}.")
    print("--- markdown_generation_example finished ---")


def deep_crawl_example():
    """Crawl multiple pages from a domain using BFS strategy."""
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import (
        FilterChain,
        URLPatternFilter,
        DomainFilter,
        ContentTypeFilter,
    )
    from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig

    # Create browser config
    browser_config = BrowserConfig(
        # Example browser config settings
        headless=True,
        viewport={"width": 1920, "height": 1080},
    )

    # Create filter chain
    filter_chain = FilterChain(
        [
            URLPatternFilter(patterns=["*core*"]),
            DomainFilter(
                allowed_domains=["docs.crawl4ai.com"],
            ),
            ContentTypeFilter(allowed_types=["text/html"]),
        ]
    )

    # Create deep crawl strategy
    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=2, include_external=False, max_pages=5, filter_chain=filter_chain
    )

    # Create run config
    run_config = CrawlerRunConfig(
        # Example run config settings
        deep_crawl_strategy=deep_crawl_strategy,
    )

    # Create the Crawl4ai configuration with all components
    crawler_config = Crawl4aiConfig(
        crawl_mode="deep", browser_config=browser_config, run_config=run_config
    )

    url = "https://docs.crawl4ai.com/"

    loader = URLLoader(urls=[url], crawler_config=crawler_config)

    docs = loader.load()

    if docs:
        print(f"Total Documents: {len(docs)}")
        for i, doc in enumerate(docs[:5], 1):
            print(f"{i}. {doc.metadata.source} ({len(doc.content)} chars)")
    else:
        print("No documents crawled.")


def scraping_strategy_example():
    """
    Demonstrates using a custom `scraping_strategy` (e.g., LXMLWebScrapingStrategy)
    in Crawl4aiConfig for potentially faster HTML parsing.
    """
    print("\n--- Running scraping_strategy_example ---")
    url = "https://www.nbcnews.com/business"

    # Use LXMLWebScrapingStrategy for potentially faster scraping
    scraping_strategy = LXMLWebScrapingStrategy()

    crawler_config = Crawl4aiConfig(scraping_strategy=scraping_strategy)

    print(f"Starting crawl of {url} with LXML scraping strategy...")
    loader = URLLoader(urls=[url], crawler_config=crawler_config)
    docs = loader.load()

    if docs:
        print(f"Crawl successful! Content Length for {url}: {len(docs[0].content)}")
        print(f"First 200 chars of content:\n{docs[0].content[:200]}")
    else:
        print(f"Failed to crawl {url}.")
    print("--- scraping_strategy_example finished ---")


def llm_extraction_example():
    """
    Demonstrates using LLMExtractionStrategy to extract structured data
    using an LLM, configured via Crawl4aiConfig.
    Requires GEMINI_API_KEY environment variable to be set.
    """
    print("\n--- Running llm_extraction_example ---")

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not found. Skipping llm_extraction_example.")
        print("Please set the GEMINI_API_KEY environment variable to run this example.")
        return

    class ArticleData(BaseModel):
        headline: str
        summary: str = Field(description="A short summary of the article")
        author: Optional[str] = None

    url = "https://news.ycombinator.com"

    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            # Corrected Gemini model name based on your provided documentation
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMINI_API_KEY"),
        ),
        schema=ArticleData.schema_json(),
        extraction_type="schema",
        instruction="Extract the headline and a short summary for the main article on the page. If author is available, extract it too.",
        # Small chunk_token_threshold for demo purposes, adjust as needed for full pages
        chunk_token_threshold=1000,
        apply_chunking=True,
        input_format="markdown",  # Can be "html", "fit_markdown"
    )

    crawler_config = Crawl4aiConfig(extraction_strategy=llm_strategy)

    print(f"Starting LLM-based extraction from {url}...")
    loader = URLLoader(urls=[url], crawler_config=crawler_config)
    docs: List[Document] = loader.load()  # Explicitly type hint for clarity

    # The output structure is `[Document(...)]` because URLLoader wraps the result.
    # The actual extracted JSON is in `docs[0].content`.
    print(
        f"Raw documents loaded: {docs}"
    )  # This will show the `Document` object structure

    if docs:
        print("\nLLM Extraction successful!")
        extracted_content = docs[0].content
        try:
            # LLM extraction returns JSON string in `content`
            extracted_data = json.loads(extracted_content)
            print("Extracted Data:", json.dumps(extracted_data, indent=2))
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM output JSON: {e}")
            print("Raw LLM output:", extracted_content)
    else:
        print(f"LLM extraction from {url} failed or returned no data.")
    print("--- llm_extraction_example finished ---")


def regex_extraction_example():
    """
    Demonstrates using RegexExtractionStrategy to extract URLs, emails, and dates
    from a webpage, configured via Crawl4aiConfig.
    """
    from langroid.parsing.url_loader import Crawl4aiConfig, URLLoader
    from crawl4ai.extraction_strategy import RegexExtractionStrategy
    from langroid.mytypes import Document
    import json

    print("\n--- Running regex_extraction_example ---")

    # Pick a real-world page that likely has email, URL, or date patterns
    url = "https://www.scrapethissite.com/pages/forms/"

    # Combine multiple regex types
    regex_strategy = RegexExtractionStrategy(
        pattern=(
            RegexExtractionStrategy.Email
            | RegexExtractionStrategy.Url
            | RegexExtractionStrategy.DateUS
        ),
    )

    crawler_config = Crawl4aiConfig(extraction_strategy=regex_strategy)

    print(f"Crawling and extracting from: {url}")
    loader = URLLoader(urls=[url], crawler_config=crawler_config)
    docs = loader.load()

    if not docs:
        print("No documents returned.")
        return

    try:
        extracted_json = json.loads(docs[0].content)
        if not isinstance(extracted_json, list) or not extracted_json:
            print("No structured matches found.")
            return

        print(f"Found {len(extracted_json)} matches:")
        for i, item in enumerate(extracted_json[:10], start=1):  # Show top 10
            label = item.get("label", "unknown")
            value = item.get("value", "")
            print(f"  {i}. [{label}] {value}")
    except json.JSONDecodeError:
        print("Failed to parse content as JSON.")
        print("Raw content:")
        print(docs[0].content)

    print("--- regex_extraction_example finished ---")


def llm_content_filter_example():
    """
    Demonstrates using LLMContentFilter within DefaultMarkdownGenerator
    to intelligently filter and format content.
    Requires OPENAI_API_KEY environment variable.
    """
    print("\n--- Running llm_content_filter_example ---")

    if not os.getenv("GEMINI_API_KEY"):
        print("GEMINI_API_KEY not found. Skipping llm_extraction_example.")
        print("Please set the GEMINI_API_KEY environment variable to run this example.")
        return

    url = "https://news.ycombinator.com"  # A page with varied content

    llm_filter = LLMContentFilter(
        llm_config=LLMConfig(
            provider="gemini/gemini-2.0-flash",
            api_token=os.getenv("GEMINI_API_KEY"),
        ),
        instruction="""
        Focus on extracting the core news headlines and summaries.
        Include:
        - Main headlines
        - Brief summaries of the linked articles (if visible on the page)
        Exclude:
        - Navigation elements, sidebars, footer content
        - Comments sections
        Format the output as clean markdown with proper code blocks and headers if applicable.
        """,
        chunk_token_threshold=2048,  # Adjust for performance/cost
        verbose=False,  # Set to True for detailed LLM logs
    )

    md_generator = DefaultMarkdownGenerator(content_filter=llm_filter)

    crawler_config = Crawl4aiConfig(markdown_strategy=md_generator)

    print(f"Starting crawl of {url} with LLM content filter...")
    loader = URLLoader(urls=[url], crawler_config=crawler_config)
    docs = loader.load()

    if docs:
        print("\nLLM Content Filter successful!")
        # The content of the Document will be the `fit_markdown` from the LLM filter
        print("Filtered Markdown (first 1000 chars):")
        print(docs[0].content[:1000])
    else:
        print(f"LLM content filter crawl from {url} failed or returned no data.")
    print("--- llm_content_filter_example finished ---")


example_functions = {
    1: ("Simple Crawl Example", simple_crawler_example),
    2: ("JSON Extraction via CSS Selectors", extract_to_json_example),
    3: ("Custom Markdown Generation", markdown_generation_example),
    4: ("Deep Crawl with BFS", deep_crawl_example),
    5: ("LXML Scraping Strategy", scraping_strategy_example),
    6: ("LLM-based Extraction", llm_extraction_example),
    7: ("Regex Extraction", regex_extraction_example),
    8: ("LLM Content Filter in Markdown", llm_content_filter_example),
}


def main_menu():
    console.rule("[bold green]Crawl4ai Example Menu")
    for i, (name, _) in example_functions.items():
        console.print(f"[cyan]{i}.[/cyan] {name}")
    console.print("[magenta]0.[/magenta] Exit")

    while True:
        try:
            choice = IntPrompt.ask("\nChoose an example to run", default=0)

            for i, (name, _) in example_functions.items():
                console.print(f"[cyan]{i}.[/cyan] {name}")
            if choice == 0:
                console.print("[bold red]Exiting. Goodbye![/bold red]")
                break
            elif choice in example_functions:
                console.rule(f"[bold yellow]Running: {example_functions[choice][0]}")
                example_functions[choice][1]()  # Run selected function
                console.print("\n[green] Finished.[/green]\n")
            else:
                console.print("[red]Invalid choice. Try again.[/red]")
        except KeyboardInterrupt:
            console.print("\n[bold red]Interrupted by user. Exiting.[/bold red]")
            break


if __name__ == "__main__":
    main_menu()
