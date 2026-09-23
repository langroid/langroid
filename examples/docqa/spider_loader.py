"""Load Markdown with Spider Cloud, without an LLM or vector database.

Set SPIDER_API_KEY in the environment or .env, then run:
    python examples/docqa/spider_loader.py https://example.com
    python examples/docqa/spider_loader.py https://example.com --crawl --limit 3
"""

import argparse

from langroid.parsing.url_loader import SpiderConfig, URLLoader


def main() -> None:
    """Load the supplied URLs and print their sources and content."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urls", nargs="+", help="Page or seed URLs")
    parser.add_argument("--crawl", action="store_true", help="Follow links")
    parser.add_argument("--limit", type=int, default=1, help="Pages per crawl seed")
    args = parser.parse_args()
    config = SpiderConfig(mode="crawl" if args.crawl else "scrape", limit=args.limit)
    for document in URLLoader(args.urls, crawler_config=config).load():
        print(f"Source: {document.metadata.source}\n{document.content}\n")


if __name__ == "__main__":
    main()
