import logging
import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import markdownify as md
from dotenv import load_dotenv

from langroid.exceptions import LangroidImportError
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.document_parser import DocumentParser, ImagePdfParser
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.pydantic_v1 import BaseSettings

if TYPE_CHECKING:
    from firecrawl import FirecrawlApp

load_dotenv()

logging.getLogger("url_loader").setLevel(logging.WARNING)


# Base crawler config and specific configurations
class BaseCrawlerConfig(BaseSettings):
    """Base configuration for web crawlers."""

    parser: Optional[Parser] = None


class TrafilaturaConfig(BaseCrawlerConfig):
    """Configuration for Trafilatura crawler."""

    threads: int = 4
    format: str = "markdown"  # or "xml" or "txt"


class FirecrawlConfig(BaseCrawlerConfig):
    """Configuration for Firecrawl crawler."""

    api_key: str = ""
    mode: str = "scrape"
    params: Dict[str, Any] = {}
    timeout: Optional[int] = None

    class Config:
        # Leverage Pydantic's BaseSettings to
        # allow setting of fields via env vars,
        # e.g. FIRECRAWL_MODE=scrape and FIRECRAWL_API_KEY=...
        env_prefix = "FIRECRAWL_"


class ExaCrawlerConfig(BaseCrawlerConfig):
    api_key: str = ""

    class Config:
        # Allow setting of fields via env vars with prefix EXA_
        # e.g., EXA_API_KEY=your_api_key
        env_prefix = "EXA_"


class BaseCrawler(ABC):
    """Abstract base class for web crawlers."""

    def __init__(self, config: BaseCrawlerConfig):
        """Initialize the base crawler.

        Args:
            config: Configuration for the crawler
        """
        self.parser = config.parser if self.needs_parser else None
        self.config: BaseCrawlerConfig = config

    @property
    @abstractmethod
    def needs_parser(self) -> bool:
        """Indicates whether the crawler requires a parser."""
        pass

    @abstractmethod
    def crawl(self, urls: List[str]) -> List[Document]:
        pass

    def _process_document(self, url: str) -> List[Document]:
        if self.parser:
            import requests
            from requests.structures import CaseInsensitiveDict

            if self._is_document_url(url):
                try:
                    doc_parser = DocumentParser.create(url, self.parser.config)
                    new_chunks = doc_parser.get_doc_chunks()
                    if not new_chunks:
                        # If the document is empty, try to extract images
                        img_parser = ImagePdfParser(url, self.parser.config)
                        new_chunks = img_parser.get_doc_chunks()
                    return new_chunks
                except Exception as e:
                    logging.error(f"Error parsing {url}: {e}")
                    return []

            else:
                try:
                    headers = requests.head(url).headers
                except Exception as e:
                    logging.warning(f"Error getting headers for {url}: {e}")
                    headers = CaseInsensitiveDict()

                content_type = headers.get("Content-Type", "").lower()
                temp_file_suffix = None
                if "application/pdf" in content_type:
                    temp_file_suffix = ".pdf"
                elif (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    in content_type
                ):
                    temp_file_suffix = ".docx"
                elif "application/msword" in content_type:
                    temp_file_suffix = ".doc"

                if temp_file_suffix:
                    try:
                        response = requests.get(url)
                        with NamedTemporaryFile(
                            delete=False, suffix=temp_file_suffix
                        ) as temp_file:
                            temp_file.write(response.content)
                            temp_file_path = temp_file.name
                        doc_parser = DocumentParser.create(
                            temp_file_path, self.parser.config
                        )
                        docs = doc_parser.get_doc_chunks()
                        os.remove(temp_file_path)
                        return docs
                    except Exception as e:
                        logging.error(f"Error downloading/parsing {url}: {e}")
                        return []
        return []

    def _is_document_url(self, url: str) -> bool:
        return any(url.lower().endswith(ext) for ext in [".pdf", ".docx", ".doc"])


class CrawlerFactory:
    """Factory for creating web crawlers."""

    @staticmethod
    def create_crawler(config: BaseCrawlerConfig) -> BaseCrawler:
        """Create a crawler instance based on configuration type.

        Args:
            config: Configuration for the crawler

        Returns:
            A BaseCrawler instance

        Raises:
            ValueError: If config type is not supported
        """
        if isinstance(config, TrafilaturaConfig):
            return TrafilaturaCrawler(config)
        elif isinstance(config, FirecrawlConfig):
            return FirecrawlCrawler(config)
        elif isinstance(config, ExaCrawlerConfig):
            return ExaCrawler(config)
        else:
            raise ValueError(f"Unsupported crawler configuration type: {type(config)}")


class TrafilaturaCrawler(BaseCrawler):
    """Crawler implementation using Trafilatura."""

    def __init__(self, config: TrafilaturaConfig):
        """Initialize the Trafilatura crawler.

        Args:
            config: Configuration for the crawler
        """
        super().__init__(config)
        self.config: TrafilaturaConfig = config

    @property
    def needs_parser(self) -> bool:
        return True

    def crawl(self, urls: List[str]) -> List[Document]:
        import trafilatura
        from trafilatura.downloads import (
            add_to_compressed_dict,
            buffered_downloads,
            load_download_buffer,
        )

        docs = []
        dl_dict = add_to_compressed_dict(urls)

        while not dl_dict.done:
            buffer, dl_dict = load_download_buffer(dl_dict, sleep_time=5)
            for url, result in buffered_downloads(buffer, self.config.threads):
                parsed_doc = self._process_document(url)
                if parsed_doc:
                    docs.extend(parsed_doc)
                else:
                    text = trafilatura.extract(
                        result,
                        no_fallback=False,
                        favor_recall=True,
                        include_formatting=True,
                        output_format=self.config.format,
                        with_metadata=True,  # Title, date, author... at start of text
                    )
                    if self.config.format in ["xml", "html"]:
                        # heading_style="ATX" for markdown headings, i.e. #, ##, etc.
                        text = md.markdownify(text, heading_style="ATX")
                    if text is None and result is not None and isinstance(result, str):
                        text = result
                    if text:
                        docs.append(
                            Document(content=text, metadata=DocMetaData(source=url))
                        )

        return docs


class FirecrawlCrawler(BaseCrawler):
    """Crawler implementation using Firecrawl."""

    def __init__(self, config: FirecrawlConfig) -> None:
        """Initialize the Firecrawl crawler.

        Args:
            config: Configuration for the crawler
        """
        super().__init__(config)
        self.config: FirecrawlConfig = config

    @property
    def needs_parser(self) -> bool:
        return False

    def _return_save_incremental_results(
        self, app: "FirecrawlApp", crawl_id: str, output_dir: str = "firecrawl_output"
    ) -> List[Document]:
        # Code used verbatim from firecrawl blog with few modifications
        # https://www.firecrawl.dev/blog/mastering-the-crawl-endpoint-in-firecrawl
        import json
        import time
        from pathlib import Path

        from tqdm import tqdm

        pbar = tqdm(desc="Pages saved", unit=" pages", dynamic_ncols=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        processed_urls: set[str] = set()
        docs = []

        while True:
            # Check current status
            status = app.check_crawl_status(crawl_id)
            new_pages = 0

            # Save new pages
            for page in status["data"]:
                url = page["metadata"]["url"]
                if url not in processed_urls:
                    content = page.get("markdown", "")
                    filename = f"{output_dir}/{len(processed_urls)}.md"
                    with open(filename, "w") as f:
                        f.write(content)
                    docs.append(
                        Document(
                            content=content,
                            metadata=DocMetaData(
                                source=url,
                                title=page["metadata"].get("title", "Unknown Title"),
                            ),
                        )
                    )
                    processed_urls.add(url)
                    new_pages += 1
            pbar.update(new_pages)  # Update progress bar with new pages

            # Break if crawl is complete
            if status["status"] == "completed":
                print(f"Saved {len(processed_urls)} pages.")
                with open(f"{output_dir}/full_results.json", "w") as f:
                    json.dump(status, f, indent=2)
                break

            time.sleep(5)  # Wait before checking again
        return docs

    def crawl(self, urls: List[str]) -> List[Document]:
        try:
            from firecrawl import FirecrawlApp
        except ImportError:
            raise LangroidImportError("firecrawl", "firecrawl")

        app = FirecrawlApp(api_key=self.config.api_key)
        docs = []
        params = self.config.params.copy()  # Create a copy of the existing params

        if self.config.timeout is not None:
            params["timeout"] = self.config.timeout  # Add/override timeout in params

        if self.config.mode == "scrape":
            for url in urls:
                try:
                    result = app.scrape_url(url, params=params)
                    metadata = result.get(
                        "metadata", {}
                    )  # Default to empty dict if missing
                    status_code = metadata.get("statusCode")

                    if status_code == 200:
                        docs.append(
                            Document(
                                content=result["markdown"],
                                metadata=DocMetaData(
                                    source=url,
                                    title=metadata.get("title", "Unknown Title"),
                                ),
                            )
                        )
                except Exception as e:
                    logging.warning(
                        f"Firecrawl encountered an error for {url}: {e}. "
                        "Skipping but continuing."
                    )
        elif self.config.mode == "crawl":
            if not isinstance(urls, list) or len(urls) != 1:
                raise ValueError(
                    "Crawl mode expects 'urls' to be a list containing a single URL."
                )

            # Start the crawl
            crawl_status = app.async_crawl_url(url=urls[0], params=params)

            # Save results incrementally
            docs = self._return_save_incremental_results(app, crawl_status["id"])
        return docs


class ExaCrawler(BaseCrawler):
    """Crawler implementation using Exa API."""

    def __init__(self, config: ExaCrawlerConfig) -> None:
        """Initialize the Exa crawler.

        Args:
            config: Configuration for the crawler
        """
        super().__init__(config)
        self.config: ExaCrawlerConfig = config

    @property
    def needs_parser(self) -> bool:
        return True

    def crawl(self, urls: List[str]) -> List[Document]:
        """Crawl the given URLs using Exa SDK.

        Args:
            urls: List of URLs to crawl

        Returns:
            List of Documents with content extracted from the URLs

        Raises:
            LangroidImportError: If the exa package is not installed
            ValueError: If the Exa API key is not set
        """
        try:
            from exa_py import Exa
        except ImportError:
            raise LangroidImportError("exa", "exa")

        if not self.config.api_key:
            raise ValueError("EXA_API_KEY key is required in your env or .env")

        exa = Exa(self.config.api_key)
        docs = []

        try:
            for url in urls:
                parsed_doc_chunks = self._process_document(url)
                if parsed_doc_chunks:
                    docs.extend(parsed_doc_chunks)
                    continue
                else:
                    results = exa.get_contents(
                        [url],
                        livecrawl="always",
                        text={
                            "include_html_tags": True,
                        },
                    )
                    result = results.results[0]
                    if result.text:
                        md_text = md.markdownify(result.text, heading_style="ATX")
                        # append a NON-chunked document
                        # (metadata.is_chunk = False, so will be chunked downstream)
                        docs.append(
                            Document(
                                content=md_text,
                                metadata=DocMetaData(
                                    source=url,
                                    title=getattr(result, "title", "Unknown Title"),
                                    published_date=getattr(
                                        result, "published_date", "Unknown Date"
                                    ),
                                ),
                            )
                        )

        except Exception as e:
            logging.error(f"Error retrieving content from Exa API: {e}")

        return docs


class URLLoader:
    """Loads URLs and extracts text using a specified crawler."""

    def __init__(
        self,
        urls: List[Any],
        parsing_config: ParsingConfig = ParsingConfig(),
        crawler_config: Optional[BaseCrawlerConfig] = None,
    ):
        """Initialize the URL loader.

        Args:
            urls: List of URLs to load
            parsing_config: Configuration for parsing
            crawler_config: Configuration for the crawler
        """
        self.urls = urls
        self.parsing_config = parsing_config

        if crawler_config is None:
            crawler_config = TrafilaturaConfig(parser=Parser(parsing_config))

        self.crawler = CrawlerFactory.create_crawler(crawler_config)
        if self.crawler.needs_parser:
            self.crawler.parser = Parser(parsing_config)

    def load(self) -> List[Document]:
        """Load the URLs using the specified crawler."""
        return self.crawler.crawl(self.urls)
