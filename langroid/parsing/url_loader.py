import logging
import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from dotenv import load_dotenv

from langroid.exceptions import LangroidImportError
from langroid.mytypes import DocMetaData, Document
from langroid.parsing.document_parser import DocumentParser, ImagePdfParser
from langroid.parsing.parser import Parser, ParsingConfig

if TYPE_CHECKING:
    from firecrawl import FirecrawlApp

load_dotenv()

logging.getLogger("url_loader").setLevel(logging.WARNING)


class BaseCrawler(ABC):
    """Abstract base class for web crawlers."""

    def __init__(self, parser: Optional[Parser] = None):
        self.parser = parser if self.needs_parser else None

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


class TrafilaturaCrawler(BaseCrawler):
    """Crawler implementation using Trafilatura."""

    def __init__(self, parser: Parser, threads: int = 4):
        super().__init__(parser)
        self.threads = threads

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
            for url, result in buffered_downloads(buffer, self.threads):
                parsed_doc = self._process_document(url)
                if parsed_doc:
                    docs.extend(parsed_doc)
                else:
                    text = trafilatura.extract(
                        result, no_fallback=False, favor_recall=True
                    )
                    if text:
                        docs.append(
                            Document(content=text, metadata=DocMetaData(source=url))
                        )

        return docs


class FirecrawlCrawler(BaseCrawler):
    """Crawler implementation using Firecrawl."""

    def __init__(
        self,
        mode: Optional[str] = "scrape",
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        super().__init__(parser=None)
        self.mode = mode
        self.params = params or {}  # Store the params, default to empty dict
        self.timeout = timeout

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
                        Document(content=content, metadata=DocMetaData(source=url))
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

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        app = FirecrawlApp(api_key=api_key)
        docs = []
        params = self.params.copy()  # Create a copy of the existing params

        if self.timeout is not None:
            params["timeout"] = self.timeout  # Add/override timeout in params

        if self.mode == "scrape":
            for url in urls:
                result = app.scrape_url(
                    url, params=params
                )  # Pass the params dictionary
                if result["metadata"]["statusCode"] == 200:
                    docs.append(
                        Document(
                            content=result["markdown"], metadata=DocMetaData(source=url)
                        )
                    )
                else:
                    logging.warning(
                        f"Firecrawl exited with error {result} "
                        f" while retrieving content from {url}"
                    )
        elif self.mode == "crawl":
            if not isinstance(urls, list) or len(urls) != 1:
                raise ValueError(
                    "Crawl mode expects 'urls' to be a list containing a single URL."
                )

            # Start the crawl
            crawl_status = app.async_crawl_url(url=urls[0], params=params)

            # Save results incrementally
            docs = self._return_save_incremental_results(app, crawl_status["id"])
        return docs


class URLLoader:
    """Loads URLs and extracts text using a specified crawler."""

    def __init__(
        self,
        urls: List[str],
        parsing_config: ParsingConfig = ParsingConfig(),
        crawler: Optional[BaseCrawler] = None,
    ):
        self.urls = urls
        self.parsing_config = parsing_config
        self.crawler = crawler or TrafilaturaCrawler(Parser(parsing_config), threads=4)

    def load(self) -> List[Document]:
        """Load the URLs using the specified crawler."""
        return self.crawler.crawl(self.urls)


if __name__ == "__main__":
    loader = URLLoader(
        urls=[
            # "https://pytorch.org",
            # "https://www.tensorflow.org",
            # "https://ai.google.dev/gemini-api/docs",
            # "https://books.toscrape.com/"
        ],
        crawler=FirecrawlCrawler(
            mode="crawl",
            params={
                "limit": 4,
            },
        ),
    )

    docs = loader.load()
    print(docs)
