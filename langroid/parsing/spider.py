from typing import List, Set, no_type_check
from urllib.parse import urlparse

from langroid.exceptions import LangroidImportError

try:
    from pydispatch import dispatcher
    from scrapy import signals
    from scrapy.crawler import CrawlerRunner
    from scrapy.http import Response
    from scrapy.linkextractors import LinkExtractor
    from scrapy.spiders import CrawlSpider, Rule
    from twisted.internet import defer, reactor
except ImportError:
    raise LangroidImportError("scrapy", "scrapy")


@no_type_check
class DomainSpecificSpider(CrawlSpider):  # type: ignore
    name = "domain_specific_spider"

    custom_settings = {"DEPTH_LIMIT": 1, "CLOSESPIDER_ITEMCOUNT": 20}

    rules = (Rule(LinkExtractor(), callback="parse_item", follow=True),)

    def __init__(self, start_url: str, k: int = 20, *args, **kwargs):  # type: ignore
        """Initialize the spider with start_url and k.

        Args:
            start_url (str): The starting URL.
            k (int, optional): The max desired final URLs. Defaults to 20.
        """
        super(DomainSpecificSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domains = [urlparse(start_url).netloc]
        self.k = k
        self.visited_urls: Set[str] = set()

    def parse_item(self, response: Response):  # type: ignore
        """Extracts URLs that are within the same domain.

        Args:
            response: The scrapy response object.
        """
        for link in LinkExtractor(allow_domains=self.allowed_domains).extract_links(
            response
        ):
            if len(self.visited_urls) < self.k:
                self.visited_urls.add(link.url)
                yield {"url": link.url}


@no_type_check
def scrapy_fetch_urls(url: str, k: int = 20) -> List[str]:
    """Fetches up to k URLs reachable from the input URL using Scrapy.

    Args:
        url (str): The starting URL.
        k (int, optional): The max desired final URLs. Defaults to 20.

    Returns:
        List[str]: List of URLs within the same domain as the input URL.
    """
    urls = []

    def _collect_urls(spider):
        """Handler for the spider_closed signal. Collects the visited URLs."""
        nonlocal urls
        urls.extend(list(spider.visited_urls))

    # Connect the spider_closed signal with our handler
    dispatcher.connect(_collect_urls, signal=signals.spider_closed)

    runner = CrawlerRunner(
        {
            "USER_AGENT": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        }
    )

    d = runner.crawl(DomainSpecificSpider, start_url=url, k=k)

    # Block until crawling is done and then stop the reactor
    crawl_deferred = defer.Deferred()

    def _crawl_done(_):
        reactor.stop()
        crawl_deferred.callback(urls)

    d.addBoth(_crawl_done)

    # Start the reactor, it will stop once the crawl is done
    reactor.run(installSignalHandlers=0)

    # This will block until the deferred gets a result
    return crawl_deferred.result


# Test the function
if __name__ == "__main__":
    fetched_urls = scrapy_fetch_urls("https://example.com", 5)
    for url in fetched_urls:
        print(url)
