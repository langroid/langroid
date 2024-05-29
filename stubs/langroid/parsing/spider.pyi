from _typeshed import Incomplete
from scrapy.http import Response as Response
from scrapy.spiders import CrawlSpider

class DomainSpecificSpider(CrawlSpider):
    name: str
    custom_settings: Incomplete
    rules: Incomplete
    start_urls: Incomplete
    allowed_domains: Incomplete
    k: Incomplete
    visited_urls: Incomplete
    def __init__(self, start_url: str, k: int = 20, *args, **kwargs) -> None: ...
    def parse_item(self, response: Response): ...

def scrapy_fetch_urls(url, k: int = 20): ...
