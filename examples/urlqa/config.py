from dataclasses import dataclass, field
from examples.base_config import ExampleBaseConfig
from typing import List

@dataclass
class URLQAConfig(ExampleBaseConfig):
    chunk_size: int = 500
    chunk_overlap: int = 50
    chain_type: str = "stuff"
    urls: List[str] = field(default_factory=lambda: [
        "https://news.ycombinator.com/item?id=35629033",
        "https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web",
        "https://www.wired.com/1995/04/maes/",
        "https://cthiriet.com/articles/scaling-laws",
        "https://www.jasonwei.net/blog/emergence",
        "https://www.quantamagazine.org/the-unpredictable-abilities-emerging-from-large-ai-models-20230316/",
        "https://ai.googleblog.com/2022/11/characterizing-emergent-phenomena-in.html",
    ])


