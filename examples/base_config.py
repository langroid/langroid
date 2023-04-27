from dataclasses import dataclass, field
from typing import List

@dataclass
class IndexConfig:
    '''
    Vector database (index) configuration
    '''
    type: str = "qdrant"
    compose_file: str = "llmagent/vector_store/docker-compose-qdrant.yml"

@dataclass
class ExampleBaseConfig:
    debug: bool = False
    index: IndexConfig = field(default_factory=IndexConfig)
    urls: List[str] = field(default_factory=lambda: [
        "https://www.understandingwar.org/backgrounder/russian-offensive"
        "-campaign-assessment-february-8-2023",
        "https://www.understandingwar.org/backgrounder/russian-offensive-campaign"
        "-assessment-february-9-2023",
    ])


