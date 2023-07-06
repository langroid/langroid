from examples.audience.agents.segmentor import SegmentorConfig, Segmentor
from llmagent.utils.configuration import Settings, set_global
from llmagent.vector_store.qdrantdb import QdrantDBConfig
from llmagent.embedding_models.models import OpenAIEmbeddingsConfig
from llmagent.vector_store.base import VectorStoreConfig
from llmagent.language_models.openai_gpt import OpenAIGPTConfig, OpenAIChatModel
from llmagent.parsing.parser import ParsingConfig, Splitter
from llmagent.prompts.prompts_config import PromptsConfig
from llmagent.cachedb.redis_cachedb import RedisCacheConfig
from llmagent.utils.system import rmdir

import os
import warnings
import pytest
import tempfile


storage_path = ".qdrant/testdata1"
rmdir(storage_path)


class _TestSegmentorConfig(SegmentorConfig):
    filename: str = None  # filled in below
    n_matches: int = 2
    debug: bool = False
    stream: bool = True  # allow streaming where needed
    conversation_mode = True
    vecdb: VectorStoreConfig = QdrantDBConfig(
        type="qdrant",
        collection_name="test-iab-taxonomy",
        storage_path=storage_path,
        embedding=OpenAIEmbeddingsConfig(
            model_type="openai",
            model_name="text-embedding-ada-002",
            dims=1536,
        ),
    )

    llm: OpenAIGPTConfig = OpenAIGPTConfig(
        stream=True,
        cache_config=RedisCacheConfig(fake=False),
        chat_model=OpenAIChatModel.GPT4,
        use_chat_for_completion=True,
    )

    parsing: ParsingConfig = ParsingConfig(
        splitter=Splitter.TOKENS,
        chunk_size=500,
        n_similar_docs=2,
    )

    prompts: PromptsConfig = PromptsConfig(
        max_tokens=1000,
    )


set_global(Settings(cache=True))  # allow cacheing


@pytest.fixture
def segmentor_agent():
    file_content = """Unique ID,Parent ID,"Condensed Name (1st, 2nd, Last Tier)",Tier 1,Tier 2,Tier 3,Tier 4,Tier 5,Tier 6,*Extension Notes
797,782,Purchase Intent* | Arts and Entertainment |  Zoos and Aquariums | ,Purchase Intent*,Arts and Entertainment,Experiences and Events,Zoos and Aquariums,,,"See ""Purchase Intent Classification"" Extension"
314,305,Interest | Business and Finance |  Information Services Industry | ,Interest,Business and Finance,Industries,Information Services Industry,,,
22,20,Demographic | Education & Occupation |  Postgraduate Education | ,Demographic,Education & Occupation,Education (Highest Level),College Education,Postgraduate Education,,
1119,1115,Purchase Intent* | Consumer Packaged Goods |  Other Snacks  | ,Purchase Intent*,Consumer Packaged Goods,Edible,General Food,Snacks,Other Snacks ,"See ""Purchase Intent Classification"" Extension"
158,156,Demographic | Language |  *Language Extension | ,Demographic,Language,Other,*Language Extension,,,See ISO-639-1-alpha-2
1614,1600,Purchase Intent* | Software |  Productivity Software | ,Purchase Intent*,Software,Computer Software,Productivity Software,,,"See ""Purchase Intent Classification"" Extension"
583,206,Interest | Real Estate,Interest,Real Estate,,,,,
688,687,Interest | Technology & Computing |  Artificial Intelligence | ,Interest,Technology & Computing,Artificial Intelligence,,,, 
395,383,Interest | Health and Medical Services |  Chiropractors | ,Interest,Health and Medical Services,Health & Pharma,Medical Services,Chiropractors,,
74,73,Demographic | Household Data |  Less Than 1 Year | ,Demographic,Household Data,Length of Residence,Less Than 1 Year,,,
322,305,Interest | Business and Finance |  Metals Industry | ,Interest,Business and Finance,Industries,Metals Industry,,,
739,738,Interest | Video Gaming |  Action Video Games | ,Interest,Video Gaming,Video Game Genres,Action Video Games,,,
1248,1219,Purchase Intent* | Consumer Packaged Goods |  Miscellaneous General Merch | ,Purchase Intent*,Consumer Packaged Goods,Non-edible,General Merchandise,Miscellaneous General Merch,,"See ""Purchase Intent Classification"" Extension"
775,753,Purchase Intent* | Apps |  Sports Apps | ,Purchase Intent*,Apps,Sports Apps,,,,"See ""Purchase Intent Classification"" Extension"
162,159,Demographic | Marital Status |  Single | ,Demographic,Marital Status,Single,,,,
897,871,Purchase Intent* | Business and Industrial |  Printing/Fax/WiFi Services | ,Purchase Intent*,Business and Industrial,Printing/Fax/WiFi Services,,,,"See ""Purchase Intent Classification"" Extension"
"""
    with tempfile.NamedTemporaryFile("w+t", delete=False) as temp_file:
        temp_file_name = temp_file.name
        temp_file.write(file_content)
    config = _TestSegmentorConfig(filename=temp_file_name)
    agent = Segmentor(config=config)

    yield agent

    # Teardown: remove the temporary file
    os.remove(temp_file_name)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


warnings.filterwarnings(
    "ignore",
    message="Token indices sequence length.*",
    # category=UserWarning,
    module="transformers",
)

QUERY_EXPECTED_PAIRS = [
    # (english_description, words expected in retrieved taxonomy names)
    ("video games", "gaming"),
    ("in market to visit a zoo", "intent,zoo"),
    ("people looking to buy productivity software", "intent,productivity"),
    ("language learners", "language"),
]


@pytest.mark.parametrize("query, expected", QUERY_EXPECTED_PAIRS)
def test_segmentor(test_settings: Settings, segmentor_agent, query: str, expected: str):
    set_global(test_settings)
    nearest_docs = segmentor_agent.get_nearest_docs(query)
    expected = [e.strip() for e in expected.split(",")]
    # at least one of the nearest docs has all the expected words
    assert any(
        [all([e in doc.content.lower() for e in expected]) for doc in nearest_docs]
    )
    ans = segmentor_agent.llm_response(query).content

    assert all([e in ans.lower() for e in expected])
