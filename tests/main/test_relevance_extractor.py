import nltk
import pytest

from langroid.agent.special.relevance_extractor_agent import (
    RelevanceExtractorAgent,
    RelevanceExtractorAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.tools.sentence_extract_tool import SentenceExtractTool
from langroid.parsing.utils import (
    clean_whitespace,
    extract_numbered_sentences,
    number_sentences,
    parse_number_range_list,
)
from langroid.utils.configuration import Settings, set_global


@pytest.mark.parametrize(
    "passage, query, expected",
    [
        (
            """
        Whales are big. 
        
        Cats like to be clean. They also like to be petted. And when they 
        are hungry they like to meow. Dogs are very friendly. They are also 
        very loyal. But so are cats. Unlike cats, dogs can get dirty.
        
        Cats are also very independent. Unlike dogs, they like to be left alone.
        """,
            "What do we know about cats?",
            "2-4,7,9,10",  # or LLM could say 2,3,4,7,9,10; we handle this below
        )
    ],
)
@pytest.mark.parametrize("fn_api", [True, False])
def test_relevance_extractor_agent(
    test_settings: Settings,
    fn_api: bool,
    passage: str,
    query: str,
    expected: str,
) -> None:
    set_global(test_settings)
    passage = clean_whitespace(passage)
    agent_cfg = RelevanceExtractorAgentConfig(
        use_tools=not fn_api,  # use tools if not fn_api
        use_functions_api=fn_api,
        query=query,
    )

    # directly send to llm and verify response is as expected
    extractor_agent = RelevanceExtractorAgent(agent_cfg)

    response = extractor_agent.llm_response(passage)
    tools = extractor_agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], SentenceExtractTool)
    assert set(parse_number_range_list(tools[0].sentence_list)) == set(
        parse_number_range_list(expected)
    )

    # create task so that:
    # - llm generates sentence-list using SentenceExtractTool
    # - agent extracts sentences using SentenceExtractTool, says DONE
    extractor_agent = RelevanceExtractorAgent(agent_cfg)
    extractor_task = Task(
        extractor_agent,
        default_human_response="",  # eliminate human response
        only_user_quits_root=False,  # allow agent_response to quit via "DONE <msg>"
    )

    result = extractor_task.run(passage)
    numbered_passage = number_sentences(passage)
    expected_sentences = extract_numbered_sentences(numbered_passage, expected)
    # the result should be the expected sentences, modulo whitespace
    assert set(nltk.sent_tokenize(result.content)) == set(
        nltk.sent_tokenize(expected_sentences)
    )
