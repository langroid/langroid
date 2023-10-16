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
    parse_number_range_list,
)
from langroid.utils.configuration import Settings, set_global


@pytest.mark.parametrize(
    "prompt, expected",
    [
        (
            """
        PASSAGE:
        (1) Whales are big. 
        
        (2) Cats like to be clean. (3) They also like to be petted. (4) And when they 
        are hungry they like to meow. (5) Dogs are very friendly. (6) They are also 
        very loyal. (7) But so are cats. (8) Unlike cats, dogs can get dirty.
        
        (9) Cats are also very independent. (10) Unlike dogs, they like 
        to be left alone.
        
        QUERY: What do we know about cats?
        """,
            "2-4,7,9,10",  # or LLM could say 2,3,4,7,9,10; we handle this below
        )
    ],
)
def test_relevance_extractor_agent(
    test_settings: Settings,
    prompt: str,
    expected: str,
) -> None:
    set_global(test_settings)
    prompt = clean_whitespace(prompt)

    # directly send to llm and verify response is as expected
    extractor_agent = RelevanceExtractorAgent(RelevanceExtractorAgentConfig())

    response = extractor_agent.llm_response(prompt)
    tools = extractor_agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], SentenceExtractTool)
    assert set(parse_number_range_list(tools[0].sentence_list)) == set(
        parse_number_range_list(expected)
    )

    # create task so that:
    # - llm generates sentence-list using SentenceExtractTool
    # - agent extracts sentences using SentenceExtractTool, says DONE
    extractor_agent = RelevanceExtractorAgent(RelevanceExtractorAgentConfig())
    extractor_task = Task(
        extractor_agent,
        default_human_response="",  # eliminate human response
        only_user_quits_root=False,  # allow agent_response to quit via "DONE <msg>"
    )

    result = extractor_task.run(prompt)
    expected_sentences = extract_numbered_sentences(prompt, expected)
    # the result should be the expected sentences, modulo whitespace
    assert set(nltk.sent_tokenize(result.content)) == set(
        nltk.sent_tokenize(expected_sentences)
    )
