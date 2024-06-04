import asyncio
from typing import List

import nltk
import pytest

from langroid.agent.batch import run_batch_tasks
from langroid.agent.special.relevance_extractor_agent import (
    RelevanceExtractorAgent,
    RelevanceExtractorAgentConfig,
)
from langroid.agent.task import Task
from langroid.agent.tools.segment_extract_tool import SegmentExtractTool
from langroid.parsing.utils import (
    clean_whitespace,
    extract_numbered_segments,
    number_segments,
    parse_number_range_list,
)
from langroid.utils.configuration import Settings, set_global
from langroid.utils.constants import NO_ANSWER


@pytest.mark.parametrize(
    "passage, query, expected",
    [
        (
            """
        Whales are big. 
        
        Cats like to be clean. They also like to be petted. And when they 
        are hungry they like to meow. Dogs are very friendly. They are also 
        very loyal. But so are cats. Unlike cats, dogs can get dirty.
        Monkeys are very naughty. They like to jump around. They also like to steal 
        bananas. 
        
        Cats are very independent. Unlike dogs, they like to be left 
        alone.
        """,
            "Characteristics of cats",
            "2-4,7,12-13",  # or LLM could say 2,3,4,7,12,10; we handle this below
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
        segment_length=1,
    )

    # directly send to llm and verify response is as expected
    extractor_agent = RelevanceExtractorAgent(agent_cfg)

    response = extractor_agent.llm_response(passage)
    tools = extractor_agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], SegmentExtractTool)
    assert set(parse_number_range_list(tools[0].segment_list)) == set(
        parse_number_range_list(expected)
    )

    # create task so that:
    # - llm generates sentence-list using SentenceExtractTool
    # - agent extracts sentences using SentenceExtractTool, says DONE
    extractor_agent = RelevanceExtractorAgent(agent_cfg)
    extractor_task = Task(
        extractor_agent,
        interactive=False,
    )

    result = extractor_task.run(passage)
    numbered_passage = number_segments(passage, granularity=agent_cfg.segment_length)
    expected_sentences = extract_numbered_segments(numbered_passage, expected)
    # the result should be the expected sentences, modulo whitespace
    result_sentences = [s.strip() for s in nltk.sent_tokenize(result.content)]
    expected_sentences = [s.strip() for s in nltk.sent_tokenize(expected_sentences)]
    assert set(result_sentences) == set(expected_sentences)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "passages, query, expected",
    [  # list of tuples
        (
            [
                "Whales are big.",
                """Cats like to be clean. They also like to be petted. And when they 
            are hungry they like to meow. Dogs are very friendly. They are also 
            very loyal. But so are cats. Unlike cats, dogs can get dirty.""",
                "Cats are very independent. Unlike dogs, they like to be left alone.",
            ],
            "Characteristics of cats",
            ["", "1-3,6", "1,2"],
        )
    ],
)
@pytest.mark.parametrize("fn_api", [True, False])
async def test_relevance_extractor_concurrent(
    test_settings: Settings,
    fn_api: bool,
    passages: List[str],
    query: str,
    expected: List[str],
) -> None:
    """
    Test concurrent extraction of relevant sentences from multiple passages.
    This is typically how we should use this extractor in a RAG pipeline.
    """
    set_global(test_settings)
    passages = [clean_whitespace(passage) for passage in passages]
    agent_cfg = RelevanceExtractorAgentConfig(
        use_tools=not fn_api,  # use tools if not fn_api
        use_functions_api=fn_api,
        query=query,
        segment_length=1,
    )
    agent_cfg.llm.stream = False  # disable streaming for concurrent calls

    # send to task.run_async and gather results
    async def _run_task(msg: str, i: int):
        # each invocation needs to create its own ChatAgent,
        # else the states gets mangled by concurrent calls!
        agent = RelevanceExtractorAgent(agent_cfg)
        task = Task(
            agent,
            name=f"Test-{i}",
            interactive=False,
        )
        return await task.run_async(msg=msg)

    # concurrent async calls to all tasks
    answers = await asyncio.gather(
        *(_run_task(passage, i) for i, passage in enumerate(passages))
    )
    assert len(answers) == len(passages)

    extracted_sentences = [
        s for a in answers for s in nltk.sent_tokenize(a.content) if s != NO_ANSWER
    ]
    expected_sentences = [
        s
        for passg, exp in zip(passages, expected)
        for s in nltk.sent_tokenize(
            extract_numbered_segments(
                number_segments(passg, granularity=agent_cfg.segment_length),
                exp,
            )
        )
        if s != ""
    ]

    expected_sentences = [s.strip() for s in expected_sentences]
    extracted_sentences = [s.strip() for s in extracted_sentences]
    assert set(extracted_sentences) == set(expected_sentences)


@pytest.mark.parametrize(
    "passages, query, expected",
    [  # list of tuples
        (
            [
                "Whales are big.",
                """Cats like to be clean. They also like to be petted. And when they 
                are hungry they like to meow. Dogs are very friendly. They are also 
                very loyal. But so are cats. Unlike cats, dogs can get dirty.""",
                "Cats are very independent. Unlike dogs, they like to be left alone.",
            ],
            "Characteristics of cats",
            ["", "1-3,6", "1,2"],
        )
    ],
)
@pytest.mark.parametrize("fn_api", [False])
def test_relevance_extractor_batch(
    test_settings: Settings,
    fn_api: bool,
    passages: List[str],
    query: str,
    expected: List[str],
) -> None:
    """
    Use `run_batch_tasks` to run the extractor on multiple passages.
    """

    set_global(test_settings)
    passages = [clean_whitespace(passage) for passage in passages]
    agent_cfg = RelevanceExtractorAgentConfig(
        use_tools=not fn_api,  # use tools if not fn_api
        use_functions_api=fn_api,
        query=query,
        segment_length=1,
    )
    agent_cfg.llm.stream = False  # disable streaming for concurrent calls

    agent = RelevanceExtractorAgent(agent_cfg)
    task = Task(
        agent,
        name="Test",
        interactive=False,
    )

    answers = run_batch_tasks(
        task,
        passages,
        input_map=lambda msg: msg,
        output_map=lambda ans: ans,
    )

    assert len(answers) == len(passages)

    extracted_sentences = [
        s for a in answers for s in nltk.sent_tokenize(a.content) if s != NO_ANSWER
    ]
    expected_sentences = [
        s
        for passg, exp in zip(passages, expected)
        for s in nltk.sent_tokenize(
            extract_numbered_segments(
                number_segments(passg, granularity=agent_cfg.segment_length),
                exp,
            )
        )
        if s != ""
    ]

    expected_sentences = [s.strip() for s in expected_sentences]
    extracted_sentences = [s.strip() for s in extracted_sentences]
    assert set(extracted_sentences) == set(expected_sentences)


@pytest.mark.parametrize(
    "passage, spec, expected",
    [
        (
            """
            <#1#> Whales are big. Dogs are very friendly. <#2#>They are also very 
            loyal.
            Buffaloes are very strong. 
            
            <#3#> They are also kind. But so are giraffes.
            
            <#10#> Cats like to be clean. They also like to be petted. And when they
            are hungry they like to meow. <#11#> Dogs are very friendly. They are also 
            very dirty. But not cats. Dogs bark.
            """,
            "2,3,11",
            "loyal,Buffaloes,kind,giraffes,Dogs,friendly,dirty,bark",
        )
    ],
)
def test_extract_numbered_segments(test_settings: Settings, passage, spec, expected):
    set_global(test_settings)
    extract = extract_numbered_segments(passage, spec)
    pieces = expected.split(",")
    assert all(piece.strip() in extract for piece in pieces)
