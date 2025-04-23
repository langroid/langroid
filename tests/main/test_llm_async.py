import asyncio
import os

import pytest

import langroid.language_models as lm
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import (
    OpenAICompletionModel,
    OpenAIGPT,
    OpenAIGPTConfig,
)
from langroid.parsing.file_attachment import FileAttachment
from langroid.utils.configuration import Settings, set_global, settings

# allow streaming globally, but can be turned off by individual models
set_global(Settings(stream=True))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "streaming, country, capital",
    [(True, "France", "Paris"), (False, "India", "Delhi")],
)
@pytest.mark.parametrize("stream_quiet", [True, False])
async def test_openai_gpt_async(
    test_settings: Settings,
    streaming,
    country,
    capital,
    stream_quiet,
):
    set_global(test_settings)
    cfg = OpenAIGPTConfig(
        stream=streaming,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        completion_model=OpenAICompletionModel.DAVINCI,
        cache_config=RedisCacheConfig(fake=False),
        async_stream_quiet=stream_quiet,
    )

    mdl = OpenAIGPT(config=cfg)
    question = "What is the capital of " + country + "?"

    set_global(Settings(cache=False))
    # chat mode via `generate`,
    # i.e. use same call as for completion, but the setting below
    # actually calls `achat` under the hood
    cfg.use_chat_for_completion = True
    # check that "agenerate" works
    response = await mdl.agenerate(prompt=question, max_tokens=50)
    assert capital in response.message
    assert not response.cached

    # actual chat mode
    messages = [
        LLMMessage(
            role=Role.SYSTEM,
            content="You are a serious, helpful assistant. Be very concise, not funny",
        ),
        LLMMessage(role=Role.USER, content=question),
    ]
    response = await mdl.achat(messages=messages, max_tokens=50)
    assert capital in response.message
    assert not response.cached

    set_global(Settings(cache=True))
    # should be from cache this time
    response = await mdl.achat(messages=messages, max_tokens=50)
    assert capital in response.message
    assert response.cached

    # pass intentional bad msg to test error handling
    if not test_settings.chat_model.startswith("litellm-proxy/"):
        messages = [
            LLMMessage(
                role=Role.FUNCTION,
                content="Hello!",
            ),
        ]

        with pytest.raises(Exception):
            await mdl.achat(messages=messages, max_tokens=50)


@pytest.mark.asyncio
async def test_llm_async_concurrent(test_settings: Settings):
    set_global(test_settings)
    cfg = OpenAIGPTConfig(
        stream=False,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        completion_model=OpenAICompletionModel.DAVINCI,
        cache_config=RedisCacheConfig(fake=False),
    )

    mdl = OpenAIGPT(config=cfg)
    N = 5
    questions = ["1+" + str(i) for i in range(N)]
    expected_answers = [str(i + 1) for i in range(N)]
    answers = await asyncio.gather(
        *(mdl.agenerate(prompt=question, max_tokens=50) for question in questions)
    )

    assert len(answers) == len(questions)
    for e in expected_answers:
        assert any(e in a.message for a in answers)

    answers = await asyncio.gather(
        *(mdl.achat(question, max_tokens=50) for question in questions)
    )
    assert len(answers) == len(questions)
    for e in expected_answers:
        assert any(e in a.message for a in answers)


@pytest.mark.asyncio
@pytest.mark.xfail(
    reason="LangDB may fail due to unknown flakiness!",
    run=True,
    strict=False,
)
@pytest.mark.parametrize(
    "model",
    [
        "langdb/gpt-4o-mini",
        "langdb/openai/gpt-4o-mini",
        "langdb/anthropic/claude-3-haiku-20240307",
        "langdb/claude-3-haiku-20240307",
        "langdb/gemini/gemini-2.0-flash-lite",
        "langdb/gemini-2.0-flash-lite",
    ],
)
async def test_llm_langdb(model: str):
    """Test that LLM access via LangDB works."""

    llm_config_langdb = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config_langdb)
    result = await llm.achat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model",
    [
        "openrouter/anthropic/claude-3.5-haiku-20241022:beta",
        "openrouter/mistralai/mistral-small-24b-instruct-2501:free",
        "openrouter/google/gemini-2.0-flash-lite-001",
    ],
)
async def test_llm_openrouter(model: str):
    # override models set via pytest ... --m <model>
    settings.chat_model = model
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config)
    result = await llm.achat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_llm_pdf_attachment_async():
    """Test sending a PDF file attachment to the LLM asynchronously."""
    from pathlib import Path

    # Path to the test PDF file
    pdf_path = Path("tests/main/data/dummy.pdf")

    # Create a FileAttachment from the PDF file
    attachment = FileAttachment.from_path(pdf_path)

    # Verify the attachment properties
    assert attachment.mime_type == "application/pdf"
    assert attachment.filename == "dummy.pdf"

    # Create messages with the attachment
    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER, content="What's title of the paper?", files=[attachment]
        ),
    ]

    # Set up the LLM with a suitable model that supports PDFs
    llm = OpenAIGPT(OpenAIGPTConfig(max_output_tokens=1000))

    # Get response from the LLM asynchronously
    response = await llm.achat(messages=messages)

    assert response is not None
    assert response.message is not None
    assert "Supply Chain" in response.message

    # follow-up question
    messages += [
        LLMMessage(role=Role.ASSISTANT, content="Supply Chain"),
        LLMMessage(role=Role.USER, content="Who is the first author?"),
    ]
    response = await llm.achat(messages=messages)
    assert response is not None
    assert response.message is not None
    assert "Takio" in response.message


@pytest.mark.xfail(
    reason="Multi-file attachment may not work yet.",
    run=True,
    strict=False,
)
@pytest.mark.asyncio
async def test_llm_multi_pdf_attachment_async():
    from pathlib import Path

    # Path to the test PDF file
    pdf_path = Path("tests/main/data/dummy.pdf")

    # Create a FileAttachment from the PDF file
    attachment = FileAttachment.from_path(pdf_path)

    # multiple attachments
    pdf_path2 = Path("tests/main/data/sample-test.pdf")

    # Create a FileAttachment from the PDF file
    attachment2 = FileAttachment.from_path(pdf_path2)

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content="How many pages are in the Supply Chain paper?",
            files=[attachment2, attachment],
        ),
    ]
    llm = OpenAIGPT(OpenAIGPTConfig(max_output_tokens=1000))
    response = await llm.achat(messages=messages)
    assert any(x in response.message for x in ["4", "four"])

    # follow-up question
    messages += [
        LLMMessage(role=Role.ASSISTANT, content="4 pages"),
        LLMMessage(
            role=Role.USER,
            content="""
            How many columns are in the table in the 
            document that is NOT about Supply Chain?
            """,
        ),
    ]
    response = await llm.achat(messages=messages)
    try:
        assert any(x in response.message for x in ["3", "three"])
    except AssertionError:
        pytest.xfail("Multi-files don't work yet?", strict=False)


@pytest.mark.asyncio
async def test_litellm_model_key_async():
    """
    Test that passing in explicit api_key works with `litellm/*` models
    """
    model = "litellm/anthropic/claude-3-5-haiku-latest"
    # disable any chat model passed via --m arg to pytest cmd
    settings.chat_model = model
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model, api_key=os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Create the LLM instance
    llm = lm.OpenAIGPT(config=llm_config)
    print(f"\nTesting with model: {llm.chat_model_orig} => {llm.config.chat_model}")
    response = await llm.achat("What is 3+4?")
    assert "7" in response.message
