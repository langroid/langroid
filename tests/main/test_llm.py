import io
import os
import random
import warnings
from pathlib import Path

import fitz  # PyMuPDF
import openai
import pytest

import langroid as lr
import langroid.language_models as lm
from langroid.cachedb.redis_cachedb import RedisCacheConfig
from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_gpt import (
    AccessWarning,
    OpenAIChatModel,
    OpenAICompletionModel,
    OpenAIGPT,
    OpenAIGPTConfig,
)
from langroid.parsing.file_attachment import FileAttachment
from langroid.parsing.parser import Parser, ParsingConfig
from langroid.parsing.utils import generate_random_sentences
from langroid.utils.configuration import Settings, set_global, settings

# allow streaming globally, but can be turned off by individual models
set_global(Settings(stream=True))


@pytest.mark.parametrize(
    "streaming, country, capital",
    [(False, "India", "Delhi"), (True, "France", "Paris")],
)
@pytest.mark.parametrize("use_cache", [True, False])
def test_openai_gpt(test_settings: Settings, streaming, country, capital, use_cache):
    test_settings.cache = False  # cache response but don't retrieve from cache
    set_global(test_settings)

    cfg = OpenAIGPTConfig(
        stream=streaming,  # use streaming output if enabled globally
        type="openai",
        max_output_tokens=100,
        min_output_tokens=10,
        completion_model=OpenAICompletionModel.DAVINCI,
        cache_config=RedisCacheConfig(fake=True) if use_cache else None,
    )

    mdl = OpenAIGPT(config=cfg)
    question = "What is the capital of " + country + "?"
    # chat mode via `generate`,
    # i.e. use same call as for completion, but the setting below
    # actually calls `chat` under the hood
    cfg.use_chat_for_completion = True
    # check that "generate" works when "use_chat_for_completion" is True
    response = mdl.generate(prompt=question, max_tokens=800)
    assert response.usage is not None and response.usage.total_tokens > 0
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
    response = mdl.chat(messages=messages, max_tokens=500)
    assert response.usage is not None and response.usage.total_tokens > 0
    assert capital in response.message
    assert not response.cached

    test_settings.cache = True
    set_global(test_settings)
    # should be from cache this time, Provided config.cache_config is not None
    response = mdl.chat(messages=messages, max_tokens=500)
    assert response.usage is not None
    if use_cache:
        response.usage.total_tokens == 0
    else:
        response.usage.total_tokens > 0

    assert capital in response.message
    assert response.cached == use_cache

    # pass intentional bad msg to test error handling
    messages = [
        LLMMessage(
            role=Role.FUNCTION,
            content="Hello!",
        ),
    ]

    with pytest.raises(Exception):
        _ = mdl.chat(messages=messages, max_tokens=500)


@pytest.mark.parametrize(
    "mode, max_tokens",
    [("completion", 100), ("chat", 100), ("completion", 1000), ("chat", 1000)],
)
def _test_context_length_error(test_settings: Settings, mode: str, max_tokens: int):
    """
    Test disabled, see TODO below.
    Also it takes too long since we are trying to test
    that it raises the expected error when the context length is exceeded.
    Args:
        test_settings: from conftest.py
        mode: "completion" or "chat"
        max_tokens: number of tokens to generate
    """
    set_global(test_settings)
    set_global(Settings(cache=False))

    cfg = OpenAIGPTConfig(
        stream=False,
        max_output_tokens=max_tokens,
        completion_model=OpenAICompletionModel.TEXT_DA_VINCI_003,
        cache_config=RedisCacheConfig(fake=False),
    )
    parser = Parser(config=ParsingConfig())
    llm = OpenAIGPT(config=cfg)
    context_length = (
        llm.chat_context_length() if mode == "chat" else llm.completion_context_length()
    )

    toks_per_sentence = int(parser.num_tokens(generate_random_sentences(1000)) / 1000)
    max_sentences = int(context_length * 1.5 / toks_per_sentence)
    big_message = generate_random_sentences(max_sentences + 1)
    big_message_tokens = parser.num_tokens(big_message)
    assert big_message_tokens + max_tokens > context_length
    response = None
    # TODO need to figure out what error type to expect here
    with pytest.raises(openai.BadRequestError) as e:
        if mode == "chat":
            response = llm.chat(big_message, max_tokens=max_tokens)
        else:
            response = llm.generate(prompt=big_message, max_tokens=max_tokens)

    assert response is None
    assert "context length" in str(e.value).lower()


@pytest.mark.parametrize(
    "mdl",
    [
        lm.OpenAIChatModel.GPT4o,
        lm.GeminiModel.GEMINI_2_PRO,
        "gemini/" + lm.GeminiModel.GEMINI_2_PRO.value,
    ],
)
@pytest.mark.parametrize("ctx", [16_000, None])
def test_llm_config_context_length(mdl: str, ctx: int | None):
    llm_config = lm.OpenAIGPTConfig(
        chat_model=mdl,
        chat_context_length=ctx,  # even if wrong, use if explicitly set
    )
    mdl = lm.OpenAIGPT(config=llm_config)
    assert mdl.chat_context_length() == ctx or mdl.info().context_length


def test_model_selection(test_settings: Settings):
    set_global(test_settings)

    defaultOpenAIChatModel = lr.language_models.openai_gpt.default_openai_chat_model

    def get_response(llm):
        llm.generate(prompt="What is the capital of France?", max_tokens=50)

    def simulate_response(llm):
        llm.run_on_first_use()

    def check_warning(
        llm,
        assert_warn,
        function=get_response,
        warning_type=AccessWarning,
        catch_errors=(ImportError,),
    ):
        if assert_warn:
            with pytest.warns(expected_warning=warning_type):
                try:
                    function(llm)
                except catch_errors:
                    pass
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", category=warning_type)

                try:
                    function(llm)
                except catch_errors:
                    pass

    # Default is GPT4o; we should not generate the warning in this case
    lr.language_models.openai_gpt.default_openai_chat_model = OpenAIChatModel.GPT4_TURBO
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT3_5_TURBO))
    check_warning(llm, False)

    llm = OpenAIGPT(config=OpenAIGPTConfig())
    check_warning(llm, False)

    # Default is GPT3.5 (simulate GPT 4 inaccessible)
    lr.language_models.openai_gpt.default_openai_chat_model = (
        OpenAIChatModel.GPT3_5_TURBO
    )

    # No warnings generated if we specify the model explicitly
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model=OpenAIChatModel.GPT3_5_TURBO))
    check_warning(llm, False)

    # No warnings generated if we are using a local model
    llm = OpenAIGPT(config=OpenAIGPTConfig(api_base="localhost:8000"))
    check_warning(llm, False, function=simulate_response)
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model="local/localhost:8000"))
    check_warning(llm, False, function=simulate_response)
    llm = OpenAIGPT(config=OpenAIGPTConfig(chat_model="litellm/ollama/llama"))
    check_warning(llm, False, function=simulate_response)

    # We should warn on the first usage of a model with auto-selected GPT-3.5
    llm = OpenAIGPT(config=OpenAIGPTConfig())
    check_warning(llm, True)

    # We should not warn on subsequent uses and models with auto-selected GPT-3.5
    check_warning(llm, False)
    llm = OpenAIGPT(config=OpenAIGPTConfig())
    check_warning(llm, False)

    lr.language_models.openai_gpt.default_openai_chat_model = defaultOpenAIChatModel


def test_keys():
    # Do not override the explicit settings below
    settings.chat_model = ""

    providers = [
        "vllm",
        "ollama",
        "llamacpp",
        "openai",
        "groq",
        "gemini",
        "glhf",
        "openrouter",
        "aimlapi",
        "deepseek",
        "cerebras",
    ]
    key_dict = {p: f"{p.upper()}_API_KEY" for p in providers}
    key_dict["llamacpp"] = "LLAMA_API_KEY"
    key_dict["aimlapi"] = "AIML_API_KEY"

    for p, var in key_dict.items():
        os.environ[var] = p

    for p in providers:
        config = lm.OpenAIGPTConfig(
            chat_model=f"{p}/model",
        )

        llm = lm.OpenAIGPT(config)

        assert llm.api_key == p

        rand_key = str(random.randint(0, 10**9))
        config = lm.OpenAIGPTConfig(
            chat_model=f"{p}/model",
            api_key=rand_key,
        )

        llm = lm.OpenAIGPT(config)
        assert llm.api_key == rand_key


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
def test_llm_langdb(model: str):
    """Test that LLM access via LangDB works."""
    # override any chat model passed via --m arg to pytest cmd
    settings.chat_model = model
    llm_config_langdb = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config_langdb)
    result = llm.chat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


@pytest.mark.parametrize(
    "model",
    [
        "openrouter/anthropic/claude-3.5-haiku-20241022:beta",
        "openrouter/mistralai/mistral-small-24b-instruct-2501:free",
        "openrouter/google/gemini-2.0-flash-lite-001",
    ],
)
def test_llm_openrouter(model: str):
    # override any chat model passed via --m arg to pytest cmd
    settings.chat_model = model
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config)
    result = llm.chat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


@pytest.mark.parametrize(
    "model",
    [
        "aimlapi/gpt-3.5-turbo",
        "aimlapi/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "aimlapi/google/gemini-1.5-flash",
    ],
)
def test_llm_aimlapi(model: str):
    settings.chat_model = model
    llm_config = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config)
    result = llm.chat("what is 3+4?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


@pytest.mark.parametrize(
    "model",
    [
        "portkey/openai/gpt-4o-mini",
        "portkey/anthropic/claude-3-5-haiku-latest",
        "portkey/google/gemini-2.0-flash-lite",
    ],
)
def test_llm_portkey(model: str):
    """Test that LLM access via Portkey works."""
    # override any chat model passed via --m arg to pytest cmd
    settings.chat_model = model

    # Skip if PORTKEY_API_KEY is not set
    if not os.getenv("PORTKEY_API_KEY"):
        pytest.skip("PORTKEY_API_KEY not set")

    # Extract provider from model string
    provider = model.split("/")[1] if "/" in model else ""
    provider_key_var = f"{provider.upper()}_API_KEY"

    # Skip if provider API key is not set
    if not os.getenv(provider_key_var):
        pytest.skip(f"{provider_key_var} not set")

    llm_config_portkey = lm.OpenAIGPTConfig(
        chat_model=model,
    )
    llm = lm.OpenAIGPT(config=llm_config_portkey)
    result = llm.chat("what is 3+4 equal to?")
    assert "7" in result.message
    if result.cached:
        assert result.usage.total_tokens == 0
    else:
        assert result.usage.total_tokens > 0


def test_portkey_params():
    """Test that PortkeyParams are correctly configured."""
    from langroid.language_models.provider_params import PortkeyParams

    # Test with explicit parameters
    params = PortkeyParams(
        api_key="test-key",
        provider="anthropic",
        virtual_key="vk-123",
        trace_id="trace-456",
        metadata={"user": "test"},
        retry={"max_retries": 3},
        cache={"enabled": True},
        cache_force_refresh=True,
        user="user-123",
        organization="org-456",
        custom_headers={"x-custom": "value"},
    )

    headers = params.get_headers()

    assert headers["x-portkey-api-key"] == "test-key"
    assert headers["x-portkey-provider"] == "anthropic"
    assert headers["x-portkey-virtual-key"] == "vk-123"
    assert headers["x-portkey-trace-id"] == "trace-456"
    assert headers["x-portkey-metadata"] == '{"user": "test"}'
    assert headers["x-portkey-retry"] == '{"max_retries": 3}'
    assert headers["x-portkey-cache"] == '{"enabled": true}'
    assert headers["x-portkey-cache-force-refresh"] == "true"
    assert headers["x-portkey-user"] == "user-123"
    assert headers["x-portkey-organization"] == "org-456"
    assert headers["x-custom"] == "value"

    # Test model string parsing
    provider, model = params.parse_model_string("portkey/anthropic/claude-3-sonnet")
    assert provider == "anthropic"
    assert model == "claude-3-sonnet"

    # Test fallback parsing
    provider2, model2 = params.parse_model_string("portkey/some-model")
    assert provider2 == ""
    assert model2 == "some-model"

    # Test provider API key retrieval
    os.environ["TEST_PROVIDER_API_KEY"] = "test-api-key"
    key = params.get_provider_api_key("test_provider")
    assert key == "test-api-key"
    del os.environ["TEST_PROVIDER_API_KEY"]


def test_portkey_integration():
    """Test that Portkey integration is properly configured in OpenAIGPT."""
    from langroid.language_models.provider_params import PortkeyParams

    # Save the current chat model setting
    original_chat_model = settings.chat_model

    # Clear any global chat model override
    settings.chat_model = ""

    try:
        # Test basic portkey model configuration
        config = lm.OpenAIGPTConfig(
            chat_model="portkey/anthropic/claude-3-haiku-20240307",
            portkey_params=PortkeyParams(
                api_key="pk-test-key",
            ),
        )

        llm = lm.OpenAIGPT(config)

        # Check that model was parsed correctly
        assert llm.config.chat_model == "claude-3-haiku-20240307"
        assert llm.is_portkey
        assert llm.api_base == "https://api.portkey.ai/v1"
        assert llm.config.portkey_params.provider == "anthropic"

        # Check headers are set correctly
        assert "x-portkey-api-key" in llm.config.headers
        assert llm.config.headers["x-portkey-api-key"] == "pk-test-key"
        assert llm.config.headers["x-portkey-provider"] == "anthropic"

    finally:
        # Restore original chat model setting
        settings.chat_model = original_chat_model


def test_followup_standalone():
    """Test that followup_to_standalone works."""

    llm = OpenAIGPT(OpenAIGPTConfig())
    dialog = [
        ("Is 5 a prime number?", "yes"),
        ("Is 10 a prime number?", "no"),
    ]
    followup = "What about 11?"
    response = llm.followup_to_standalone(dialog, followup)
    assert response is not None
    assert "prime" in response.lower() and "11" in response


def test_llm_pdf_attachment():
    """Test sending a PDF file attachment to the LLM."""

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

    # Get response from the LLM
    response = llm.chat(messages=messages)

    assert response is not None
    assert response.message is not None
    assert "Supply Chain" in response.message

    # follow-up question
    messages += [
        LLMMessage(role=Role.ASSISTANT, content="Supply Chain"),
        LLMMessage(role=Role.USER, content="Who is the first author?"),
    ]
    response = llm.chat(messages=messages)
    assert response is not None
    assert response.message is not None
    assert "Takio" in response.message


@pytest.mark.xfail(
    reason="Multi-file attachments may not work yet",
    run=True,
    strict=False,
)
def test_llm_multi_pdf_attachments():

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
    # Set up the LLM with a suitable model that supports PDFs
    llm = OpenAIGPT(OpenAIGPTConfig(max_output_tokens=1000))

    response = llm.chat(messages=messages)
    print(response.message)
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
    response = llm.chat(messages=messages)
    assert any(x in response.message for x in ["3", "three"])


def test_llm_pdf_bytes_and_split():
    """Test sending PDF files to LLM as bytes and split into pages."""

    # Path to the test PDF file
    pdf_path = Path("tests/main/data/dummy.pdf")

    # Test creating attachment from bytes
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    attachment_from_bytes = FileAttachment.from_bytes(
        content=pdf_bytes,
        filename="supply_chain_paper.pdf",
    )

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content="Who is the first author of this paper?",
            files=[attachment_from_bytes],
        ),
    ]

    llm = OpenAIGPT(OpenAIGPTConfig(max_output_tokens=100))
    response = llm.chat(messages=messages)

    assert response is not None
    assert "Takio" in response.message

    # Test creating attachment from file-like object
    pdf_io = io.BytesIO(pdf_bytes)
    attachment_from_io = FileAttachment.from_io(
        file_obj=pdf_io,
        filename="paper_from_io.pdf",
    )

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content="What is the title of this paper?",
            files=[attachment_from_io],
        ),
    ]

    response = llm.chat(messages=messages)
    assert "Supply Chain" in response.message

    # Test splitting PDF into pages and sending individual pages
    doc = fitz.open(pdf_path)
    page_attachments = []

    for i, page in enumerate(doc):
        # Extract page as PDF
        page_pdf = io.BytesIO()
        page_doc = fitz.open()
        page_doc.insert_pdf(doc, from_page=i, to_page=i)
        page_doc.save(page_pdf)
        page_pdf.seek(0)

        # Create attachment for this page
        page_attachment = FileAttachment.from_io(
            file_obj=page_pdf,
            filename=f"page_{i+1}.pdf",
        )
        page_attachments.append(page_attachment)

    # Send just the first page
    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content="Based on just this page, what is this document about?",
            files=[page_attachments[0]],
        ),
    ]

    response = llm.chat(messages=messages)
    assert "supply chain" in response.message.lower()

    # Test with multiple pages as separate attachments
    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content="I'm sending you pages from a paper. "
            "How many figures are shown across all pages?",
            files=page_attachments,
        ),
    ]

    response = llm.chat(messages=messages)
    assert response is not None
    assert any(
        x in response.message.lower() for x in ["figure", "diagram", "illustration"]
    )


@pytest.mark.parametrize(
    "path",
    [
        "tests/main/data/color-shape-series.jpg",
        "tests/main/data/color-shape-series.png",
        "tests/main/data/color-shape-series.pdf",
        "https://upload.wikimedia.org/wikipedia/commons/1/18/Seriation_task_w_shapes.jpg",
    ],
)
def test_llm_image_input(path: str):
    attachment = FileAttachment.from_path(path, detail="low")

    messages = [
        LLMMessage(role=Role.SYSTEM, content="You are a helpful assistant."),
        LLMMessage(
            role=Role.USER,
            content="How many squares are here?",
            files=[attachment],
        ),
    ]
    # Set up the LLM with a suitable model that supports PDFs
    llm = OpenAIGPT(OpenAIGPTConfig(max_output_tokens=500))

    response = llm.chat(messages=messages)
    print(response.message)
    assert any(x in response.message for x in ["three", "3"])


def test_litellm_model_key():
    """
    Test that passing in explicit api_key works with `litellm/*` models
    """
    model = "litellm/anthropic/claude-3-5-haiku-latest"
    # disable any chat model passed via --m arg to pytest cmd
    settings.chat_model = model

    class CustomOpenAIGPTConfig(lm.OpenAIGPTConfig):
        """OpenAI config that doesn't auto-load from environment variables."""

        class Config:
            # Set to empty string to disable environment variable loading
            env_prefix = ""

    llm_config = CustomOpenAIGPTConfig(
        chat_model=model, api_key=os.getenv("ANTHROPIC_API_KEY", "")
    )

    # Create the LLM instance
    llm = lm.OpenAIGPT(config=llm_config)
    print(f"\nTesting with model: {llm.chat_model_orig} => {llm.config.chat_model}")
    response = llm.chat("What is 3+4?")
    assert "7" in response.message
