from typing import Optional, List
import langroid as lr
import langroid.language_models as lm
import langroid.utils.configuration
from langroid.language_models import OpenAIGPTConfig
from langroid.utils.configuration import Settings
from langroid.agent.special import DocChatAgentConfig
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig, Splitter
from generation_config_models import load_generation_config, GenerationConfig

# Constants
MODEL_MAP = {
    "1": lm.OpenAIChatModel.GPT4o,
    "2": lm.OpenAIChatModel.GPT4,
    "3": lm.OpenAIChatModel.GPT4o_MINI,
    "4": lm.OpenAIChatModel.GPT4_TURBO,
    "5": lm.OpenAIChatModel.GPT4_32K,
    "6": lm.OpenAIChatModel.GPT3_5_TURBO,
    "7": "ollama/mistral:7b-instruct-v0.2-q8_0",
    "8": "gemini/" + lm.GeminiModel.GEMINI_2_FLASH,
    "9": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH,
    "10": "gemini/" + lm.GeminiModel.GEMINI_1_5_FLASH_8B,
    "11": "gemini/" + lm.GeminiModel.GEMINI_1_5_PRO,
}

MISTRAL_MAX_OUTPUT_TOKENS = 16_000


def get_global_settings(debug: bool = False, nocache: bool = True) -> Settings:
    """
    Retrieve global Langroid settings.

    Args:
        debug (bool): If True, enables debug mode.
        nocache (bool): If True, disables caching.

    Returns:
        Settings: Langroid's global configuration object.
    """
    return langroid.utils.configuration.Settings(
        debug=debug,
        cache=not nocache,
    )


def create_llm_config(
    chat_model_option: str, temperature: Optional[float] = None
) -> OpenAIGPTConfig:
    """
    Creates an LLM (Language Learning Model) configuration based on the selected model.

    This function uses the user's selection (identified by `chat_model_option`)
    to retrieve the corresponding chat model from the `MODEL_MAP` and create
    an `OpenAIGPTConfig` object with the specified settings.

    Args:
        chat_model_option (str): The key corresponding to the user's selected model.

    Returns:
        OpenAIGPTConfig: A configuration object for the selected LLM.

    Raises:
        ValueError: If the user provided`chat_model_option` does not exist in `MODEL_MAP`.
    """

    chat_model = MODEL_MAP.get(chat_model_option)
    # Load generation configuration from JSON
    generation_config: GenerationConfig = load_generation_config(
        "examples/multi-agent-debate/generation_config.json"
    )

    if not chat_model:
        raise ValueError(f"Invalid model selection: {chat_model_option}")

    # Determine max_output_tokens based on the selected model
    max_output_tokens_config = (
        MISTRAL_MAX_OUTPUT_TOKENS
        if chat_model_option == "7"
        else generation_config.max_output_tokens
    )

    # Use passed temperature if provided; otherwise, use the one from the JSON config
    effective_temperature = (
        temperature if temperature is not None else generation_config.temperature
    )

    # Create and return the LLM configuration
    return OpenAIGPTConfig(
        chat_model=chat_model,
        min_output_tokens=generation_config.min_output_tokens,
        max_output_tokens=max_output_tokens_config,
        temperature=effective_temperature,
        seed=generation_config.seed,
    )


def get_base_llm_config(
    chat_model_option: str, temperature: Optional[float] = None
) -> OpenAIGPTConfig:
    """
    Prompt the user to select a base LLM configuration and return it.

    Args:
        config_agent_name (str): The name of the agent being configured.

    Returns:
        OpenAIGPTConfig: The selected LLM's configuration.
    """

    # Pass temperature only if it is provided
    if temperature is not None:
        return create_llm_config(chat_model_option, temperature)
    return create_llm_config(chat_model_option)


def get_questions_agent_config(
    searched_urls: List[str], chat_model: str
) -> DocChatAgentConfig:
    """
    Configure a document-centric Langroid document chat agent based on a
    list of URLs and a chat model.

    Args:
        searched_urls (List[str]): URLs of the documents to be included in the agent's database.
        chat_model (str): The name of the chat model to be used for generating responses.

    Returns:
        DocChatAgentConfig: The configuration for the document-centric chat agent.
    """

    config = DocChatAgentConfig(
        llm=lr.language_models.OpenAIGPTConfig(
            chat_model=chat_model,  # The specific chat model configuration
        ),
        vecdb=lr.vector_store.QdrantDBConfig(
            collection_name="AI_debate",  # Name of the collection in the vector database
            replace_collection=True,  # Whether to replace the collection if it already exists
        ),
        conversation_mode=False,  # Whether the agent is in conversation mode
        n_query_rephrases=0,  # Number of times to rephrase queries
        hypothetical_answer=False,  # Whether to generate hypothetical answers
        extraction_granularity=5,  # Level of detail for extraction granularity
        n_neighbor_chunks=2,  # Number of neighboring chunks to consider in responses
        n_fuzzy_neighbor_words=50,  # Number of words to consider in fuzzy neighbor matching
        use_fuzzy_match=True,  # Whether to use fuzzy matching for text queries
        use_bm25_search=True,  # Whether to use BM25 for search ranking
        cache=True,  # Whether to cache results
        debug=False,  # Debug mode enabled
        stream=True,  # Whether to stream data continuously
        split=True,  # Whether to split documents into manageable chunks
        n_similar_chunks=5,  # Number of similar chunks to retrieve
        n_relevant_chunks=5,  # Number of relevant chunks to retrieve
        parsing=ParsingConfig(
            splitter=Splitter.TOKENS,  # Method to split documents
            chunk_size=200,  # Size of each chunk
            overlap=50,  # Overlap between chunks
            max_chunks=10_000,  # Maximum number of chunks
            n_neighbor_ids=4,  # Number of neighbor IDs to consider in vector space
            min_chunk_chars=200,  # Minimum number of characters in a chunk
            discard_chunk_chars=4,  # Number of characters to discard from chunk boundaries
            pdf=PdfParsingConfig(
                library="fitz",  # Library used for PDF parsing
            ),
        ),
        doc_paths=searched_urls,  # Document paths from searched URLs
    )

    return config
