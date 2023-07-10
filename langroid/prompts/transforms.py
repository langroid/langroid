import asyncio
from typing import List, Tuple

import aiohttp

from langroid.language_models.base import LanguageModel
from langroid.mytypes import DocMetaData, Document
from langroid.prompts.dialog import collate_chat_history
from langroid.prompts.templates import EXTRACTION_PROMPT


async def get_verbatim_extract_async(
    question: str,
    passage: Document,
    LLM: LanguageModel,
) -> str:
    """
    Asynchronously, get verbatim extract from passage that is relevant to a question.
    """
    async with aiohttp.ClientSession():
        templatized_prompt = EXTRACTION_PROMPT
        final_prompt = templatized_prompt.format(question=question, content=passage)
        final_extract = await LLM.agenerate(prompt=final_prompt, max_tokens=1024)

    return final_extract.message.strip()


async def _get_verbatim_extracts(
    question: str,
    passages: List[Document],
    LLM: LanguageModel,
) -> List[Document]:
    async with aiohttp.ClientSession():
        verbatim_extracts = await asyncio.gather(
            *(get_verbatim_extract_async(question, P, LLM) for P in passages)
        )
    metadatas = [P.metadata for P in passages]
    # return with metadata so we can use it downstream, e.g. to cite sources
    return [
        Document(content=e, metadata=m) for e, m in zip(verbatim_extracts, metadatas)
    ]


def get_verbatim_extracts(
    question: str,
    passages: List[Document],
    LLM: LanguageModel,
) -> List[Document]:
    """
    From each passage, extract verbatim text that is relevant to a question,
    using concurrent API calls to the LLM.
    Args:
        question: question to be answered
        passages: list of passages from which to extract relevant verbatim text
        LLM: LanguageModel to use for generating the prompt and extract
    Returns:
        list of verbatim extracts (Documents) from passages that are relevant to
        question
    """
    return asyncio.run(_get_verbatim_extracts(question, passages, LLM))


def generate_summarizer_prompt(question: str, texts: List[str], k: int = 1) -> str:
    # Request for k demonstrations
    demo_request = f"""
    Please provide {k} demonstrations of synthesizing answers based on 
    relevant text fragments for different questions. Include the question, 
    relevant text fragments, and the final synthesized answer for each 
    demonstration.
    """

    # Placeholder for demonstrations
    demo_placeholder = "\n".join(
        [
            f"Question: [Question {i}]\n-----------\n"
            f"Content: [Relevant text {i}]\n-----------\nFinal Answer: [Answer {i}]\n"
            for i in range(1, k + 1)
        ]
    )

    # Format the actual question and texts
    actual_question_str = f"Question: {question}\n-----------\n"
    content_lines = "\n".join([f"Content: {text}" for text in texts])
    actual_question_str += content_lines + "\n-----------\nFinal Answer:\n"

    # Combine the request, demonstrations, and
    # actual question to form the complete prompt
    complete_prompt = demo_request + demo_placeholder + "\n" + actual_question_str
    return complete_prompt


def make_summarizer_demos(k: int) -> str:
    # Define modified original question for LLM.generate
    # templatized_prompt = f"""
    # generate {k} few-shot demos of answering a question based on a list of
    # text contents extracted from a long document, where some or all
    # contents may be irrelevant to the question. When there is no relevant
    # text, the answer should be "I don't know". Each demo should be structured as
    # Question:, Content:, Content:, and so on, and Final Answer: Use 1-3
    # sentences for each piece of content.
    # """
    idk_instruction = ""
    if k > 1:
        idk_instruction = (
            "At least one of the demos should have an " "'I don't know' answer. "
        )

    meta_prompt = (
        f"""
    Generate a templatized prompt for answering questions based on document extracts.
    The prompt should include clear instructions, {k} few-shot demos, and placeholders
    for the input question and extracts.
    
    The instructions should specify that the answer must be based solely on the
    provided extracts. Making up an answer should be discouraged if the information
    is not in the extracts. If none of the extracts are relevant to the question,
    the response should be 'I don't know'.
    
    Each demo should consist of:
       - A sample question (Question:)
       - A series of extracts from a document (Extract:, Extract:, ...),
         with each extract being 1-5 sentences long.
       - A sample answer (Answer:)
    
    {idk_instruction}
    The final prompt should include placeholders:
       - A placeholder {{Question}} for the input question
       - A placeholder {{Extracts}} for the input extracts
    
    The final prompt should end with 'Answer:' to provide the response.
    """
    ).strip()
    return meta_prompt


def get_summary_answer(
    question: str, passages: List[Document], LLM: LanguageModel, k: int = 1
) -> Document:
    templatized_prompt = """
    Use the provided extracts (with sources)  to answer the question. If there's not 
    enough information, respond with "I don't know." Justify your answer by citing 
    your sources, as in these examples:
    
    Extract: The tree species in the garden include oak, maple, and birch.
    Source: https://en.wikipedia.org/wiki/Tree
    Extract: The oak trees are known for their longevity and strength.
    Source: https://en.wikipedia.org/wiki/Oak
    Question: What types of trees are in the garden?
    Answer: The types of trees in the garden include oak, maple, and birch.
    SOURCE: https://en.wikipedia.org/wiki/Tree
    TEXT: The tree species in the garden include oak, maple, and birch.
    
    Extract: The experiment involved three groups: control, low dose, and high dose.
    Source: https://en.wikipedia.org/wiki/Experiment
    Extract: The high dose group showed significant improvement in symptoms.
    Source: https://en.wikipedia.org/wiki/Experiment
    Extract: The control group did not receive any treatment and served as a baseline.
    Source: https://en.wikipedia.org/wiki/Experiment
    Question: How many groups were involved which group showed significant improvement?
    Answer: There were three groups and the high dose group showed significant 
    improvement in symptoms.
    SOURCE: https://en.wikipedia.org/wiki/Experiment
    TEXT: The experiment involved three groups: control, low dose, and high dose.
    SOURCE: https://en.wikipedia.org/wiki/Experiment
    TEXT: The high dose group showed significant improvement in symptoms.
    
    
    Extract: The CEO announced several new initiatives during the company meeting.
    Source: https://en.wikipedia.org/wiki/CEO
    Extract: The financial performance of the company has been strong this year.
    Source: https://en.wikipedia.org/wiki/CEO
    Question: What new initiatives did the CEO announce?
    Answer: I don't know.
    
    {extracts}
    {question}
    Answer:
    """.strip()

    # templatized_prompt = LLM.generate(prompt=prompt, max_tokens=1024)
    # Define an auxiliary function to transform the list of
    # passages into a single string
    def stringify_passages(passages: List[Document]) -> str:
        return "\n".join(
            [
                f"""
            Extract: {p.content}
            Source: {p.metadata.source}
            """
                for p in passages
            ]
        )

    passages_str = stringify_passages(passages)
    # Substitute Q and P into the templatized prompt
    final_prompt = templatized_prompt.format(
        question=f"Question:{question}", extracts=passages_str
    )

    # Generate the final verbatim extract based on the final prompt
    final_answer = LLM.generate(prompt=final_prompt, max_tokens=1024).message.strip()
    parts = final_answer.split("SOURCE:", maxsplit=1)
    if len(parts) > 1:
        content = parts[0].strip()
        sources = parts[1].strip()
    else:
        content = final_answer
        sources = ""
    return Document(
        content=content,
        metadata=DocMetaData(source="SOURCE: " + sources),
    )


def followup_to_standalone(
    LLM: LanguageModel, chat_history: List[Tuple[str, str]], question: str
) -> str:
    """
    Given a chat history and a question, convert it to a standalone question.
    Args:
        chat_history: list of tuples of (question, answer)
        query: follow-up question

    Returns: standalone version of the question
    """
    history = collate_chat_history(chat_history)

    prompt = f"""
    Given the conversationn below, and a follow-up question, rephrase the follow-up 
    question as a standalone question.
    
    Chat history: {history}
    Follow-up question: {question} 
    """.strip()
    standalone = LLM.generate(prompt=prompt, max_tokens=1024).message.strip()
    return standalone
