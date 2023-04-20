from llmagent.language_models.base import LanguageModel
import aiohttp
import asyncio
from typing import List




def make_verbatim_templatized_prompt(num_shots:int=2) -> str:
    templatized_prompt = f"""
    You are a language model. I need a prompt template that includes 
    placeholders for a question '{{question}}' and a passage '{{passage}}'. 
    The template should be designed so that you can extract verbatim any 
    parts of the passage that are relevant to the question. Please generate 
    the prompt template with {num_shots} few-shot demonstrations. The 
    few-shot demos should come first, each demo should contain 
    `Question:`, `Passage:`, `Verbatim Extract:`, and 
    then the actual question, passage, and the prompt should end with 
    'Verbatim Extract:'. There should only be placeholders for the question 
    and passage. 
    """
    return templatized_prompt


def get_single_verbatim_extract(
        question: str, passage: str, LLM: LanguageModel, num_shots: int = 0
) -> str:
    """
    Extract verbatim text from a passage that is relevant to a question.

    Rather than have a "frozen" prompt that we use for all questions and
    passages, this function uses the LLM itself to first generate a prompt for
    extracting verbatim text from a passage. The prompt is templatized to
    include the provided question.
    This has a few advantages:
    (a) we don't hard-code lengthy prompts, and only state our INTENTION here,
    (b) this opens up the possibility that the prompt could be TAILORED to
    the question and passage, this could improve the quality of the answer,
    (c) we don't have to come up with the few-shot demos -- the LLM can do it!


    If the delay or cost of the extra "prompt-generation" step is a concern,
    we could always use a cache mechanism to simply retrieve an older
    templatized prompt.
    Args:
        question: question to be answered
        passage: text from which to extract relevant verbatim text
        LLM: language model to use for generating the prompt and extract
        num_shots: number of few-shot demonstrations to use in prompt

    Returns:
        verbatim extract from passage that is relevant to question, if any
    """

    # Initial question for invoking LLM.generate (to get the templatized prompt)
    # Generate the templatized prompt
    prompt = make_verbatim_templatized_prompt(num_shots)
    templatized_prompt = LLM.generate(prompt=prompt, max_tokens=1024)

    # Substitute provided question and passage into the templatized prompt
    final_prompt = templatized_prompt.format(
        question=question, passage=passage
    )

    # Generate the final verbatim extract based on the final prompt
    final_extract = LLM.generate(prompt=final_prompt, max_tokens=1024)

    return final_extract.strip()

async def get_verbatim_extract_async(
        question: str,
        passage: str,
        LLM: LanguageModel,
        num_shots: int= 1
) -> str:
    """
    Async version of `get_verbatim_extract`
    """
    async with aiohttp.ClientSession():
        prompt = make_verbatim_templatized_prompt(num_shots)
        templatized_prompt = await LLM.agenerate(prompt=prompt, max_tokens=1024)
        # Substitute provided question and passage into the templatized prompt
        final_prompt = templatized_prompt.format(
            question=question, passage=passage
        )
        # Generate the final verbatim extract based on the final prompt
        final_extract = await LLM.agenerate(
            prompt=final_prompt,
            max_tokens=1024
        )

    return final_extract.strip()


async def _get_verbatim_extracts(
        question: str,
        passages: List[str],
        LLM: LanguageModel,
        num_shots: int= 1
) -> List[str]:
    async with aiohttp.ClientSession():
        verbatim_extracts = await asyncio.gather(
            *(get_verbatim_extract_async(question, P, LLM, num_shots=num_shots)
              for P in passages)
        )
    return verbatim_extracts


def get_verbatim_extracts(
        question: str,
        passages: List[str],
        LLM: LanguageModel,
        num_shots: int= 1
) -> List[str]:
    """
    From each passage, extract verbatim text that is relevant to a question,
    using concurrent API calls to the LLM.
    Args:
        question: question to be answered
        passages: list of passages from which to extract relevant verbatim text
        LLM: LanguageModel to use for generating the prompt and extract
        num_shots: number of few-shot demonstrations to use in prompt
    Returns:
        list of verbatim extracts from passages that are relevant to question
    """
    return asyncio.run(_get_verbatim_extracts(
        question, passages, LLM, num_shots=num_shots
    ))

def generate_summarizer_prompt(question: str, texts: List[str], k:int=1):
    # Request for k demonstrations
    demo_request = f"""
    Please provide {k} demonstrations of synthesizing answers based on 
    relevant text fragments for different questions. Include the question, 
    relevant text fragments, and the final synthesized answer for each 
    demonstration.
    """

    # Placeholder for demonstrations
    demo_placeholder = "\n".join([f"Question: [Question {i}]\n-----------\nContent: [Relevant text {i}]\n-----------\nFinal Answer: [Answer {i}]\n" for i in range(1, k+1)])

    # Format the actual question and texts
    actual_question_str = f"Question: {question}\n-----------\n"
    content_lines = "\n".join([f"Content: {text}" for text in texts])
    actual_question_str += content_lines + "\n-----------\nFinal Answer:\n"

    # Combine the request, demonstrations, and actual question to form the complete prompt
    complete_prompt = demo_request + demo_placeholder + "\n" + actual_question_str
    return complete_prompt

def make_summarizer_demos(k):
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
        idk_instruction = ("At least one of the demos should have an "
                           "'I don't know' answer. ")

    meta_prompt = (f"""
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
    """).strip()
    return meta_prompt

def get_summary_answer(
        question: str,
        passages: List[str],
        LLM: LanguageModel,
        k:int=1
) -> str:
    templatized_prompt = """
    You are tasked with answering a question based on provided 
    extracts from a long document. You will receive a question followed by a 
    series of extracts. If the extracts contain information relevant to the 
    question, use the information to produce an accurate answer. If none of 
    the extracts contain information relevant to the question, respond with 
    "I don't know." Below are three examples. Your response should follow the 
    format of these demos:     

    Question: What is the capital city of France and what is a famous landmark located in this city?
    Extract: France is a country located in Western Europe. It is known for its rich history, diverse culture, and world-renowned cuisine. The French language is spoken by the majority of the population.
    Extract: The Eiffel Tower is an iconic landmark located in Paris, France. It was designed and built by the engineer Gustave Eiffel for the 1889 Exposition Universelle, a world's fair held to celebrate the 100th anniversary of the French Revolution. The tower is made of iron and stands at 324 meters tall.
    Answer: The capital city of France is Paris, and the Eiffel Tower is a famous landmark located in this city.
    
    Demo 2
    Question: How does photosynthesis work in plants?
    Extract: Photosynthesis is the process by which plants convert carbon dioxide and water into glucose and oxygen using sunlight as an energy source. This process occurs in specialized structures called chloroplasts, which are found in plant cells.
    Extract: During photosynthesis, chlorophyll molecules in the chloroplasts absorb sunlight and energize electrons. The excited electrons are used to drive a series of chemical reactions that ultimately produce glucose and release oxygen as a byproduct.
    Answer: Photosynthesis is the process by which plants convert carbon dioxide and water into glucose and oxygen using sunlight as an energy source. This process occurs in chloroplasts, where chlorophyll molecules absorb sunlight and energize electrons. These excited electrons drive chemical reactions that produce glucose and release oxygen as a byproduct.
    
    Demo 3
    Question: What are the symptoms of the common cold?
    Extract: The common cold is a viral infection that affects the upper respiratory tract. It is caused by various viruses, the most common of which is the rhinovirus. The infection is usually mild and self-limiting.
    Extract: Penguins are flightless birds that inhabit the polar regions of the Earth. They are known for their distinct black and white plumage and their unique waddling gait. Penguins are excellent swimmers and divers.
    Answer: I don't know.
    
    Question: {question}
    {extracts}
    Answer:   
    """.strip()
    #templatized_prompt = LLM.generate(prompt=prompt, max_tokens=1024)
    # Define an auxiliary function to transform the list of passages into a single string
    def stringify_passages(passages):
        return "\n".join([f"Extract:{p}" for p in passages])

    passages = stringify_passages(passages)
    # Substitute Q and P into the templatized prompt
    final_prompt = templatized_prompt.format(question=question,
                                             extracts=passages)

    # Generate the final verbatim extract based on the final prompt
    final_answer = LLM.generate(prompt=final_prompt, max_tokens=1024)

    return final_answer.strip()

