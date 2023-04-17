from llmagent.language_models.base import LanguageModel
from llmagent.language_models.openai_gpt3 import OpenAIGPT3
from dotenv import load_dotenv
import os


def get_verbatim_extract(
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
        question:
        passage:
        LLM:
        num_shots:
    Returns:
    """

    # Initial question for invoking LLM.generate (to get the templatized prompt)
    initial_question = f"""
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
    # Generate the templatized prompt
    templatized_prompt = LLM.generate(prompt=initial_question, max_tokens=1024)

    # Substitute provided question and passage into the templatized prompt
    final_prompt = templatized_prompt.format(
        question=question, passage=passage
    )

    # Generate the final verbatim extract based on the final prompt
    final_extract = LLM.generate(prompt=final_prompt, max_tokens=1024)

    return final_extract.strip()


# Example usage
if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    LLM = OpenAIGPT3(api_key=api_key)
    Q = "What are the benefits of exercise?"
    P = """
    Exercise is a powerful tool for enhancing physical health and mental 
    well-being. Through activities such as jogging, weightlifting, or yoga, 
    individuals can strengthen their muscles, improve cardiovascular fitness, 
    and boost mood by releasing endorphins. Exercise also aids in stress 
    reduction, better sleep, and weight management. Incorporating regular 
    exercise into one's routine fosters a healthier, more balanced lifestyle.           
    """
    # Request 2 few-shot demonstrations and get the final verbatim extract
    verbatim_extract = get_verbatim_extract(Q, P, LLM, num_shots=2)
    print(verbatim_extract)
