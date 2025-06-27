"""
ADE (Adverse Drug Event) probability estimation task:

Given a pair of (Drug Category, Adverse Event), have the LLM generate an estimate
of the probability that the drug category is associated with an increased risk
of the adverse event.

Run this N times (without caching) to get statistics on the estimates.
Illustrates the use of `llm_response_batch`.

Default model is GPT4o, see how to specify alternative models below.

Example run:

python3 examples/basic/ drug-outcomes.py \
    --model litellm/claude-3-5-sonnet-20240620 --temp 0.1 \
    --pair "(Antibiotics, Acute Liver Injury)" --n 20 --reason true

Interesting models to try:
- gpt-4o (default)
- gpt-4
- litellm/claude-3-5-sonnet-20240620
- groq/llama3-70b-8192

See reference below for specific (DrugCategory, ADE) pairs to test.

References:
    - Guides to using Langroid with local and non-OpenAI models:
        https://langroid.github.io/langroid/tutorials/local-llm-setup/
        https://langroid.github.io/langroid/tutorials/non-openai-llms/
    - OMOP Ground Truth table of known Drug-ADE associations:
        (see page 16 for the table of Drug-ADE pairs)
        https://www.brookings.edu/wp-content/uploads/2012/04/OMOP-methods-review.pdf
"""

import re

import numpy as np
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.utils.configuration import settings

# Turn off cache retrieval, to get independent estimates on each run
settings.cache = False

MODEL = lm.OpenAIChatModel.GPT4o
TEMP = 0.1
PAIR = "(Antibiotics, Acute Liver Injury)"
N = 20
# should LLM include reasoning along with probability?
# (meant to test whether including reasoning along with the probability
# improves accuracy and/or variance of estimates)
REASON: bool = False


def extract_num(x: str) -> int:
    """
    Extracts an integer from a string that contains a number.

    Args:
        x (str): The input string containing the number.

    Returns:
        int: The extracted integer.

    Raises:
        ValueError: If no number is found in the expected format.
    """
    match = re.search(r"\d+", x)
    if match:
        return int(match.group(0))
    else:
        return -1


def main(
    model: str = MODEL,
    temp: float = TEMP,
    pair: str = PAIR,
    n: int = N,
    reason: bool = REASON,
):
    REASONING_PROMPT = (
        """
            IMPORTANT: Before showing your estimated probability, 
            you MUST show 2-3 sentences with your REASONING, and THEN give your 
            percent probability estimate in the range [0,100].
    """
        if reason
        else ""
    )

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                temperature=temp,
                chat_model=model,
            ),
            name="ADE-Estimator",
            system_message=f"""
            You are a clinician with deep knowledge of Adverse Drug Events (ADEs) 
            of various drugs and categories of drugs.
            You will be given a (DRUG CATEGORY, ADVERSE OUTCOME) pair,
            you have to estimate the probability that this DRUG CATEGORY
            is associated with INCREASED RISK of the ADVERSE OUTCOME. 
    
            {REASONING_PROMPT}
                
            You must give your probability estimate as a SINGLE NUMBER e.g. 56, 
            which means 56%.             
            DO NOT GIVE A RANGE OF PROBABILITIES, ONLY A SINGLE NUMBER. 
            """,
        )
    )

    results = lr.llm_response_batch(
        agent,
        [pair] * n,
        # ["(Beta Blockers, Mortality after Myocardial Infarction)"]*20,
    )
    probs = [extract_num(r.content) for r in results]
    cached = [r.metadata.cached for r in results]
    n_cached = sum(cached)
    # eliminate negatives (due to errs)
    probs = [p for p in probs if p >= 0]
    mean = np.mean(probs)
    std = np.std(probs)
    std_err = std / np.sqrt(len(probs))
    hi = max(probs)
    lo = min(probs)
    print(f"Stats for {pair} with {model} temp {temp} reason {reason}:")
    print(
        f"N: {len(probs)} ({n_cached} cached ) Mean: {mean:.2f}, Std: {std:.2f}, StdErr:"
        f" {std_err:.2f}, min: {lo:.2f}, max: {hi:.2f}"
    )
    toks, cost = agent.llm.tot_tokens_cost()
    print(f"Tokens: {toks}, Cost: {cost:.2f}")


if __name__ == "__main__":
    Fire(main)
