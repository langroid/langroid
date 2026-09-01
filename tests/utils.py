import pytest

# The OpenAI Assistants API beta was sunset on 2026-08-26, and every endpoint
# (including `POST /assistants`) now returns a bare 404, so any test that talks
# to the live API can never pass. Skip such tests until Langroid's
# `OpenAIAssistant` is migrated to the Responses API.
# See https://developers.openai.com/api/docs/assistants/migration
assistants_api_sunset = pytest.mark.skip(
    reason="OpenAI Assistants API was sunset on 2026-08-26 and returns 404; "
    "pending migration to the Responses API"
)


def contains_approx_float(s: str, x: int | float, k: int = 0) -> bool:
    """
    Check if a string contains a float that is approximately equal to x.
    E.g., s = "The average income is $100,000.134", x = 100000.13, k = 2

    Args:
        s (str): the string to search
        x (int|float): the float or int to search for
        k (int): the number of decimal places to round to

    Returns:
        bool: True if s contains a float or int that is approximately equal to x

    """
    for word in s.split():
        # Remove commas and dollar signs
        clean_word = word.replace(",", "").replace("$", "").replace("%", "")
        # Remove trailing period if present
        if clean_word.endswith("."):
            clean_word = clean_word[:-1]
        if clean_word.endswith("$"):
            clean_word = clean_word[:-1]

        try:
            float_val = float(clean_word)
            if round(float_val, k) == round(x, k):
                return True
        except ValueError:
            # Not a valid float, continue to next word
            pass

    return False
