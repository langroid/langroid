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
