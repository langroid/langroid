from typing import List, Tuple


def collate_chat_history(inputs: List[Tuple[str]]) -> str:
    """
    Collate (human, ai) pairs into a single, string
    Args:
        inputs:
    Returns:
    """
    pairs = [
        f"""Human:{human}
        AI:{ai}
        """
        for human, ai in inputs
    ]
    return "\n".join(pairs)
