from typing import Tuple, no_type_check

from pyparsing import Empty, Literal, ParseException, SkipTo, StringEnd, Word, alphanums


@no_type_check
def parse_message(msg: str) -> Tuple[str, str]:
    """
    Parse the intended recipient and content of a message.
    Message format is assumed to be TO[<recipient>]:<message>.
    The TO[<recipient>]: part is optional.

    Args:
        msg (str): message to parse

    Returns:
        str, str: task-name of intended recipient, and content of message
            (if recipient is not specified, task-name is empty string)

    """
    if msg is None:
        return "", ""

    # Grammar definition
    name = Word(alphanums)
    to_start = Literal("TO[").suppress()
    to_end = Literal("]:").suppress()
    to_field = (to_start + name("name") + to_end) | Empty().suppress()
    message = SkipTo(StringEnd())("text")

    # Parser definition
    parser = to_field + message

    try:
        parsed = parser.parseString(msg)
        return parsed.name, parsed.text
    except ParseException:
        return "", msg
