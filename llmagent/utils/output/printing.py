from rich import print
from rich.text import Text
from llmagent.utils.configuration import settings


def print_long_text(color, style, preamble, text, chars=100):
    text = " ".join(text.split())
    truncated_text = (
        text[:chars] + "..." + text[-chars:] if len(text) > 2 * chars else text
    )
    styled_text = Text(truncated_text, style=style)
    print(f"[{color}]{preamble} {styled_text}")


def show_if_debug(text, preamble, chars=100, color="red", style="italic"):
    if settings.debug:
        print_long_text(color, style, preamble, text, chars)


# code to show a spinner while waiting for a long-running process

# from tqdm import trange
#
# def black_box():
# # ...code for your black box function here...
#
# # Use trange to display a spinner
# for i in trange(1, desc="Processing", unit="call", ncols=75, ascii=True, leave=False):
#     result = black_box()
#
# # Update progress bar to completion
# print("Done!")
