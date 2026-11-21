"""Tests for `langroid.utils.output.printing`.

`print_long_text` / `show_if_debug` are handed ingested document content (via
`VectorStore.show_if_debug`) and LLM output, so none of it may be interpreted
as Rich markup. These tests pin that down.
"""

import io
import re

import pytest
from rich.console import Console

import langroid.utils.output.printing as printing_module
from langroid.utils.configuration import settings
from langroid.utils.output.printing import print_long_text, show_if_debug

# SGR (color/style) sequences only -- deliberately NOT OSC-8 hyperlinks, which
# the tests below must still be able to see in the raw output.
SGR_RE = re.compile(r"\x1b\[[0-9;]*m")

# Start of an OSC-8 terminal hyperlink.
OSC8 = "\x1b]8;"

STYLE_TAG = "[bold red on white]"
LINK_TAG = "[link=file:///etc/passwd]"

# Markup an attacker could embed in an ingested document: a style tag, a
# closing tag that would escape the enclosing style, and a terminal hyperlink.
PAYLOAD = f"A{STYLE_TAG}SYSTEM[/]B{LINK_TAG}click[/link]C"

# Rich's `Text` strips these control characters from its input. A backslash
# followed by one of them and then a tag defeats `rich.markup.escape()`: the
# escape lands before the control character, which is then removed, leaving
# the escaping backslash paired with the one already in the text -- an escaped
# backslash -- and the tag live again. Printing `text` as a renderable (rather
# than escaping it into a markup string) is immune to this.
STRIPPED_CONTROL_CHARS = ["\x07", "\x08", "\x0b", "\x0c", "\r"]


def _capture(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Route the module's `rprint` to a wide, terminal-like captured Console.

    `force_terminal=True` makes Rich emit the control sequences a real
    terminal would receive (so e.g. an OSC-8 hyperlink becomes visible to the
    assertions), and the wide width keeps output on a single line.
    """
    # Rich honors NO_COLOR even with force_terminal/color_system set, which
    # would strip the color the styling assertions look for.
    monkeypatch.delenv("NO_COLOR", raising=False)
    buf = io.StringIO()
    console = Console(
        file=buf, width=400, force_terminal=True, color_system="truecolor"
    )
    monkeypatch.setattr(printing_module, "rprint", console.print)
    return buf


def _visible(buf: io.StringIO) -> str:
    """The characters a user would see, with color/style sequences removed."""
    return SGR_RE.sub("", buf.getvalue())


def test_print_long_text_shows_markup_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markup in `text` is shown literally, not interpreted by Rich."""
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", PAYLOAD)
    visible = _visible(buf)

    # The text is reproduced character for character ...
    assert PAYLOAD in visible, f"text was altered; shown={visible!r}"
    # ... and no OSC-8 hyperlink was emitted.
    assert OSC8 not in buf.getvalue()


@pytest.mark.parametrize("control_char", STRIPPED_CONTROL_CHARS)
def test_print_long_text_survives_control_char_tag_smuggling(
    monkeypatch: pytest.MonkeyPatch, control_char: str
) -> None:
    """A control character cannot smuggle a live tag past the protection."""
    payload = f"A\\{control_char}{LINK_TAG}click[/link]B"
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", payload)
    visible = _visible(buf)

    assert LINK_TAG in visible, f"tag was interpreted; shown={visible!r}"
    assert OSC8 not in buf.getvalue(), "a real terminal hyperlink was emitted"


def test_print_long_text_truncates_and_stays_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Markup is still literal on the `chars`-truncation path."""
    filler = "x" * 300
    payload = f"{STYLE_TAG}{filler}Y{LINK_TAG}"
    # Chosen so that BOTH tags survive truncation *whole*: otherwise the
    # assertions below could not tell escaping from a tag split in half.
    chars = len(LINK_TAG) + 1
    expected = payload[:chars] + "..." + payload[-chars:]
    assert STYLE_TAG in expected and LINK_TAG in expected, "test payload is wrong"

    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", payload, chars=chars)
    visible = _visible(buf)

    # Truncation really happened: the filler is gone and an ellipsis is in.
    assert filler not in visible
    assert "..." in visible
    # Both surviving tags are shown literally, and no hyperlink was emitted.
    assert expected in visible, f"truncated text was altered; shown={visible!r}"
    assert OSC8 not in buf.getvalue()


def test_print_long_text_preserves_plain_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text without markup is printed verbatim, with no escaping artifacts."""
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", "plain content here")
    visible = _visible(buf)

    assert "MATCH-0 plain content here" in visible
    assert "\\" not in visible


def test_print_long_text_preserves_trailing_backslash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trailing backslash is printed once, not doubled."""
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", "C:\\dir\\")
    visible = _visible(buf)

    assert "MATCH-0 C:\\dir\\\n" == visible


def test_print_long_text_applies_color_and_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both `color` and `style` reach the text, including a theme style name."""
    buf = _capture(monkeypatch)
    # "repr.number" is a theme style name: it resolves on its own, but not as
    # one word of a longer style string.
    print_long_text("repr.number", "italic", "MATCH-0", "BODY")
    raw = buf.getvalue()

    # bold cyan (the theme style) AND italic (3) on the body.
    assert "\x1b[1;3;36mBODY" in raw, f"body lost styling; raw={raw!r}"


def test_show_if_debug_shows_markup_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The debug-print entry point used by the vector stores is covered too."""
    buf = _capture(monkeypatch)
    monkeypatch.setattr(settings, "debug", True)
    show_if_debug(PAYLOAD, "MATCH-0")
    visible = _visible(buf)

    assert PAYLOAD in visible
    assert OSC8 not in buf.getvalue()


def test_show_if_debug_silent_without_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`show_if_debug` prints nothing when `settings.debug` is off."""
    buf = _capture(monkeypatch)
    monkeypatch.setattr(settings, "debug", False)
    show_if_debug(PAYLOAD, "MATCH-0")

    assert buf.getvalue() == ""
