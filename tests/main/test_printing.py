"""Tests for `langroid.utils.output.printing`.

`print_long_text` / `show_if_debug` are handed ingested document content (via
`VectorStore.show_if_debug`) and LLM output, so none of it may be interpreted
as Rich markup. These tests pin that down.
"""

import io
import re
from typing import Optional

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


ESC = "\x1b"

# Raw terminal escape sequences an attacker could embed in a document. Rich's
# markup protection does not touch these: they are not markup, they are bytes
# the terminal itself acts on.
ESCAPE_PAYLOADS = {
    "osc8_hyperlink": f"A{ESC}]8;;http://evil.example{ESC}\\CLICK{ESC}]8;;{ESC}\\B",
    "osc8_bel_terminated": f"A{ESC}]8;;http://evil.example\x07CLICK{ESC}]8;;\x07B",
    "sgr_reverse": f"A{ESC}[7mREVERSED{ESC}[0mB",
    "sgr_conceal": f"A{ESC}[8mHIDDEN{ESC}[0mB",
    "clear_screen": f"A{ESC}[2JB",
    "alt_screen": f"A{ESC}[?1049hB",
    "window_title": f"A{ESC}]0;PWNED\x07B",
    "del_char": "A\x7fB",
    # U+009B is the 8-bit Control Sequence Introducer: the same attack without
    # a preceding ESC byte.
    "c1_csi": "A\x9b2JB",
    "c1_osc": "A\x9d8;;http://evil.example\x9cB",
}


@pytest.mark.parametrize("name", sorted(ESCAPE_PAYLOADS))
def test_print_long_text_strips_terminal_escapes(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Raw terminal escape sequences in `text` never reach the terminal.

    Printing `text` as a renderable (rather than inside the markup string)
    means Rich no longer breaks these up with its highlighter, so they must be
    stripped explicitly.
    """
    payload = ESCAPE_PAYLOADS[name]
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", payload)
    raw = buf.getvalue()

    # No control character the payload contained survives -- except that ESC
    # also legitimately appears in the console's own SGR codes, so that one is
    # checked two indirect ways instead.
    for char in sorted(set(payload) - {"A", "B"}):
        if char == ESC or char.isprintable():
            continue
        assert char not in raw, f"{name}: {char!r} survived; raw={raw!r}"
    # None of the payload's own ESC sequences appear verbatim. This check (not
    # just `ESC not in _visible(...)`) is what makes the pure-SGR payloads
    # meaningful: `_visible` strips SGR sequences, so on its own it could not
    # tell the payload's `\x1b[7m` from the console's own coloring.
    for sequence in re.findall(f"{ESC}[^{ESC}]*", payload):
        assert sequence not in raw, f"{name}: {sequence!r} survived; raw={raw!r}"
    assert ESC not in _visible(buf), f"{name}: an escape survived; raw={raw!r}"
    assert OSC8 not in raw, f"{name}: hyperlink/OSC sequence emitted; raw={raw!r}"
    # The printable characters around the escape are kept.
    assert "A" in _visible(buf) and "B" in _visible(buf)


def test_print_long_text_keeps_newlines_and_tabs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stripping control characters keeps the ones documents legitimately use."""
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", "line1\nline2\tcol2")
    visible = _visible(buf)

    assert "line1" in visible and "line2" in visible and "col2" in visible
    assert visible.count("\n") >= 2  # the embedded newline plus the final one


@pytest.mark.parametrize("chars", [None, 20])
def test_print_long_text_keeps_words_apart_across_a_form_feed(
    monkeypatch: pytest.MonkeyPatch, chars: Optional[int]
) -> None:
    """A control character that acted as a separator does not join two words."""
    buf = _capture(monkeypatch)
    print_long_text("red", "italic red", "MATCH-0", "first\fsecond", chars=chars)
    visible = _visible(buf)

    assert "first second" in visible, f"words were joined; shown={visible!r}"


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
