"""Extract text from a PDF with `pypdfium2`, in a throwaway child process.

This file is executed as a SCRIPT in a separate interpreter -- see
`langroid/parsing/pdf_isolate.py`, which is the only caller:

    python .../langroid/parsing/pdf_worker.py <max_memory_bytes>

It is never imported by Langroid, and it must keep importing nothing from
Langroid: the whole point is a child whose address space is almost entirely
available to pdfium, and whose death costs the caller nothing but one
document. Importing `langroid` here would add hundreds of MB of mappings to
the very budget this process exists to enforce.

Protocol
--------
The PDF arrives as raw bytes on stdin. Results leave on stdout as a stream
of records, each `tag(1) + length(4, big-endian) + payload`:

- ``L`` cap status, ALWAYS the first record: ``ok`` if the address-space cap
  was installed, otherwise the reason it could not be. The caller needs
  "the cap is in force" as a signal separate from "no bomb was found",
  because on some platforms (macOS) the cap cannot be installed at all and a
  silent pass there would mean nothing.
- ``P`` one page's text, UTF-8, in page order.
- ``E`` the parse failed; payload is ``<exception class>\\n<message>``.
- ``D`` end of stream, zero length. Only this record means "finished".

Reaching EOF without a ``D`` means this process died -- which is the
expected outcome for a decompression bomb under the cap, and is what the
caller turns into a clear error.
"""

import os
import struct
import sys
from typing import Callable

# Running this file as a script puts its own directory (`langroid/parsing/`)
# first on the import path. The worker needs nothing from there, and leaving
# it would let a module name in that package shadow a stdlib or pypdfium2
# import, so drop it.
if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    del sys.path[0]

# Block size for reading the PDF off stdin. Large enough to keep the syscall
# count down on a big document, small enough not to over-allocate on a tiny
# one. Record writes do not use it; they are sized by the record.
_READ_BLOCK = 64 * 1024


def _record(tag: bytes, payload: bytes = b"") -> bytes:
    """Frame one protocol record."""
    return tag + struct.pack(">I", len(payload)) + payload


# Built before any parsing, while allocation is still certain to succeed: once
# the cap bites, building this message is exactly what would fail.
_OUT_OF_MEMORY_RECORD = _record(
    b"E",
    b"MemoryError\nThe document's decompressed content exceeded the "
    b"memory cap for the PDF parsing process.",
)


def _install_memory_cap(max_bytes: int) -> str:
    """Cap this process's address space at `max_bytes`.

    Returns:
        ``"ok"`` if the cap is in force, else the reason it is not. Never
        raises: a platform that cannot cap must still parse documents.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return "the 'resource' module is unavailable on this platform"
    # CPython only defines RLIMIT_AS where the OS does, so on OpenBSD (which
    # has no address-space limit) the attribute is simply absent. Reading it
    # directly would raise AttributeError out of this function, killing the
    # worker before it could report anything -- and every document would then
    # look like a bomb.
    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    if rlimit_as is None:  # pragma: no cover - OpenBSD and friends
        return "this platform has no RLIMIT_AS; no memory cap is in force"
    try:
        soft, hard = resource.getrlimit(rlimit_as)
    except (OSError, ValueError) as e:  # pragma: no cover - platform-specific
        return f"RLIMIT_AS could not be read: {e}"
    # Only ever tighten. A deployment that already runs this process under a
    # limit lower than the configured cap means it; installing the cap must
    # not hand the parse MORE memory than it had, which is what setting the
    # soft limit to `max_bytes` would do whenever the hard limit allows it.
    for inherited in (soft, hard):
        if inherited != resource.RLIM_INFINITY and inherited < max_bytes:
            max_bytes = inherited
    try:
        resource.setrlimit(rlimit_as, (max_bytes, hard))
    except (OSError, ValueError, OverflowError) as e:
        # macOS rejects every setrlimit(RLIMIT_AS) call with EINVAL, so this
        # is the normal path there, not an edge case. OverflowError is the
        # other one: a configured cap too large for the platform's rlim_t
        # must not take the worker down with it.
        return f"RLIMIT_AS could not be set ({e}); no memory cap is in force"
    return "ok"


def _read_stdin() -> bytes:
    """Read the whole PDF from stdin."""
    chunks: list[bytes] = []
    while True:
        block = sys.stdin.buffer.read(_READ_BLOCK)
        if not block:
            return b"".join(chunks)
        chunks.append(block)


def _emit_quietly(emit: "Callable[[bytes], None]", record: bytes) -> None:
    """Send a final record, accepting that nobody may be listening.

    Used only on the way out: by then there is nothing left to do about a
    reader that has gone away, and a traceback would be noise.
    """
    try:
        emit(record)
    except BrokenPipeError:
        pass


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write(f"usage: {sys.argv[0]} <max_memory_bytes>\n")
        return 2
    max_bytes = int(sys.argv[1])

    # Move the record stream to a private fd and point fd 1 at stderr, so that
    # anything which writes to stdout -- a library banner, a warning, pdfium
    # itself -- cannot corrupt the framing.
    out_fd = os.dup(1)
    os.dup2(2, 1)

    def emit(record: bytes) -> None:
        """Write one whole record.

        `os.write` on a pipe is free to accept fewer bytes than offered once
        the payload passes the pipe buffer, which a page of text easily does;
        a single unchecked call would silently truncate the frame and
        desynchronise the caller's reader.

        Raises:
            BrokenPipeError: Nobody is reading any more. The caller has gone
                away or stopped consuming pages, so there is no point
                extracting the rest of the document: `main` lets this
                propagate and exits instead of parsing on alone.
        """
        view = memoryview(record)
        while view:
            written = os.write(out_fd, view)
            view = view[written:]

    try:
        emit(_record(b"L", _install_memory_cap(max_bytes).encode()))
    except BrokenPipeError:
        return 1

    try:
        pdf_bytes = _read_stdin()
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(pdf_bytes)
        try:
            for i in range(len(doc)):
                page = doc[i]
                textpage = page.get_textpage()
                text = textpage.get_text_bounded()
                textpage.close()
                page.close()
                emit(_record(b"P", text.encode("utf-8", "replace")))
        finally:
            doc.close()
    except BrokenPipeError:
        # Nobody is listening; stop extracting pages for a reader that left.
        return 1
    except MemoryError:
        # Say nothing that needs an allocation.
        _emit_quietly(emit, _OUT_OF_MEMORY_RECORD)
        return 1
    except BaseException as e:
        detail = f"{type(e).__name__}\n{e}"
        _emit_quietly(emit, _record(b"E", detail.encode("utf-8", "replace")))
        return 1
    _emit_quietly(emit, _record(b"D"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
