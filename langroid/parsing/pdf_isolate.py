"""Run the default PDF parser in a child process under a memory cap.

`pypdfium2` (the default PDF library) inflates a document's `/FlateDecode`
content streams inside pdfium, with no budget of its own, so a few hundred KB
of PDF can expand to hundreds of MB and drive the hosting process into the
OOM killer. Nothing that reads the file can bound that reliably: the only
number a reader has for a stream's expanded size is one the document's author
chose.

So the parse does not happen here. It happens in `pdf_worker.py`, a throwaway
interpreter whose address space is capped, and this module is the half that
starts it, streams its pages back page by page, and turns its death into an
ordinary parse error. A bomb then costs one child process and one clear
exception instead of the caller.

Platform note: the cap is `RLIMIT_AS`, which Linux enforces and macOS does
not implement for address space at all -- `setrlimit(RLIMIT_AS, ...)` fails
there with EINVAL for every value. The worker reports whether the cap went
in, and this module logs a warning the first time it did not, because "the
document was fine" and "nothing was checked" must not look alike.
"""

import logging
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Generator, Optional, Tuple

logger = logging.getLogger(__name__)

_WORKER = Path(__file__).with_name("pdf_worker.py")

# Tags of the worker's record protocol; see pdf_worker's module docstring.
_TAG_LIMIT = b"L"
_TAG_PAGE = b"P"
_TAG_ERROR = b"E"
_TAG_DONE = b"D"

_HEADER_LEN = 5

# A sanity rail, not a recommendation. Below some cap the worker cannot even
# finish starting, and then EVERY document comes back as "expanded past the
# cap" -- a misconfiguration that looks exactly like an attack. This floor
# only has to be under any cap that could work: measured on Linux with
# Pillow and numpy loaded, every PDF in this repo (up to 1.3 MB, image-only
# and scanned included) extracts fine under a 32 MB cap, so 16 MB is below
# anything a real deployment would choose.
MIN_MEMORY_BYTES = 16 * 1024 * 1024

# How much of the worker's stderr to quote when it dies. pdfium can be chatty,
# and the useful part is always at the end.
_STDERR_TAIL = 2000

# How long to let a worker that has reported a clean finish exit by itself
# before killing it. Its pages are already delivered either way.
_EXIT_GRACE_SECONDS = 5

# Warn once per process, not once per document: a user on macOS would
# otherwise get this line for every PDF they parse.
_warned_no_cap = False


def _read_exact(stream: IO[bytes], n: int) -> Optional[bytes]:
    """Read exactly `n` bytes, or return None if the stream ends first.

    A short read is indistinguishable from a clean end only at a record
    boundary, which is why the caller treats `None` as "the worker died"
    rather than as "no more pages".
    """
    chunks = []
    remaining = n
    while remaining > 0:
        block = stream.read(remaining)
        if not block:
            return None
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _read_record(stream: IO[bytes]) -> Optional[Tuple[bytes, bytes]]:
    """Read one `(tag, payload)` record, or None at end of stream."""
    header = _read_exact(stream, _HEADER_LEN)
    if header is None:
        return None
    tag = header[:1]
    (length,) = struct.unpack(">I", header[1:])
    payload = _read_exact(stream, length) if length else b""
    if payload is None:
        return None
    return tag, payload


def _note_cap_status(status: str, source: str) -> None:
    """Record whether the memory cap is actually in force."""
    global _warned_no_cap
    if status == "ok":
        return
    if not _warned_no_cap:
        _warned_no_cap = True
        logger.warning(
            "PDF parsing is isolated in a child process, but its memory cap "
            "could not be installed on this platform, so a malicious PDF can "
            "still exhaust memory (the child's, not this process's): %s. "
            "First seen while parsing %s.",
            status,
            source,
        )


def _raise_worker_error(payload: bytes, source: str, max_memory_bytes: int) -> None:
    """Re-raise the exception the worker reported."""
    text = payload.decode("utf-8", "replace")
    kind, _, message = text.partition("\n")
    if kind == "MemoryError":
        # The cap bit cleanly -- Python noticed the failed allocation instead
        # of pdfium aborting. Same situation as a killed worker, so it must
        # give the same actionable message, naming the knob to turn.
        raise ValueError(
            f"Could not parse PDF {source}: {message} "
            f"(PdfParsingConfig.max_memory_bytes = {max_memory_bytes}). "
            f"Raise that cap if the document is legitimate, or set it to 0 "
            f"to parse in this process with no cap."
        )
    if kind == "PdfiumError":
        # A malformed PDF raised `PdfiumError` when this parse ran in-process,
        # and callers catch it; moving the parse to a child process should not
        # change the exception they see.
        pdfium_error = _pdfium_error_class()
        if pdfium_error is not None:
            raise pdfium_error(f"{message} (while parsing {source})")
    raise ValueError(f"Could not parse PDF {source}: {kind}: {message}")


def _pdfium_error_class() -> Optional[type]:
    """`pypdfium2.PdfiumError`, or None if pypdfium2 is not importable here."""
    try:
        import pypdfium2 as pdfium
    except ImportError:  # pragma: no cover - the caller imports it first
        return None
    error_class = getattr(pdfium, "PdfiumError", None)
    if isinstance(error_class, type) and issubclass(error_class, BaseException):
        return error_class
    return None  # pragma: no cover - only if pypdfium2 drops the name


def _describe_death(
    proc: "subprocess.Popen[bytes]",
    stderr_file: IO[bytes],
    max_memory_bytes: int,
    source: str,
) -> str:
    """Explain a worker that exited without a `D` record."""
    returncode = proc.wait()
    try:
        # Seek to the tail rather than reading the file and slicing it: how
        # much a dying pdfium writes to stderr is the document's choice, and
        # reading all of it would move an unbounded allocation into the very
        # process this module keeps safe.
        size = stderr_file.seek(0, 2)
        stderr_file.seek(max(0, size - _STDERR_TAIL))
        tail = stderr_file.read(_STDERR_TAIL).decode("utf-8", "replace").strip()
    except OSError:  # pragma: no cover - the temp file should always be read
        tail = ""
    if returncode < 0:
        how = f"killed by signal {-returncode}"
    else:
        how = f"exited with status {returncode}"
    detail = f"\nWorker stderr (tail):\n{tail}" if tail else ""
    return (
        f"Could not parse PDF {source}: the isolated parsing process {how} "
        f"before finishing. This is what a PDF whose compressed streams "
        f"expand past the memory cap looks like "
        f"(PdfParsingConfig.max_memory_bytes = {max_memory_bytes}). Raise "
        f"that cap if the document is legitimate, or set it to 0 to parse "
        f"in this process with no cap.{detail}"
    )


def iterate_pdf_pages_isolated(
    pdf_bytes: bytes,
    max_memory_bytes: int,
    source: str = "bytes",
) -> Generator[Tuple[int, str], None, None]:
    """Yield `(page index, text)` for a PDF parsed in a capped child process.

    Args:
        pdf_bytes: The raw PDF.
        max_memory_bytes: Address-space cap for the child process. Must be at
            least `MIN_MEMORY_BYTES`, so a value below that -- including any
            negative one -- is refused here rather than parsed. Only exactly
            0 means "do not isolate", and `PyPDFium2Parser` acts on that
            before calling this function.
        source: Path, URL or `"bytes"`, used only in messages.

    Yields:
        Tuple[int, str]: Page index and that page's extracted text.

    Raises:
        ValueError: The cap is too small to run a worker at all, or the
            document could not be parsed -- including the case where the
            child process was killed for exceeding the cap.
        pypdfium2.PdfiumError: The document is malformed (the same error the
            in-process parse raises).
    """
    if max_memory_bytes < MIN_MEMORY_BYTES:
        # Below the floor the worker cannot finish starting, so every
        # document -- bomb or not -- would come back as "exceeded the cap".
        # Saying so is far better than rejecting the user's whole corpus.
        raise ValueError(
            f"PdfParsingConfig.max_memory_bytes is {max_memory_bytes}, which "
            f"is below the {MIN_MEMORY_BYTES} bytes a PDF parsing process "
            f"needs just to start. Raise it, or set it to 0 to parse "
            f"in-process with no cap."
        )
    if not _WORKER.is_file():
        # Fail closed. Quietly parsing in-process instead would hand back the
        # vulnerability this module exists to remove.
        raise RuntimeError(
            f"The PDF parsing worker is missing: {_WORKER}. This is needed to "
            "parse PDFs under a memory cap; set "
            "PdfParsingConfig.max_memory_bytes = 0 to parse in-process "
            "without one."
        )

    # A pipe for stderr would deadlock: we read stdout page by page, and a
    # worker that filled the stderr pipe meanwhile would block forever.
    with tempfile.TemporaryFile() as stderr_file:
        try:
            proc = subprocess.Popen(
                [sys.executable, str(_WORKER), str(max_memory_bytes)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
            )
        except OSError as e:
            raise RuntimeError(
                f"Could not start the isolated PDF parsing process: {e}. Set "
                "PdfParsingConfig.max_memory_bytes = 0 to parse in-process."
            ) from e

        assert proc.stdin is not None and proc.stdout is not None
        try:
            try:
                proc.stdin.write(pdf_bytes)
            except OSError:
                # The worker is already gone -- most likely the document
                # itself is bigger than the cap. The record stream below
                # reports it; nothing is decided here.
                pass
            finally:
                # Must happen on every path. The worker reads stdin to EOF
                # before it parses, so an unclosed pipe after a failed write
                # would leave it waiting for input while we wait for pages.
                try:
                    proc.stdin.close()
                except OSError:
                    pass

            first = _read_record(proc.stdout)
            if first is None:
                raise ValueError(
                    _describe_death(proc, stderr_file, max_memory_bytes, source)
                )
            if first[0] != _TAG_LIMIT:
                raise ValueError(
                    f"Could not parse PDF {source}: the isolated parsing "
                    f"process sent an unexpected first record {first[0]!r}."
                )
            _note_cap_status(first[1].decode("utf-8", "replace"), source)

            index = 0
            while True:
                record = _read_record(proc.stdout)
                if record is None:
                    raise ValueError(
                        _describe_death(proc, stderr_file, max_memory_bytes, source)
                    )
                tag, payload = record
                if tag == _TAG_PAGE:
                    yield index, payload.decode("utf-8", "replace")
                    index += 1
                elif tag == _TAG_DONE:
                    # The worker said it finished, so let it exit on its own
                    # rather than falling into the `finally` below and being
                    # killed a millisecond before it would have. Only the
                    # grace period is new: the pages are already delivered,
                    # so a worker that overstays it is still just killed.
                    try:
                        proc.wait(timeout=_EXIT_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        pass
                    return
                elif tag == _TAG_ERROR:
                    _raise_worker_error(payload, source, max_memory_bytes)
                else:
                    raise ValueError(
                        f"Could not parse PDF {source}: the isolated parsing "
                        f"process sent an unknown record {tag!r}."
                    )
        finally:
            # Covers the abandoned-generator case too: whoever stops consuming
            # pages must not leave a worker behind.
            if proc.poll() is None:
                proc.kill()
            for handle in (proc.stdin, proc.stdout):
                if handle is not None:
                    try:
                        handle.close()
                    except OSError:
                        pass
            proc.wait()
