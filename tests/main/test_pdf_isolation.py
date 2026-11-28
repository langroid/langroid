"""Tests for parsing PDFs in a memory-capped child process.

GHSA-v6ch-vpg9-38xq: pdfium inflates a PDF's compressed streams with no
budget, so a few hundred KB of PDF can expand to hundreds of MB and take the
hosting process down with it. The fix parses in a throwaway child process
under `RLIMIT_AS`.

A platform note that matters for reading these tests: Linux enforces
`RLIMIT_AS`, and macOS does not implement it at all -- `setrlimit` there
fails with EINVAL for every value. So the test that proves a bomb is
contained can only run on Linux, and is skipped elsewhere rather than
quietly passing. `test_worker_reports_whether_the_cap_went_in` is the test
that keeps that honest: it asserts the worker says which of the two
situations it is in, so "no bomb found" can never be confused with "nothing
was checked".
"""

import os
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

from langroid.parsing.document_parser import DocumentParser, DocumentType
from langroid.parsing.parser import ParsingConfig, PdfParsingConfig

MB = 1024 * 1024
GIB = 1024 * MB


def isolate():  # type: ignore[no-untyped-def]
    """Import the new module lazily.

    Deliberately NOT a module-level import: pytest would then fail to collect
    this whole file on a tree without the fix, and
    `test_bomb_rejected_with_stock_config_no_new_settings` could never
    demonstrate the defect it exists to pin down.
    """
    import langroid.parsing.pdf_isolate as module

    return module


DATA = Path(__file__).parent / "data"
REAL_PDF = DATA / "openr-1-3.pdf"

ENFORCES_RLIMIT_AS = sys.platform.startswith("linux")
only_linux = pytest.mark.skipif(
    not ENFORCES_RLIMIT_AS,
    reason="RLIMIT_AS is enforced on Linux only; macOS setrlimit rejects it",
)


def make_bomb(mb: int) -> bytes:
    """A one-page PDF whose /Contents inflates to about `mb` MB.

    This is the reported proof of concept: text operators rather than noise,
    so that pdfium's text extraction really does decompress and process the
    content.
    """
    co = zlib.compressobj(9)
    chunk = b"BT /F1 12 Tf 0 0 Td (AAAA) Tj ET\n" * 4096
    parts, written = [], 0
    while written < mb * MB:
        parts.append(co.compress(chunk))
        written += len(chunk)
    parts.append(co.flush())
    comp = b"".join(parts)
    body = (
        b"<< /Length " + str(len(comp)).encode() + b" /Filter /FlateDecode >>"
        b"\nstream\n" + comp + b"\nendstream"
    )
    objs = [
        (1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        (2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>"),
        (
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        ),
        (4, body),
        (5, b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"),
    ]
    out, offsets = bytearray(b"%PDF-1.7\n"), {}
    for num, obj in objs:
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 6\n0000000000 65535 f \n"
    for n in range(1, 6):
        out += f"{offsets[n]:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def parse_pages(path: Path, max_memory_bytes: int) -> list[str]:
    """Extract page text through the public parser, at a given cap."""
    config = ParsingConfig(
        pdf=PdfParsingConfig(library="pypdfium2", max_memory_bytes=max_memory_bytes)
    )
    parser = DocumentParser.create(str(path), config, doc_type=DocumentType.PDF)
    return [text for _, text in parser.iterate_pages()]


def read_records(blob: bytes) -> list[tuple[str, bytes]]:
    """Decode the worker's `tag + length + payload` record stream."""
    records, pos = [], 0
    while pos < len(blob):
        assert pos + 5 <= len(blob), f"truncated record header at byte {pos}"
        tag = blob[pos : pos + 1].decode()
        (length,) = struct.unpack(">I", blob[pos + 1 : pos + 5])
        payload = blob[pos + 5 : pos + 5 + length]
        assert len(payload) == length, f"truncated payload for {tag!r}"
        records.append((tag, payload))
        pos += 5 + length
    return records


def run_worker(pdf_bytes: bytes, cap: int) -> tuple[int, list[tuple[str, bytes]]]:
    """Run the worker directly, bypassing the caller-side floor check."""
    proc = subprocess.run(
        [sys.executable, str(isolate()._WORKER), str(cap)],
        input=pdf_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode, read_records(proc.stdout)


# --------------------------------------------------------------------------
# The isolated parse must be invisible to everyone but an attacker.
# --------------------------------------------------------------------------


def test_isolated_parse_returns_the_same_text_as_parsing_in_process() -> None:
    """The default (isolated) path and the escape hatch must agree exactly."""
    isolated = parse_pages(REAL_PDF, GIB)
    in_process = parse_pages(REAL_PDF, 0)
    assert isolated == in_process
    assert len(isolated) == 3
    assert isolated[0].strip()


def test_real_pdfs_parse_under_a_tight_cap() -> None:
    """A 64 MB cap is enough for every PDF fixture in tests/main/data.

    The predecessor of this fix was abandoned for rejecting legitimate
    documents, so the headroom claim behind the default is worth asserting
    rather than asserting once in a comment.
    """
    fixtures = sorted(pdf for pdf in DATA.glob("*.pdf") if not pdf.name.startswith("_"))
    assert fixtures, "no PDF fixtures found; this test would be vacuous"
    for pdf in fixtures:
        pages = parse_pages(pdf, 64 * MB)
        assert pages, f"{pdf.name} produced no pages under a 64 MB cap"


def test_a_malformed_pdf_still_raises_pdfium_error() -> None:
    """Moving the parse to a child process must not change the error type."""
    pdfium = pytest.importorskip("pypdfium2")
    junk = DATA / "_not_a_pdf.pdf"
    junk.write_bytes(b"%PDF-1.7\nthis is not a PDF at all\n")
    try:
        with pytest.raises(pdfium.PdfiumError):
            parse_pages(junk, GIB)
    finally:
        junk.unlink()


# --------------------------------------------------------------------------
# Did the check actually run? (separate signal from what it found)
# --------------------------------------------------------------------------


def test_worker_reports_whether_the_cap_went_in() -> None:
    """The worker's first record must state the cap's fate, either way.

    On Linux it must say the cap is installed. On macOS it must say why it is
    not. What it may never do is stay silent, because then a document that
    simply was not a bomb and a platform where nothing was checked look
    identical.
    """
    returncode, records = run_worker(REAL_PDF.read_bytes(), GIB)
    assert records, "worker produced no records at all"
    tag, payload = records[0]
    assert tag == "L", f"first record was {tag!r}, expected the cap status"
    status = payload.decode()
    if ENFORCES_RLIMIT_AS:
        assert status == "ok", f"cap not installed on Linux: {status}"
    else:
        assert status != "ok", "this platform cannot cap, yet status says ok"
        assert "RLIMIT_AS" in status or "resource" in status, status
    assert returncode == 0
    assert [t for t, _ in records] == ["L", "P", "P", "P", "D"]


def test_pages_arrive_in_order_and_the_stream_is_terminated() -> None:
    """A clean parse ends with an explicit `D`; EOF alone means a dead child."""
    _, records = run_worker(REAL_PDF.read_bytes(), GIB)
    assert [t for t, _ in records][-1] == "D"
    texts = [p.decode() for t, p in records if t == "P"]
    assert texts == parse_pages(REAL_PDF, 0)


# --------------------------------------------------------------------------
# The attack.
# --------------------------------------------------------------------------


@only_linux
def test_decompression_bomb_is_rejected_under_the_cap() -> None:
    """The reported attack: 611 KB of PDF expanding to ~200 MB.

    This one sets an explicit tight cap, so it says nothing about the unfixed
    tree (where the setting does not exist). The test below is the one that
    reproduces the defect.
    """
    bomb = DATA / "_bomb.pdf"
    bomb.write_bytes(make_bomb(200))
    try:
        with pytest.raises(ValueError) as caught:
            parse_pages(bomb, 256 * MB)
    finally:
        bomb.unlink()
    message = str(caught.value)
    assert "max_memory_bytes" in message, message
    assert str(bomb) in message, message


@only_linux
def test_bomb_rejected_with_stock_config_no_new_settings() -> None:
    """The counter-verifiable one: nothing here exists only after the fix.

    Deliberately written with no symbol the fix introduces -- a stock
    `ParsingConfig()`, no new field, no new import -- so that on the unfixed
    tree it fails by DID NOT RAISE, i.e. by reproducing the vulnerability,
    rather than by an AttributeError that would prove nothing.

    It pays for that with wall-clock time: pdfium is left to build text
    structures until an allocation fails, which at the default 1 GiB cap
    takes tens of seconds. That is the honest cost of the default and is
    disclosed as such, not an artifact of the test.
    """
    bomb = DATA / "_bomb_stock.pdf"
    bomb.write_bytes(make_bomb(400))
    try:
        config = ParsingConfig()
        parser = DocumentParser.create(str(bomb), config, doc_type=DocumentType.PDF)
        with pytest.raises(ValueError) as caught:
            [text for _, text in parser.iterate_pages()]
        # Not just any ValueError: the one about the cap.
        assert "max_memory_bytes" in str(caught.value), str(caught.value)
    finally:
        bomb.unlink()


@only_linux
def test_the_bomb_is_what_dies_not_the_caller() -> None:
    """The point of the child process: our own memory must not move."""
    import resource

    def peak_mb() -> float:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    bomb_bytes = make_bomb(200)
    before = peak_mb()
    returncode, records = run_worker(bomb_bytes, 256 * MB)
    # Separate "the check ran" from "the check found nothing": a worker that
    # failed to start also exits nonzero and sends no `D`.
    assert records, "the worker produced no records; it never ran"
    assert records[0] == ("L", b"ok"), f"cap was not in force: {records[0]!r}"
    assert returncode != 0, "the worker survived a bomb it should have died on"
    tags = [t for t, _ in records]
    assert "D" not in tags, "worker claimed a clean finish"
    assert "P" not in tags, "worker emitted a page; it got further than it should"
    # It must have died ON the bomb, not on something incidental like a
    # missing pypdfium2, which would also exit nonzero with no `D`.
    for tag, payload in records:
        if tag == "E":
            assert b"MemoryError" in payload or b"memory cap" in payload, payload
    assert peak_mb() - before < 200, (
        f"this process grew from {before:.0f} to {peak_mb():.0f} MB while a "
        "child parsed the bomb"
    )


def test_a_large_but_legitimate_expansion_still_parses() -> None:
    """20 MB of real content must not be mistaken for an attack."""
    big = DATA / "_big_but_fine.pdf"
    big.write_bytes(make_bomb(20))
    try:
        pages = parse_pages(big, GIB)
    finally:
        big.unlink()
    assert len(pages) == 1
    assert pages[0].strip(), "the page parsed but came back empty"


# --------------------------------------------------------------------------
# Failing closed.
# --------------------------------------------------------------------------


def test_a_cap_too_small_to_work_is_refused_up_front() -> None:
    """Better one clear error than every document looking like a bomb."""
    with pytest.raises(ValueError) as caught:
        list(isolate().iterate_pdf_pages_isolated(REAL_PDF.read_bytes(), 1024, "x.pdf"))
    assert "below the" in str(caught.value)
    assert str(isolate().MIN_MEMORY_BYTES) in str(caught.value)


def test_a_missing_worker_is_an_error_not_a_silent_in_process_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falling back to an uncapped parse would hand back the vulnerability."""
    module = isolate()
    monkeypatch.setattr(module, "_WORKER", Path("/nonexistent/pdf_worker.py"))
    with pytest.raises(RuntimeError) as caught:
        list(module.iterate_pdf_pages_isolated(REAL_PDF.read_bytes(), GIB, "x"))
    assert "worker is missing" in str(caught.value)


needs_waitpid = pytest.mark.skipif(
    not hasattr(os, "WNOHANG"),
    reason="os.WNOHANG (and so this leak check) does not exist on Windows",
)


def _has_no_child_processes() -> bool:
    """True if this process has no children, REAPING any that have exited.

    Reaping is why the caller skips rather than asserts when it finds a child
    up front: in a pytest process there is normally none, but if something
    else owns one, this call would consume its exit status and the test's
    verdict would say nothing about our worker anyway.
    """
    try:
        os.waitpid(-1, os.WNOHANG)
    except ChildProcessError:
        return True
    # Either a child is still running (0, 0) or one was just reaped: both
    # mean "there was a child".
    return False


@needs_waitpid
def test_abandoning_the_generator_leaves_no_worker_behind() -> None:
    """Whoever stops reading pages early must not leak a worker.

    A caller that breaks out of the page loop -- an exception, a `head`-style
    read of the first page -- is the normal case, not an exotic one, and a
    worker left running then is a process leak per document.
    """
    if not _has_no_child_processes():
        pytest.skip("this process already has children; a leak is unattributable")

    generator = isolate().iterate_pdf_pages_isolated(
        REAL_PDF.read_bytes(), GIB, "x.pdf"
    )
    next(generator)  # starts the worker
    generator.close()  # the generator's `finally` must kill and reap it

    assert _has_no_child_processes(), "a PDF worker process was left behind"
