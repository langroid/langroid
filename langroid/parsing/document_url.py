"""Bounded HTTP document retrieval and text decoding."""

from __future__ import annotations

import codecs
import re
from typing import TYPE_CHECKING, Mapping
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from langroid.parsing.parser import ParsingConfig

DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 30.0
DEFAULT_MAX_SIZE = 10 * 1024 * 1024
_STREAM_CHUNK_SIZE = 64 * 1024
_CHARSET_RE = re.compile(
    r"charset\s*=\s*[\"']?([^\s;\"']+)",
    flags=re.IGNORECASE,
)


def is_http_url(source: str) -> bool:
    """Return whether source has an HTTP(S) scheme, ignoring case."""
    parsed = urlparse(source)
    return parsed.scheme.lower() in {"http", "https"}


def url_extension_source(source: str) -> str:
    """Return the URL path or local source used for extension detection."""
    parsed = urlparse(source)
    if parsed.scheme.lower() in {"http", "https"}:
        return parsed.path.lower()
    return source.lower()


def fetch_url_bytes(
    url: str,
    *,
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
    read_timeout: float = DEFAULT_READ_TIMEOUT,
    max_size: int = DEFAULT_MAX_SIZE,
    sample_size: int | None = None,
) -> tuple[bytes, Mapping[str, str]]:
    """Fetch a URL with timeouts and bounded streamed consumption.

    Args:
        url: HTTP(S) document URL.
        connect_timeout: Maximum seconds allowed to establish a connection.
        read_timeout: Maximum seconds allowed between response bytes.
        max_size: Maximum complete response size in bytes.
        sample_size: Optional byte count after which sampling may stop early.

    Returns:
        The response bytes and headers.

    Raises:
        ValueError: If a complete response exceeds ``max_size``.
        requests.exceptions.RequestException: If the request fails or times out.
    """
    if max_size <= 0:
        raise ValueError("URL maximum document size must be positive")
    if sample_size is not None and sample_size <= 0:
        raise ValueError("URL sample size must be positive")

    timeout = (connect_timeout, read_timeout)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if sample_size is None and content_length is not None:
            try:
                declared_size = int(content_length)
            except ValueError:
                declared_size = None
            if declared_size is not None and declared_size > max_size:
                raise ValueError(
                    f"URL document exceeds maximum size of {max_size} bytes; "
                    "increase ParsingConfig.url_max_size to allow larger "
                    "documents"
                )

        limit = sample_size if sample_size is not None else max_size
        chunk_size = min(_STREAM_CHUNK_SIZE, limit + 1)
        body = bytearray()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            body.extend(chunk)
            if sample_size is not None and len(body) >= sample_size:
                return bytes(body[:sample_size]), response.headers
            if len(body) > max_size:
                raise ValueError(
                    f"URL document exceeds maximum size of {max_size} bytes; "
                    "increase ParsingConfig.url_max_size to allow larger "
                    "documents"
                )
        return bytes(body), response.headers


def fetch_configured_url(
    url: str, config: ParsingConfig
) -> tuple[bytes, Mapping[str, str]]:
    """Fetch a complete URL using document parsing configuration."""
    return fetch_url_bytes(
        url,
        connect_timeout=config.url_connect_timeout,
        read_timeout=config.url_read_timeout,
        max_size=config.url_max_size,
    )


def fetch_url_sample(
    url: str, config: ParsingConfig | None = None
) -> tuple[bytes, Mapping[str, str]]:
    """Fetch at most 1 KiB using optional parsing configuration."""
    if config is None:
        return fetch_url_bytes(url, sample_size=1024)
    return fetch_url_bytes(
        url,
        connect_timeout=config.url_connect_timeout,
        read_timeout=config.url_read_timeout,
        max_size=config.url_max_size,
        sample_size=1024,
    )


def decode_document_text(
    content: bytes,
    headers: Mapping[str, str] | None = None,
    *,
    html: bool,
) -> str:
    """Decode document bytes using BOM, HTTP, and HTML declarations.

    BOMs take precedence, followed by a valid charset in the HTTP
    ``Content-Type`` header and, for HTML, encoding detected from its meta
    declarations. If none applies, UTF-8 is used with replacement so malformed
    or legacy undeclared documents remain ingestible.
    """
    bom_encoding = _bom_encoding(content)
    if bom_encoding is not None:
        return content.decode(bom_encoding)

    http_encoding = _http_encoding(headers or {})
    if http_encoding is not None:
        return content.decode(http_encoding, errors="replace")

    if html:
        detected = BeautifulSoup(content, "html.parser").original_encoding
        html_encoding = _validated_encoding(detected)
        if html_encoding is not None:
            return content.decode(html_encoding, errors="replace")

    return content.decode("utf-8", errors="replace")


def _validated_encoding(encoding: str | None) -> str | None:
    """Return a canonical codec name, or None for an invalid declaration."""
    if encoding is None:
        return None
    try:
        return codecs.lookup(encoding).name
    except LookupError:
        return None


def _http_encoding(headers: Mapping[str, str]) -> str | None:
    """Extract and validate an explicitly declared HTTP charset."""
    content_type = headers.get("Content-Type", "")
    match = _CHARSET_RE.search(content_type)
    return _validated_encoding(match.group(1)) if match else None


def _bom_encoding(content: bytes) -> str | None:
    """Return the codec selected by a Unicode byte-order mark."""
    if content.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return "utf-32"
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if content.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    return None
