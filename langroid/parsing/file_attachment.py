import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel

_HTTP_URL_SCHEMES = frozenset({"http", "https"})
_FTP_URL_SCHEMES = frozenset({"ftp"})


def _is_full_url(value: str, schemes: frozenset[str]) -> bool:
    """Whether a value is a URL with an allowed scheme and valid hostname."""
    try:
        parsed_url = urlparse(value)
        hostname = parsed_url.hostname
    except ValueError:
        return False
    return (
        parsed_url.scheme.casefold() in schemes
        and hostname is not None
        and bool(hostname.strip())
    )


class FileAttachment(BaseModel):
    """Represents a file attachment to be sent to an LLM API."""

    content: bytes
    filename: Optional[str] = None
    mime_type: str = "application/octet-stream"
    url: str | None = None
    detail: str | None = None

    def __init__(self, **data: Any) -> None:
        """Initialize with sensible defaults for filename if not provided."""
        if "filename" not in data or data["filename"] is None:
            # Generate a more readable unique filename
            unique_id = str(uuid.uuid4())[:8]
            data["filename"] = f"attachment_{unique_id}.bin"
        super().__init__(**data)

    @classmethod
    def _from_path(
        cls,
        file_path: Union[str, Path],
        detail: Optional[str] = None,
    ) -> "FileAttachment":
        """Create a FileAttachment from a file path.

        Args:
            file_path: Path to the file to attach

        Returns:
            FileAttachment instance
        """
        path = Path(file_path)
        with open(path, "rb") as f:
            content = f.read()

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "application/octet-stream"

        return cls(
            content=content,
            filename=path.name,
            mime_type=mime_type,
            detail=detail,
        )

    @classmethod
    def _from_url(
        cls,
        url: str,
        content: Optional[bytes] = None,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> "FileAttachment":
        """Create a FileAttachment from a URL.

        Args:
            url: URL to the file
            content: Optional raw bytes content (if already fetched)
            filename: Optional name to use for the file
            mime_type: MIME type of the content, guessed from filename or url

        Returns:
            FileAttachment instance
        """
        if filename is None and url:
            # Extract filename from URL if possible

            parsed_url = urlparse(url)
            path = parsed_url.path
            filename = path.split("/")[-1] if path else None

        if mime_type is None and filename:
            mime_type, _ = mimetypes.guess_type(filename)

        return cls(
            content=content or b"",  # Empty bytes if no content provided
            filename=filename,
            mime_type=mime_type or "application/octet-stream",
            url=url,
            detail=detail,
        )

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        detail: str | None = None,
        mime_type: str | None = None,
    ) -> "FileAttachment":
        """Create a FileAttachment from either a local file path or a URL.

        Args:
            path: Path to the file or URL to fetch.
            detail: Optional image detail level.
            mime_type: Optional MIME type for a URL. Local paths infer their
                MIME type from the filename.

        Returns:
            FileAttachment instance

        Raises:
            ValueError: If path is an FTP URL, which is not supported.
        """
        # Convert to string if Path object
        path_str = str(path)

        if _is_full_url(path_str, _FTP_URL_SCHEMES):
            raise ValueError("FTP URLs are not supported; use an HTTP(S) URL")

        # Check if it's a URL
        if _is_full_url(path_str, _HTTP_URL_SCHEMES):
            return cls._from_url(
                url=path_str,
                detail=detail,
                mime_type=mime_type,
            )
        else:
            # Assume it's a local file path
            return cls._from_path(path_str, detail=detail)

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "FileAttachment":
        """Create a FileAttachment from bytes content.

        Args:
            content: Raw bytes content
            filename: Optional name to use for the file
            mime_type: MIME type of the content, guessed from filename if provided

        Returns:
            FileAttachment instance
        """
        if mime_type is None and filename is not None:
            mime_type, _ = mimetypes.guess_type(filename)

        return cls(
            content=content,
            filename=filename,
            mime_type=mime_type or "application/octet-stream",
        )

    @classmethod
    def from_io(
        cls,
        file_obj: BinaryIO,
        filename: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> "FileAttachment":
        """Create a FileAttachment from a file-like object.

        Args:
            file_obj: File-like object with binary content
            filename: Optional name to use for the file
            mime_type: MIME type of the content, guessed from filename if provided

        Returns:
            FileAttachment instance
        """
        content = file_obj.read()
        return cls.from_bytes(content, filename, mime_type)

    @classmethod
    def from_text(
        cls,
        text: str,
        filename: Optional[str] = None,
        mime_type: str = "text/plain",
        encoding: str = "utf-8",
    ) -> "FileAttachment":
        """Create a FileAttachment from text content.

        Args:
            text: Text content to include
            filename: Optional name to use for the file
            mime_type: MIME type of the content
            encoding: Text encoding to use

        Returns:
            FileAttachment instance
        """
        content = text.encode(encoding)
        return cls(content=content, filename=filename, mime_type=mime_type)

    def to_base64(self) -> str:
        """Convert content to base64 encoding.

        Returns:
            Base64 encoded string
        """
        return base64.b64encode(self.content).decode("utf-8")

    def to_data_uri(self) -> str:
        """Convert content to a data URI.

        Returns:
            A data URI string containing the base64-encoded content with MIME type
        """
        base64_content = self.to_base64()
        return f"data:{self.mime_type};base64,{base64_content}"

    def _content_url(self) -> str:
        """URL to send to the API for this attachment.

        Returns:
            The original URL when it is a full http/https URL that the API can
            fetch directly, else a base64-encoded data URI of the content.
        """
        # If we have a URL and it's a full http/https URL, use it directly
        if isinstance(self.url, str) and _is_full_url(
            self.url,
            _HTTP_URL_SCHEMES,
        ):
            return self.url
        # Otherwise use base64 data URI
        return self.to_data_uri()

    def to_dict(self, model: str) -> Dict[str, Any]:
        """
        Convert to a dictionary suitable for API requests.
        Tested only for PDF files.

        Returns:
            Dictionary with file data
        """
        if isinstance(self.mime_type, str):
            if self.mime_type.casefold().startswith("video/"):
                # Videos are sent as `video_url` content-parts, mirroring the
                # `image_url` parts used for images: a generic `file` part is not
                # recognized as video input by the API.
                return dict(
                    type="video_url",
                    video_url=dict(url=self._content_url()),
                )
            if (
                self.mime_type.casefold().startswith("image/")
                or "gemini" in model.casefold()
            ):
                image_url_dict: Dict[str, str] = dict(url=self._content_url())

                # Add detail parameter if specified
                if self.detail:
                    image_url_dict["detail"] = self.detail

                return dict(
                    type="image_url",
                    image_url=image_url_dict,
                )

        # For non-image files and unexpected runtime MIME values
        return dict(
            type="file",
            file=dict(
                filename=self.filename,
                file_data=self.to_data_uri(),
            ),
        )
