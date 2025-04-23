import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union
from urllib.parse import urlparse

from langroid.pydantic_v1 import BaseModel


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
    ) -> "FileAttachment":
        """Create a FileAttachment from either a local file path or a URL.

        Args:
            path_or_url: Path to the file or URL to fetch

        Returns:
            FileAttachment instance
        """
        # Convert to string if Path object
        path_str = str(path)

        # Check if it's a URL
        if path_str.startswith(("http://", "https://", "ftp://")):
            return cls._from_url(url=path_str, detail=detail)
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

    def to_dict(self, model: str) -> Dict[str, Any]:
        """
        Convert to a dictionary suitable for API requests.
        Tested only for PDF files.

        Returns:
            Dictionary with file data
        """
        if (
            self.mime_type
            and self.mime_type.startswith("image/")
            or "gemini" in model.lower()
        ):
            # for gemini models, we use `image_url` for both pdf-files and images

            image_url_dict = {}

            # If we have a URL and it's a full http/https URL, use it directly
            if self.url and (
                self.url.startswith("http://") or self.url.startswith("https://")
            ):
                image_url_dict["url"] = self.url
            # Otherwise use base64 data URI
            else:
                image_url_dict["url"] = self.to_data_uri()

            # Add detail parameter if specified
            if self.detail:
                image_url_dict["detail"] = self.detail

            return dict(
                type="image_url",
                image_url=image_url_dict,
            )
        else:
            # For non-image files
            return dict(
                type="file",
                file=dict(
                    filename=self.filename,
                    file_data=self.to_data_uri(),
                ),
            )
