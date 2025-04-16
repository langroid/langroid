import base64
import mimetypes
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional, Union

from langroid.pydantic_v1 import BaseModel


class FileAttachment(BaseModel):
    """Represents a file attachment to be sent to an LLM API."""

    content: bytes
    filename: Optional[str] = None
    mime_type: str = "application/octet-stream"

    def __init__(self, **data: Any) -> None:
        """Initialize with sensible defaults for filename if not provided."""
        if "filename" not in data or data["filename"] is None:
            # Generate a more readable unique filename
            unique_id = str(uuid.uuid4())[:8]
            data["filename"] = f"attachment_{unique_id}.bin"
        super().__init__(**data)

    @classmethod
    def from_path(cls, file_path: Union[str, Path]) -> "FileAttachment":
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

        return cls(content=content, filename=path.name, mime_type=mime_type)

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
        if "gemini" in model.lower():
            return dict(type="image_url", image_url=dict(url=self.to_data_uri()))
        elif "claude" in model.lower():
            # optimistically try this: some API proxies like litellm
            # support this, and others may not.
            return dict(
                type="file",
                file=dict(
                    file_data=self.to_data_uri(),
                ),
            )
        else:
            # fallback: assume file upload is similar to OpenAI API
            return dict(
                type="file",
                file=dict(
                    filename=self.filename,
                    file_data=self.to_data_uri(),
                ),
            )
