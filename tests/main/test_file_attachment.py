import base64
import io
import tempfile
from pathlib import Path

from langroid.parsing.file_attachment import FileAttachment


class TestFileAttachment:
    def test_from_bytes(self):
        """Test creating attachment from bytes."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(
            content=content, filename="test.txt", mime_type="text/plain"
        )

        assert attachment.content == content
        assert attachment.filename == "test.txt"
        assert attachment.mime_type == "text/plain"

    def test_from_io(self):
        """Test creating attachment from BytesIO object."""
        content = b"test content"
        file_obj = io.BytesIO(content)

        attachment = FileAttachment.from_io(
            file_obj=file_obj, filename="test.txt", mime_type="text/plain"
        )

        assert attachment.content == content
        assert attachment.filename == "test.txt"
        assert attachment.mime_type == "text/plain"

    def test_from_text(self):
        """Test creating attachment from text."""
        text = "Hello, world!"
        attachment = FileAttachment.from_text(text=text)

        assert attachment.content == text.encode("utf-8")
        assert attachment.mime_type == "text/plain"
        assert attachment.filename is not None  # Should have default filename

    def test_from_path(self):
        """Test creating attachment from file path."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            tmp.write(b"test content")
            tmp.flush()

            attachment = FileAttachment.from_path(tmp.name)

            assert attachment.content == b"test content"
            assert attachment.filename == Path(tmp.name).name
            assert attachment.mime_type == "text/plain"

    def test_default_filename(self):
        """Test default filename generation when none provided."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(content=content)

        assert attachment.filename is not None
        assert "attachment_" in attachment.filename
        assert attachment.filename.endswith(".bin")

    def test_to_base64(self):
        """Test base64 encoding."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(content=content)

        expected = base64.b64encode(content).decode("utf-8")
        assert attachment.to_base64() == expected

    def test_to_data_uri(self):
        """Test data URI generation."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(content=content, mime_type="text/plain")

        data_uri = attachment.to_data_uri()
        expected_base64 = base64.b64encode(content).decode("utf-8")
        expected_uri = f"data:text/plain;base64,{expected_base64}"

        assert data_uri == expected_uri

    def test_to_dict(self):
        """Test conversion to dict for API requests."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(
            content=content, filename="test.txt", mime_type="text/plain"
        )

        result = attachment.to_dict("gpt-4.1")
        assert result is not None

    def test_mime_type_inference(self):
        """Test MIME type is correctly inferred from filename."""
        content = b"test content"

        pdf = FileAttachment.from_bytes(content=content, filename="doc.pdf")
        assert pdf.mime_type == "application/pdf"

        png = FileAttachment.from_bytes(content=content, filename="image.png")
        assert png.mime_type == "image/png"

        # Change .xyz to .unknown123 which should definitely be unrecognized
        unknown = FileAttachment.from_bytes(content=content, filename="file.unknown123")
        assert unknown.mime_type == "application/octet-stream"
