import base64
import io
import tempfile
from pathlib import Path

import pytest

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

    def test_to_dict_image(self):
        """Images are sent as `image_url` content-parts."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(
            content=content, filename="image.png", mime_type="image/png"
        )

        result = attachment.to_dict("gpt-4.1")

        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == attachment.to_data_uri()

    def test_to_dict_video(self):
        """Videos are sent as `video_url` content-parts, not generic file parts."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(content=content, filename="clip.mp4")

        assert attachment.mime_type == "video/mp4"

        result = attachment.to_dict("test-model")

        assert result["type"] == "video_url"
        assert result["video_url"]["url"] == attachment.to_data_uri()
        assert "file" not in result

    def test_to_dict_video_mime_type_case_insensitive(self):
        """Video MIME matching is case-insensitive."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="clip.mp4",
            mime_type="Video/MP4",
        )

        result = attachment.to_dict("gpt-4.1")

        assert result["type"] == "video_url"
        assert result["video_url"]["url"] == attachment.to_data_uri()

    @pytest.mark.parametrize(
        "model",
        ["gemini-2.5-flash", "litellm/gemini/gemini-2.5-flash"],
    )
    def test_to_dict_video_for_gemini_model(self, model: str):
        """Gemini model routing does not reclassify videos as images."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="clip.webm",
            mime_type="video/webm",
        )

        result = attachment.to_dict(model)

        assert result["type"] == "video_url"
        assert result["video_url"]["url"] == attachment.to_data_uri()
        assert "image_url" not in result

    @pytest.mark.parametrize("scheme", ["http", "https"])
    def test_to_dict_video_url_passthrough(self, scheme: str) -> None:
        """A remote video URL is passed through instead of being base64-encoded."""
        url = f"{scheme}://example.com/videos/clip.mp4"
        attachment = FileAttachment.from_path(url)

        assert attachment.mime_type == "video/mp4"

        result = attachment.to_dict("test-model")

        assert result["type"] == "video_url"
        assert result["video_url"]["url"] == url

    def test_to_dict_video_url_mixed_case_scheme(self) -> None:
        """HTTP URL schemes are matched case-insensitively."""
        url = "HTTPS://example.com/videos/clip.mp4"
        attachment = FileAttachment.from_path(url)

        result = attachment.to_dict("test-model")

        assert result == {
            "type": "video_url",
            "video_url": {"url": url},
        }

    def test_to_dict_scheme_only_video_url_uses_data_uri(self) -> None:
        """An HTTP scheme without a host is not treated as a remote URL."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="clip.mp4",
        )
        attachment.url = "https:clip.mp4"

        result = attachment.to_dict("test-model")

        assert result == {
            "type": "video_url",
            "video_url": {"url": attachment.to_data_uri()},
        }

    def test_from_path_colon_filename_is_local(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A local filename resembling a scheme-only URI remains local."""
        monkeypatch.chdir(tmp_path)
        local_path = Path("https:clip.mp4")
        local_path.write_bytes(b"video content")

        attachment = FileAttachment.from_path(local_path.name)

        assert attachment.content == b"video content"
        assert attachment.filename == local_path.name
        assert attachment.url is None

    def test_to_dict_extensionless_video_url(self) -> None:
        """An explicit MIME type supports video URLs without file extensions."""
        url = "https://example.com/watch?id=123"
        attachment = FileAttachment.from_path(url, mime_type="video/mp4")

        result = attachment.to_dict("test-model")

        assert result == {
            "type": "video_url",
            "video_url": {"url": url},
        }

    @pytest.mark.parametrize("url", [object(), 123, ["https://example.com"]])
    def test_to_dict_odd_runtime_url_uses_data_uri(self, url: object) -> None:
        """Non-string runtime URL values safely fall back to attachment data."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="clip.mp4",
        )
        attachment.__dict__["url"] = url

        result = attachment.to_dict("test-model")

        assert result == {
            "type": "video_url",
            "video_url": {"url": attachment.to_data_uri()},
        }

    def test_to_dict_non_media_file(self):
        """Non-image, non-video files keep using generic `file` content-parts."""
        content = b"test content"
        attachment = FileAttachment.from_bytes(content=content, filename="doc.pdf")

        result = attachment.to_dict("test-model")

        assert result["type"] == "file"
        assert result["file"]["filename"] == "doc.pdf"
        assert result["file"]["file_data"] == attachment.to_data_uri()

    def test_to_dict_non_video_file_for_gemini_model(self) -> None:
        """Gemini non-video attachments retain image URL serialization."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="doc.pdf",
            mime_type="application/pdf",
        )
        attachment.detail = "high"

        result = attachment.to_dict("gemini-2.5-flash")

        assert result == {
            "type": "image_url",
            "image_url": {
                "url": attachment.to_data_uri(),
                "detail": "high",
            },
        }

    def test_to_dict_unknown_type(self):
        """Unknown MIME types keep using generic `file` content-parts."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="file.unknown123",
        )

        result = attachment.to_dict("gpt-4.1")

        assert attachment.mime_type == "application/octet-stream"
        assert result == {
            "type": "file",
            "file": {
                "filename": "file.unknown123",
                "file_data": attachment.to_data_uri(),
            },
        }

    def test_to_dict_null_mime_type(self):
        """A post-construction null MIME type keeps generic serialization."""
        attachment = FileAttachment.from_bytes(
            content=b"test content",
            filename="file.unknown123",
        )
        attachment.__dict__["mime_type"] = None

        result = attachment.to_dict("gpt-4.1")

        assert result == {
            "type": "file",
            "file": {
                "filename": "file.unknown123",
                "file_data": "data:None;base64,dGVzdCBjb250ZW50",
            },
        }

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
