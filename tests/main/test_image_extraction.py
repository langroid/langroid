"""Tests for image extraction in Document, DocumentParser, and URLLoader."""

import pytest

from langroid.mytypes import Document, DocMetaData


def test_document_has_images_field():
    """Test that Document class has images field."""
    doc = Document(
        content="Test content",
        metadata=DocMetaData(source="test"),
        images=["image1.png", "image2.jpg"],
    )

    assert hasattr(doc, "images")
    assert isinstance(doc.images, list)
    assert len(doc.images) == 2
    assert doc.images[0] == "image1.png"
    assert doc.images[1] == "image2.jpg"


def test_document_images_defaults_to_empty_list():
    """Test that images field defaults to empty list when not provided."""
    doc = Document(
        content="Test content",
        metadata=DocMetaData(source="test"),
    )

    assert hasattr(doc, "images")
    assert isinstance(doc.images, list)
    assert len(doc.images) == 0


def test_url_loader_extract_image_urls():
    """Test that URLLoader can extract image URLs from HTML."""
    from langroid.parsing.url_loader import TrafilaturaCrawler, TrafilaturaConfig

    html_content = """
    <html>
        <body>
            <img src="/logo.png" alt="Logo">
            <img src="https://example.com/photo.jpg" alt="Photo">
            <img src="../icon.svg" alt="Icon">
        </body>
    </html>
    """

    crawler = TrafilaturaCrawler(TrafilaturaConfig())
    image_urls = crawler.extract_image_urls(html_content, "https://example.com/page")

    assert len(image_urls) == 3
    # Relative URLs should be converted to absolute
    assert "https://example.com/logo.png" in image_urls
    assert "https://example.com/photo.jpg" in image_urls
    # Parent directory reference should be resolved
    assert any("icon.svg" in url for url in image_urls)
