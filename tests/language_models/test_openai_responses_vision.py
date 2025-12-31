import pytest

from langroid.language_models.base import LLMMessage, Role
from langroid.language_models.openai_responses import (
    OpenAIResponses,
    OpenAIResponsesConfig,
)
from langroid.parsing.file_attachment import FileAttachment


@pytest.mark.openai_responses
@pytest.mark.slow
@pytest.mark.vision
class TestVision:
    def test_image_input(self):
        """Model can process image inputs."""
        # Use a small test image (1x1 red pixel as data URI)
        red_pixel = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        config = OpenAIResponsesConfig(
            chat_model="gpt-4o",  # Vision-capable model
            stream=False,
        )
        llm = OpenAIResponses(config)

        attachment = FileAttachment(content=b"", url=red_pixel)
        messages = [
            LLMMessage(
                role=Role.USER,
                content="What color is this image?",
                files=[attachment],
            ),
        ]

        response = llm.chat(messages, max_tokens=50)

        assert response.message is not None
        assert "red" in response.message.lower()

    def test_image_url_http(self):
        """Model can process image from HTTP URL."""
        config = OpenAIResponsesConfig(
            chat_model="gpt-4o",
            stream=False,
        )
        llm = OpenAIResponses(config)

        # Use a public test image URL
        image_url = "https://via.placeholder.com/150/FF0000/FFFFFF?text=RED"

        attachment = FileAttachment(content=b"", url=image_url)
        messages = [
            LLMMessage(
                role=Role.USER,
                content="What is the dominant color in this image?",
                files=[attachment],
            ),
        ]

        response = llm.chat(messages, max_tokens=50)

        assert response.message is not None
        assert "red" in response.message.lower()

    def test_multiple_images(self):
        """Model can process multiple images in one message."""
        # Two small test images: red and blue pixels
        red_pixel = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )
        blue_pixel = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )

        config = OpenAIResponsesConfig(
            chat_model="gpt-4o",
            stream=False,
        )
        llm = OpenAIResponses(config)

        attachments = [
            FileAttachment(content=b"", url=red_pixel),
            FileAttachment(content=b"", url=blue_pixel),
        ]
        messages = [
            LLMMessage(
                role=Role.USER,
                content="What colors do you see in these two images?",
                files=attachments,
            ),
        ]

        response = llm.chat(messages, max_tokens=100)

        assert response.message is not None
        message_lower = response.message.lower()
        assert "red" in message_lower or "first" in message_lower
        assert "blue" in message_lower or "second" in message_lower

    def test_image_with_streaming(self):
        """Streaming works with image inputs."""
        red_pixel = (
            "data:image/png;base64,"
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        )

        config = OpenAIResponsesConfig(
            chat_model="gpt-4o",
            stream=True,  # Enable streaming
        )
        llm = OpenAIResponses(config)

        attachment = FileAttachment(content=b"", url=red_pixel)
        messages = [
            LLMMessage(
                role=Role.USER,
                content="Describe this image in one word.",
                files=[attachment],
            ),
        ]

        response = llm.chat(messages, max_tokens=20)

        assert response.message is not None
        assert len(response.message) > 0
        assert response.usage is not None

    def test_image_attachment_conversion(self):
        """FileAttachment correctly converts to API format."""
        from langroid.language_models.openai_responses import OpenAIResponses

        # Test data URI image
        data_uri = "data:image/png;base64,ABC123"
        attachment = FileAttachment(
            content=b"dummy_content",  # Required field
            url=data_uri,
            mime_type="image/png",
        )

        # Create instance to access helper methods
        config = OpenAIResponsesConfig(chat_model="gpt-4o")
        llm = OpenAIResponses(config)

        messages = [
            LLMMessage(
                role=Role.USER,
                content="Test",
                files=[attachment],
            ),
        ]

        # Convert messages to input parts
        input_parts = llm._messages_to_input_parts(messages)

        # Should have text part and image part
        assert len(input_parts) >= 2

        # Find the image part - Responses API uses "input_image" type
        image_parts = [p for p in input_parts if p.get("type") == "input_image"]
        assert len(image_parts) == 1
        assert "image_url" in image_parts[0]

        # For data URI, should be in the image_url field
        assert image_parts[0]["image_url"] == data_uri
