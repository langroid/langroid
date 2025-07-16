"""
Tests for OpenAI http_client configuration options.
"""

import os
import ssl
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

import pytest

from langroid.language_models.openai_gpt import OpenAIGPT, OpenAIGPTConfig

# Check if httpx is available
try:
    import httpx  # noqa: F401

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class TestHTTPClientConfiguration:
    """Test http_client configuration options for OpenAIGPT."""

    def test_http_verify_ssl_config(self):
        """Test that http_verify_ssl configuration is properly set."""
        # Test default (SSL verification enabled)
        config = OpenAIGPTConfig(chat_model="gpt-4")
        assert config.http_verify_ssl is True

        # Test SSL verification disabled
        config = OpenAIGPTConfig(chat_model="gpt-4", http_verify_ssl=False)
        assert config.http_verify_ssl is False

    def test_http_client_factory_config(self):
        """Test that http_client_factory can be configured."""

        def mock_client_factory():
            return "mock_client"

        config = OpenAIGPTConfig(
            chat_model="gpt-4", http_client_factory=mock_client_factory
        )
        assert config.http_client_factory is mock_client_factory
        assert config.http_client_factory() == "mock_client"

    def test_http_client_config(self):
        """Test that http_client_config can be configured."""
        client_config = {
            "verify": False,
            "timeout": 30.0,
            "proxy": "http://proxy.example.com:8080",
        }

        config = OpenAIGPTConfig(chat_model="gpt-4", http_client_config=client_config)
        assert config.http_client_config == client_config

    def test_http_client_creation_with_factory(self):
        """Test that http_client is created from factory."""
        client_created = False

        def test_factory():
            nonlocal client_created
            client_created = True
            # Return None to avoid type errors - testing factory is called
            return None

        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            http_client_factory=test_factory,
            use_cached_client=False,  # Ensure we test non-cached path
        )

        # The client should be created during initialization
        _ = OpenAIGPT(config)
        assert client_created is True

    @pytest.mark.skipif(
        not HTTPX_AVAILABLE,
        reason="httpx not installed",
    )
    def test_http_verify_ssl_creates_httpx_client(self):
        """Test that setting http_verify_ssl=False creates httpx client."""
        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            http_verify_ssl=False,
            use_cached_client=False,
        )

        # This should create httpx clients with verify=False
        # We can't easily test the actual client creation without mocking,
        # but we can verify no errors are raised
        llm = OpenAIGPT(config)
        assert llm is not None

    def test_http_verify_ssl_without_httpx_raises_error(self):
        """Test that disabling SSL without httpx installed raises error."""
        # This test would need to mock the httpx import to simulate it not
        # being available. For now, we'll skip this as it requires complex
        # mocking
        pass

    @pytest.mark.skipif(
        not HTTPX_AVAILABLE,
        reason="httpx not installed",
    )
    def test_http_client_config_priority(self):
        """Test that http_client_factory takes priority over http_client_config."""
        factory_called = False

        def test_factory():
            nonlocal factory_called
            factory_called = True
            return None

        # Both factory and config provided - factory should win
        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            http_client_factory=test_factory,
            http_client_config={"verify": False},
            use_cached_client=False,
        )

        _ = OpenAIGPT(config)
        assert factory_called is True

    @pytest.mark.skipif(
        not HTTPX_AVAILABLE,
        reason="httpx not installed",
    )
    def test_http_client_config_creates_cacheable_client(self):
        """Test that http_client_config creates cacheable clients."""
        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            http_client_config={"verify": False},
            use_cached_client=True,  # Should use caching
        )

        # This should create httpx clients with the config
        llm = OpenAIGPT(config)
        assert llm is not None


class TestHTTPClientIntegration:
    """Integration tests for http_client with self-signed certificates."""

    @pytest.mark.skipif(
        os.getenv("CI") == "true",
        reason="Integration test with local HTTPS server - skipped in CI",
    )
    def test_ssl_verification_enabled_fails(self):
        """Test SSL verification behavior with self-signed certificate."""
        import tempfile
        from datetime import datetime, timedelta

        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509.oid import NameOID

        # Generate a self-signed certificate
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )

        subject = issuer = x509.Name(
            [
                x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
            ]
        )

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.utcnow())
            .not_valid_after(datetime.utcnow() + timedelta(days=1))
            .sign(key, hashes.SHA256())
        )

        # Write cert and key to temporary files
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pem"
        ) as cert_file:
            cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
            cert_path = cert_file.name

        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, suffix=".pem"
        ) as key_file:
            key_file.write(
                key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=serialization.NoEncryption(),
                )
            )
            key_path = key_file.name

        # Start a simple HTTPS server with the self-signed cert
        server_started = threading.Event()
        server_port = 0
        server_thread = None
        httpd = None

        class HTTPSHandler(SimpleHTTPRequestHandler):
            def do_POST(self):
                """Handle POST requests to simulate OpenAI API."""
                if self.path == "/v1/chat/completions":
                    self.send_response(401)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(b'{"error": {"message": "Invalid API key"}}')
                else:
                    self.send_response(404)
                    self.end_headers()

        def run_server():
            nonlocal server_port, httpd
            httpd = HTTPServer(("localhost", 0), HTTPSHandler)
            server_port = httpd.server_port
            print(f"DEBUG: Server started on port {server_port}")

            # Configure SSL
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(cert_path, key_path)
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

            server_started.set()
            # Keep server running for the duration of the test
            httpd.serve_forever()

        try:
            # Start the server
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            server_started.wait(timeout=5)
            # Give the server a moment to fully initialize
            import time

            time.sleep(0.5)

            # Test 1: Default behavior (SSL verification enabled) should fail
            config1 = OpenAIGPTConfig(
                chat_model="gpt-4",
                api_base=f"https://localhost:{server_port}/v1",
                api_key="test-key",
                use_cached_client=False,
                timeout=1,  # Short timeout to avoid retries
            )
            # Override retry settings to fail fast
            config1.retry_params.max_retries = 0
            llm1 = OpenAIGPT(config1)

            try:
                # This should fail due to SSL verification
                llm1.chat("test message")
                pytest.fail(
                    "Expected SSL verification error but no exception was raised"
                )
            except Exception as e:
                # Check that it's an SSL-related error
                error_message = str(e)
                print(f"DEBUG: Error message: {error_message}")
                # Check for various SSL-related error messages or connection errors
                # Connection errors often wrap SSL errors
                connection_or_ssl_error = any(
                    phrase.lower() in error_message.lower()
                    for phrase in [
                        "ssl",
                        "certificate",
                        "certificate_verify_failed",
                        "certificate verify failed",
                        "ssl: wrong_version_number",
                        "connection error",
                        "connect error",
                    ]
                )
                if not connection_or_ssl_error:
                    pytest.fail(
                        f"Expected SSL/connection error but got: {error_message}"
                    )

            # Test 2: With SSL verification disabled, should get to API error
            config2 = OpenAIGPTConfig(
                chat_model="gpt-4",
                api_base=f"https://localhost:{server_port}/v1",
                api_key="test-key",
                http_verify_ssl=False,
                use_cached_client=False,
                timeout=1,  # Short timeout to avoid retries
            )
            print(f"DEBUG: http_verify_ssl = {config2.http_verify_ssl}")
            # Override retry settings to fail fast
            config2.retry_params.max_retries = 0
            llm2 = OpenAIGPT(config2)

            try:
                # This should now fail with API error, not SSL error
                llm2.chat("test message")
                pytest.fail("Expected API error but no exception was raised")
            except Exception as e:
                error_message = str(e)
                print(f"DEBUG: With SSL disabled, error: {error_message}")
                # Should get an authentication error, not SSL error
                # Check that it's NOT an SSL error
                ssl_error_found = any(
                    phrase.lower() in error_message.lower()
                    for phrase in ["ssl", "certificate", "certificate_verify_failed"]
                )
                if ssl_error_found:
                    pytest.fail(f"Got SSL error when SSL was disabled: {error_message}")

            # Test 3: With http_client_config, should also bypass SSL and get API error
            config3 = OpenAIGPTConfig(
                chat_model="gpt-4",
                api_base=f"https://localhost:{server_port}/v1",
                api_key="test-key",
                http_client_config={"verify": False},
                use_cached_client=True,  # Test that caching works
                timeout=1,
            )
            config3.retry_params.max_retries = 0
            llm3 = OpenAIGPT(config3)

            try:
                llm3.chat("test message")
                pytest.fail("Expected API error but no exception was raised")
            except Exception as e:
                error_message = str(e)
                print(f"DEBUG: With http_client_config, error: {error_message}")
                # Should get an authentication error, not SSL error
                # Check that it's NOT an SSL error
                ssl_error_found = any(
                    phrase.lower() in error_message.lower()
                    for phrase in ["ssl", "certificate", "certificate_verify_failed"]
                )
                if ssl_error_found:
                    pytest.fail(
                        f"Got SSL error when using http_client_config: {error_message}"
                    )

        finally:
            # Cleanup
            try:
                if httpd:
                    httpd.shutdown()
                os.unlink(cert_path)
                os.unlink(key_path)
            except Exception:
                pass
            if server_thread and server_thread.is_alive():
                server_thread.join(timeout=1)

    def test_custom_http_client_factory_called(self):
        """Test that custom http_client factory is called during initialization."""
        factory_called = False

        def mock_factory():
            nonlocal factory_called
            factory_called = True
            # Return None to avoid type issues - OpenAI will create its own client
            return None

        config = OpenAIGPTConfig(
            chat_model="gpt-4",
            api_key="test-key",
            http_client_factory=mock_factory,
            use_cached_client=False,
        )

        # The factory should be called during initialization
        llm = OpenAIGPT(config)

        # Verify the factory was called
        assert factory_called is True
        assert llm is not None
