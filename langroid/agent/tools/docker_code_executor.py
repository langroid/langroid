import asyncio
import contextlib
import logging
import shlex
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Union

from langroid.pydantic_v1 import BaseSettings

try:
    import docker
    from docker.errors import DockerException, ImageNotFound, NotFound
except ImportError as e:
    raise RuntimeError(
        "Missing dependencies for DockerCodeExecutor. "
        "Please install the 'docker' library: pip install docker"
    ) from e

try:
    import asyncio_atexit  # For cleanup
except ImportError as e:
    raise RuntimeError(
        "Missing dependencies for DockerCodeExecutor. "
        "Please install 'asyncio-atexit': pip install asyncio-atexit"
    ) from e


class CodeBlock(BaseSettings):
    """Represents a block of code to be executed."""

    code: str
    language: str


class CodeResult(BaseSettings):
    """Represents the result of executing code in the container."""

    exit_code: int  # 0 for success, non-zero for error
    output: str  # Captured stdout/stderr


class DockerCodeExecutorConfig(BaseSettings):
    """Configuration for DockerCodeExecutor"""

    image: str = "python:3.11-slim"  # Recommended stable image
    container_name: Optional[str] = None  # Autogenerate if None
    timeout: int = 1000  # Execution timeout in seconds
    # Host directory, defaults to temp
    work_dir: Optional[Union[str, Path]] = None
    auto_remove: bool = True  # Remove container on stop
    stop_container: bool = True  # Stop container on __aexit__ or program exit


class DockerCodeExecutor:
    """
    Executes Python code blocks in a Docker container.
    """

    def __init__(self, config: Optional[DockerCodeExecutorConfig] = None):
        self.config = config or DockerCodeExecutorConfig()

        if self.config.timeout < 1:
            raise ValueError("Timeout must be >= 1.")  # Validate timeout

        # Generate or use provided container name
        self.container_name = (
            self.config.container_name or f"langroid-exec-{uuid.uuid4().hex[:12]}"
        )

        # Prepare host working directory
        if self.config.work_dir:
            self._host_work_dir = Path(self.config.work_dir).resolve()
            self._host_work_dir.mkdir(parents=True, exist_ok=True)  # Ensure it exists
        else:
            self._temp_dir_obj: Optional[tempfile.TemporaryDirectory[str]] = (
                tempfile.TemporaryDirectory()
            )
            self._host_work_dir = Path(self._temp_dir_obj.name).resolve()

        self._docker_client: Optional[docker.DockerClient] = None
        self._container = None
        self._is_running = False
        self._cleanup_registered = False

        logging.info(
            f"Executor initialized: {self.container_name}, "
            f"Image={self.config.image}, WorkDir={self._host_work_dir}"
        )

    @property
    def work_dir(self) -> Path:
        return self._host_work_dir

    @property
    def container_work_dir(self) -> str:
        return "/workspace"  # Fixed mount point inside container

    async def __aenter__(self) -> "DockerCodeExecutor":
        """Asynchronous entry point for the context manager."""
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        """Asynchronous exit point for the context manager."""
        await self.stop()

    async def start(self) -> None:
        if self._is_running:
            return  # Already running, no-op

        logging.info(f"Starting executor {self.container_name}...")
        try:
            # Connect to Docker daemon
            self._docker_client = await asyncio.to_thread(docker.from_env)
            if self._docker_client is not None:
                await asyncio.to_thread(self._docker_client.ping)

                # Pull image if it's not available locally
                try:
                    await asyncio.to_thread(
                        self._docker_client.images.get, self.config.image
                    )
                except ImageNotFound:
                    logging.info(f"Pulling image '{self.config.image}'...")
                    await asyncio.to_thread(
                        self._docker_client.images.pull, self.config.image
                    )
                    logging.info(f"Image '{self.config.image}' pulled.")

                # Remove any leftover container with the same name
                with contextlib.suppress(NotFound, DockerException):
                    existing = await asyncio.to_thread(
                        self._docker_client.containers.get, self.container_name
                    )
                    await asyncio.to_thread(existing.remove, force=True)
                    logging.info(f"Removed existing container '{self.container_name}'.")

                # Create and start a fresh container
                container = await asyncio.to_thread(
                    self._docker_client.containers.create,
                    image=self.config.image,
                    name=self.container_name,
                    working_dir=self.container_work_dir,
                    volumes={
                        str(self.work_dir): {
                            "bind": self.container_work_dir,
                            "mode": "rw",
                        }
                    },
                    command=[
                        "/bin/sh",
                        "-c",
                        "trap 'exit 0' TERM; sleep infinity & wait",
                    ],
                    detach=True,
                    auto_remove=self.config.auto_remove,
                    tty=True,
                )
                await asyncio.to_thread(container.start)
                await asyncio.to_thread(container.reload)

            # Verify container is running
            if container.status != "running":
                logs = await asyncio.to_thread(container.logs)
                log_str = logs.decode(errors="replace")
                raise RuntimeError(
                    f"Start failed: status={container.status}, logs={log_str}"
                )

            self._container = container
            self._is_running = True
            logging.info(
                f"Executor started: {self.container_name} (ID={container.short_id})"
            )

            # Register cleanup at exit
            if self.config.stop_container and not self._cleanup_registered:
                asyncio_atexit.register(self._async_cleanup)
                self._cleanup_registered = True

        except Exception as e:
            logging.error(f"Docker startup error: {e}")
            await self._cleanup_resources()
            raise RuntimeError(f"Startup failed: {e}") from e

    async def stop(self) -> None:
        # Stop the container and unregister cleanup handler
        if not self._is_running or not self._container:
            await self._cleanup_resources()
            return

        logging.info(f"Stopping executor {self.container_name}...")
        try:
            if self.config.stop_container and self._docker_client:
                container_id = self._container.id
                container = await asyncio.to_thread(
                    self._docker_client.containers.get, container_id
                )

                if container.status == "running":
                    # Graceful stop
                    await asyncio.to_thread(container.stop, timeout=10)

                # Remove if auto_remove is disabled
                if not self.config.auto_remove:
                    with contextlib.suppress(NotFound, DockerException):
                        await asyncio.to_thread(container.remove)

        except (DockerException, NotFound) as e:
            logging.warning(f"Error during stop: {e}")

        finally:
            # Clean resources and unregister handler
            await self._cleanup_resources()
            if self._cleanup_registered:
                with contextlib.suppress(ValueError):
                    asyncio_atexit.unregister(self._async_cleanup)
                self._cleanup_registered = False

    async def _async_cleanup(self) -> None:
        # Cleanup called on process exit
        if self._is_running:
            await self.stop()
        await self._cleanup_resources()

    async def _cleanup_resources(self) -> None:
        # Reset state and remove temp dirs
        self._is_running = False
        self._container = None
        if hasattr(self, "_temp_dir_obj") and self._temp_dir_obj:
            await asyncio.to_thread(self._temp_dir_obj.cleanup)
            self._temp_dir_obj = None

    async def restart(self) -> None:
        logging.info(f"Restarting executor {self.container_name}...")
        await self.stop()
        await asyncio.sleep(0.5)  # Ensure resources freed
        await self.start()

    async def execute_code_blocks(self, code_blocks: List[CodeBlock]) -> CodeResult:
        # Execute the first Python block found
        if not self._is_running or not self._container:
            return CodeResult(exit_code=-1, output="Executor is not running.")

        python_block = next((cb for cb in code_blocks if cb.language == "python"), None)
        if not python_block or not python_block.code.strip():
            return CodeResult(exit_code=1, output="No Python code to execute.")

        script = python_block.code
        name = f"script_{uuid.uuid4().hex[:8]}.py"
        host_path = self.work_dir / name
        container_path = f"{self.container_work_dir}/{name}"

        try:
            # Write user code to host-mounted file
            await asyncio.to_thread(host_path.write_text, script, encoding="utf-8")

            cmd = [
                "timeout",
                "--foreground",
                str(self.config.timeout),
                "python",
                "-u",
                container_path,  # unbuffered output
            ]
            logging.info(f"Running: {' '.join(shlex.quote(c) for c in cmd)}")

            # Execute inside container
            result = await asyncio.to_thread(
                self._container.exec_run,
                cmd=cmd,
                workdir=self.container_work_dir,
                stream=False,  # Get result at the end,
                demux=False,  # Combine stdout/stderr
            )

            code = result.exit_code or 0
            out = (result.output or b"").decode("utf-8", errors="replace").strip()

            # Handle timeouts and errors
            if code == 124:
                out += f"\n--- Timed out after {self.config.timeout}s ---"
            elif code != 0:
                out += f"\n--- Exit code {code} ---"

            return CodeResult(exit_code=code, output=out)

        except DockerException as e:
            logging.error(f"Exec error: {e}")
            return CodeResult(exit_code=-1, output=str(e))
        except Exception as e:
            logging.error(f"Unexpected exec error: {e}")
            import traceback

            traceback_str = traceback.format_exc()
            logging.error(f"Traceback: {traceback_str}")
            return CodeResult(
                exit_code=-1, output=f"Internal executor error: {e}\n{traceback_str}"
            )

        finally:
            # Clean up script file
            with contextlib.suppress(Exception):
                await asyncio.to_thread(host_path.unlink)
