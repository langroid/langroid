"""
Tests for QdrantDB lock release functionality.
This tests the fix for the file lock conflict issue when using QdrantDB with 
local storage.
"""

import tempfile
from pathlib import Path

from langroid.vector_store.qdrantdb import QdrantDB, QdrantDBConfig


def test_qdrant_explicit_close():
    """Test that explicitly calling close() releases the lock file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = QdrantDBConfig(
            cloud=False,
            collection_name="test_collection",
            storage_path=temp_dir,
        )

        # Create first instance
        db1 = QdrantDB(config)
        db1.clear_empty_collections()

        # Verify lock file exists
        lock_file = Path(temp_dir) / ".lock"
        assert lock_file.exists(), "Lock file should exist after creating QdrantDB"

        # Close the instance
        db1.close()

        # Now we should be able to create another instance without error
        db2 = QdrantDB(config)
        assert db2 is not None, "Should be able to create second instance after close()"

        # Verify we're not using a .new directory
        assert not Path(temp_dir + ".new").exists(), "Should not create .new directory"

        db2.close()


def test_qdrant_context_manager():
    """Test that using context manager properly releases the lock file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = QdrantDBConfig(
            cloud=False,
            collection_name="test_collection",
            storage_path=temp_dir,
        )

        # Use context manager for first instance
        with QdrantDB(config) as db1:
            db1.clear_empty_collections()
            lock_file = Path(temp_dir) / ".lock"
            assert lock_file.exists(), "Lock file should exist inside context"

        # After exiting context, we should be able to create another instance
        with QdrantDB(config) as db2:
            assert db2 is not None, "Should create second instance after context exit"
            # Verify we're not using a .new directory
            assert not Path(
                temp_dir + ".new"
            ).exists(), "Should not create .new directory"


def test_qdrant_context_manager_with_exception():
    """Test that context manager releases lock even when exception occurs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = QdrantDBConfig(
            cloud=False,
            collection_name="test_collection",
            storage_path=temp_dir,
        )

        # Use context manager with exception
        try:
            with QdrantDB(config) as db1:
                # Simulate some operation
                db1.clear_empty_collections()
                # Force an exception
                raise ValueError("Test exception")
        except ValueError:
            pass  # Expected

        # Lock should still be released after exception
        with QdrantDB(config) as db2:
            assert db2 is not None, "Should create instance after exception in context"
            assert not Path(
                temp_dir + ".new"
            ).exists(), "Should not create .new directory"


def test_qdrant_multiple_sequential_operations():
    """Test multiple sequential create/close operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = QdrantDBConfig(
            cloud=False,
            collection_name="test_collection",
            storage_path=temp_dir,
        )

        # Multiple sequential operations
        for i in range(3):
            db = QdrantDB(config)
            db.clear_empty_collections()
            db.close()

            # Verify no .new directories are created
            assert not Path(
                temp_dir + ".new"
            ).exists(), f"Iteration {i}: Should not create .new directory"


def test_qdrant_no_close_creates_new_directory():
    """Test that not closing properly falls back to creating .new directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = QdrantDBConfig(
            cloud=False,
            collection_name="test_collection",
            storage_path=temp_dir,
        )

        # Create first instance without closing
        db1 = QdrantDB(config)

        # Try to create second instance - should fall back to .new
        db2 = QdrantDB(config)

        # Verify .new directory was created due to lock conflict
        assert Path(
            temp_dir + ".new"
        ).exists(), "Should create .new directory when lock exists"

        # Clean up
        db1.close()
        db2.close()
