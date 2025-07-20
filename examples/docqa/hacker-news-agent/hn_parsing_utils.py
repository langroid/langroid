from typing import Any, List, Optional, Dict
from langroid.mytypes import DocMetaData, Document
from pydantic import Field, BaseModel


class ThreadMetadata(DocMetaData):
    """Metadata for HN thread documents"""

    post_id: str = Field(description="The unique ID of the HN post")
    title: str = Field(description="The title of the HN post")
    post_url: str = Field(description="The URL of the original post")
    author: str = Field(description="The author of the post")
    points: str = Field(description="Number of points/upvotes")
    age: str = Field(description="How long ago the post was made")
    comment_count: str = Field(description="Number of comments")
    thread_id: Optional[str] = Field(
        default=None, description="Thread ID if this is a comment thread"
    )
    total_comments: int = Field(
        default=0, description="Total number of comments in the thread"
    )
    max_nesting_level: int = Field(
        default=0, description="Maximum nesting level of comments"
    )


class ThreadDoc(BaseModel):  # or ThreadDoc(Document) if Document is a BaseModel
    """Document representing an HN thread"""

    content: str = Field(
        description="The full content of the thread including post and comments"
    )
    metadata: ThreadMetadata


def parse_comment_thread(thread: Dict[str, Any]) -> str:
    """Parse a comment thread into readable text"""
    content_parts = []

    # Add top-level comment
    top_comment = thread.get("top_level_comment", {})
    if top_comment:
        content_parts.append(top_comment.get("comment_text_clean", ""))
        content_parts.append("")  # Empty line for separation

    # Add replies
    for reply in thread.get("all_replies", []):
        indent = "  " * reply.get("nesting_level", 0)  # Indent based on nesting level
        content_parts.append(f"{indent}{reply.get('comment_text_clean', '')}")
        content_parts.append("")  # Empty line for separation

    return "\n".join(content_parts)


def process_hn_json(
    json_data: Dict[str, Any], split_strategy: str = "per_thread"
) -> List[ThreadDoc]:
    """Process HN JSON data into ThreadDoc objects

    Args:
        split_strategy:
            - "per_thread": Each comment thread = 1 document (default)
            - "per_comment": Each individual comment = 1 document
            - "whole_post": Entire post + all threads = 1 document
    """
    documents = []
    post_info = json_data.get("post", {})

    if split_strategy == "whole_post":
        # Single document with everything
        content_parts = [
            f"# {post_info.get('title', 'Untitled Post')}",
            f"Post by: {post_info.get('author', 'Unknown')}",
            "## All Comment Threads:",
            "",
        ]

        for thread in json_data.get("comment_threads", []):
            content_parts.append(f"### Thread {thread.get('thread_id', '')}")
            thread_content = parse_comment_thread(thread)
            content_parts.append(thread_content)
            content_parts.append("---")  # Separator

        full_content = "\n".join(content_parts)

        metadata = ThreadMetadata(
            post_id=post_info.get("post_id", ""),
            thread_id=post_info.get("post_id", ""),
            title=post_info.get("title", ""),
            post_url=post_info.get("post_url", ""),
            author=post_info.get("author", ""),
            points=post_info.get("points", "0"),
            age=post_info.get("age", ""),
            comment_count=post_info.get("comment_count", "0"),
            total_comments=json_data.get("total_comments", 0),
            max_nesting_level=max(
                [
                    t.get("max_nesting_level", 0)
                    for t in json_data.get("comment_threads", [])
                ],
                default=0,
            ),
        )

        doc = ThreadDoc(content=full_content, metadata=metadata)
        documents.append(doc)

    elif split_strategy == "per_comment":
        # Each individual comment = separate document
        for thread in json_data.get("comment_threads", []):
            # Top-level comment
            top_comment = thread.get("top_level_comment", {})
            if top_comment:
                content += top_comment.get("comment_text_clean", "")

                metadata = ThreadMetadata(
                    post_id=post_info.get("post_id", ""),
                    title=post_info.get("title", ""),
                    post_url=post_info.get("post_url", ""),
                    author=top_comment.get("author", "Unknown"),
                    points=post_info.get("points", "0"),
                    age=top_comment.get("age", "unknown"),
                    comment_count="1",
                    thread_id=thread.get("thread_id", ""),
                    total_comments=1,
                    max_nesting_level=0,
                )

                doc = ThreadDoc(content=content, metadata=metadata)
                documents.append(doc)

            # Each reply as separate document
            for reply in thread.get("all_replies", []):
                content += reply.get("comment_text_clean", "")

                metadata = ThreadMetadata(
                    post_id=post_info.get("post_id", ""),
                    title=post_info.get("title", ""),
                    post_url=post_info.get("post_url", ""),
                    author=reply.get("author", "Unknown"),
                    points=post_info.get("points", "0"),
                    age=reply.get("age", "unknown"),
                    comment_count="1",
                    thread_id=thread.get("thread_id", ""),
                    total_comments=1,
                    max_nesting_level=reply.get("nesting_level", 0),
                )

                doc = ThreadDoc(content=content, metadata=metadata)
                documents.append(doc)

    else:  # per_thread (default)
        # Process each comment thread as a separate document
        for thread in json_data.get("comment_threads", []):
            # Create content combining post info and thread comments
            content_parts = [
                f"# {post_info.get('title', 'Untitled Post')}",
                "",
                "## Comments Thread:",
                "",
            ]

            # Add thread content
            thread_content = parse_comment_thread(thread)
            content_parts.append(thread_content)

            full_content = "\n".join(content_parts)

            # Create metadata
            metadata = ThreadMetadata(
                post_id=post_info.get("post_id", ""),
                title=post_info.get("title", ""),
                post_url=post_info.get("post_url", ""),
                author=post_info.get("author", ""),
                points=post_info.get("points", "0"),
                age=post_info.get("age", ""),
                comment_count=post_info.get("comment_count", "0"),
                thread_id=thread.get("thread_id", ""),
                total_comments=thread.get("total_replies", 0),
                max_nesting_level=thread.get("max_nesting_level", 0),
            )

            # Create document
            doc = ThreadDoc(content=full_content, metadata=metadata)

            documents.append(doc)

    return documents
