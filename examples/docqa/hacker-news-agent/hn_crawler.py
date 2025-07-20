import asyncio
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
import markdownify as md
from datetime import datetime


def extract_comment_id_from_permalink(permalink):
    """
    Extracts the comment ID from an HN permalink URL (e.g., item?id=44594475).
    Returns None if not found.
    """
    match = re.search(r"id=(\d+)", permalink)
    return match.group(1) if match else None


def extract_links_from_html(html_content):
    """
    Extract all links from HTML content and return them with their text.
    Returns a list of dictionaries with 'url', 'text', and 'domain' keys.
    """
    if not html_content:
        return []

    links = []

    # Find all <a> tags with href attributes
    link_pattern = r'<a\s+(?:[^>]*?\s+)?href=["\'](.*?)["\'](?:[^>]*?)>(.*?)</a>'
    matches = re.findall(link_pattern, html_content, re.IGNORECASE | re.DOTALL)

    for url, text in matches:
        # Clean up the URL and text
        url = url.strip()
        text = re.sub(r"<[^<]+?>", "", text).strip()  # Remove any HTML tags from text

        # Skip empty URLs or anchor links
        if not url or url.startswith("#"):
            continue

        # Handle relative URLs (though HN usually uses absolute URLs)
        if url.startswith("//"):
            url = "https:" + url
        elif url.startswith("/"):
            url = "https://news.ycombinator.com" + url

        # Extract domain for categorization
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Skip HN internal links unless they're interesting
            if domain == "news.ycombinator.com" and not any(
                x in url for x in ["item?id=", "user?id="]
            ):
                continue

        except:
            domain = "unknown"

        links.append(
            {
                "url": url,
                "text": text or url,  # Use URL as text if no text found
                "domain": domain,
            }
        )

    # Remove duplicates while preserving order
    seen_urls = set()
    unique_links = []
    for link in links:
        if link["url"] not in seen_urls:
            seen_urls.add(link["url"])
            unique_links.append(link)

    return unique_links


def extract_md_from_html(html_content):
    """Extract clean text from HTML content."""
    if not html_content:
        return ""
    return md.markdownify(html_content)


def group_comments_by_thread(all_comments):
    """
    Group comments into threads where each top-level comment
    contains all its descendants flattened.
    """
    print(f"Processing {len(all_comments)} comments...")

    # Add comment_id from permalink and validate
    valid_comments = []
    for i, comment in enumerate(all_comments):
        # Extract comment_id from permalink if not present
        if not comment.get("comment_id"):
            permalink = comment.get("permalink", "")
            comment_id = extract_comment_id_from_permalink(permalink)
            if comment_id:
                comment["comment_id"] = comment_id
            else:
                print(
                    f"Warning: Comment {i} missing comment_id and couldn't extract from permalink: {comment}"
                )
                continue

        if not comment.get("author"):
            print(f"Warning: Comment {i} missing author: {comment}")
            continue

        # Extract links from comment text
        comment_html = comment.get("comment_text", "")
        links = extract_links_from_html(comment_html)
        comment["links"] = links

        # Also add clean text version
        comment["comment_text_clean"] = extract_md_from_html(comment_html)

        # Add link count and domains for easy filtering
        comment["link_count"] = len(links)
        comment["linked_domains"] = list(set(link["domain"] for link in links))

        valid_comments.append(comment)

    print(f"Found {len(valid_comments)} valid comments")

    if not valid_comments:
        return []

    # First, identify top-level comments (indent_width = 0 or None)
    top_level_comments = []

    for comment in valid_comments:
        indent_width = comment.get("indent_width", "0")
        if not indent_width or indent_width == "0":
            comment["nesting_level"] = 0
            top_level_comments.append(comment)
        else:
            try:
                comment["nesting_level"] = int(indent_width) // 40
            except (ValueError, TypeError):
                print(
                    f"Warning: Invalid indent_width '{indent_width}' for comment {comment.get('comment_id')}"
                )
                comment["nesting_level"] = 0

    print(f"Found {len(top_level_comments)} top-level comments")

    # Now assign each comment to its thread based on document order
    comment_threads = []

    for i, top_comment in enumerate(top_level_comments):
        thread = {
            "thread_id": top_comment["comment_id"],
            "top_level_comment": top_comment.copy(),
            "all_replies": [],  # Flattened list of all descendants
        }

        # Find the range of comments that belong to this thread
        try:
            current_top_index = valid_comments.index(top_comment)
        except ValueError:
            print(
                f"Warning: Could not find top comment {top_comment.get('comment_id')} in valid_comments"
            )
            continue

        # Find next top-level comment index (or end of list)
        next_top_index = len(valid_comments)
        if i + 1 < len(top_level_comments):
            next_top_comment = top_level_comments[i + 1]
            try:
                next_top_index = valid_comments.index(next_top_comment)
            except ValueError:
                print(
                    f"Warning: Could not find next top comment {next_top_comment.get('comment_id')} in valid_comments"
                )

        # All comments between current and next top-level belong to this thread
        for j in range(current_top_index + 1, next_top_index):
            reply_comment = valid_comments[j].copy()
            # Remove indent_width from final output since we don't need it
            reply_comment.pop("indent_width", None)
            thread["all_replies"].append(reply_comment)

        # Remove indent_width from top-level comment too
        thread["top_level_comment"].pop("indent_width", None)

        # Add thread stats
        thread["total_replies"] = len(thread["all_replies"])
        thread["max_nesting_level"] = max(
            [r.get("nesting_level", 0) for r in thread["all_replies"]], default=0
        )

        # Aggregate all links in the thread
        all_thread_links = []
        all_thread_links.extend(thread["top_level_comment"].get("links", []))
        for reply in thread["all_replies"]:
            all_thread_links.extend(reply.get("links", []))

        # Remove duplicate links at thread level
        seen_urls = set()
        unique_thread_links = []
        for link in all_thread_links:
            if link["url"] not in seen_urls:
                seen_urls.add(link["url"])
                unique_thread_links.append(link)

        thread["all_links"] = unique_thread_links
        thread["total_links"] = len(unique_thread_links)
        thread["all_domains"] = list(
            set(link["domain"] for link in unique_thread_links)
        )

        comment_threads.append(thread)

    return comment_threads


async def hn_crawl_session(urls):
    if isinstance(urls, str):
        urls = [urls]

    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = Path("hn_sessions") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session_info = {
        "session_id": session_id,
        "created_at": datetime.now().isoformat(),
        "urls_count": len(urls),
        "successful_crawls": 0,
        "failed_urls": [],
        "post_ids": [],
    }
    with open(session_dir / "session_info.json", "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2, ensure_ascii=False)

    schema = {
        "name": "HN Post and All Comments",
        "baseSelector": "#hnmain",
        "fields": [
            {
                "name": "post",
                "selector": "table.fatitem",
                "type": "nested",
                "fields": [
                    {
                        "name": "post_id",
                        "selector": "tr.athing.submission",
                        "type": "attribute",
                        "attribute": "id",
                    },
                    {"name": "title", "selector": ".titleline > a", "type": "text"},
                    {
                        "name": "post_url",
                        "selector": ".titleline > a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                    {"name": "author", "selector": ".hnuser", "type": "text"},
                    {"name": "points", "selector": ".score", "type": "text"},
                    {"name": "age", "selector": ".age > a", "type": "text"},
                    {
                        "name": "comment_count",
                        "selector": ".subline a:last-child",
                        "type": "text",
                    },
                ],
            },
            {
                "name": "all_comments",
                "selector": "table.comment-tree tr.athing.comtr",
                "type": "nested_list",
                "fields": [
                    {"name": "author", "selector": ".hnuser", "type": "text"},
                    {"name": "age", "selector": ".age > a", "type": "text"},
                    {"name": "comment_text", "selector": ".commtext", "type": "html"},
                    {
                        "name": "indent_width",
                        "selector": ".ind > img",
                        "type": "attribute",
                        "attribute": "width",
                    },
                    {
                        "name": "permalink",
                        "selector": ".age > a",
                        "type": "attribute",
                        "attribute": "href",
                    },
                ],
            },
        ],
    }

    config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        extraction_strategy=JsonCssExtractionStrategy(schema),
    )

    async with AsyncWebCrawler() as crawler:
        for url in urls:
            try:
                print(f"Processing: {url}")
                match = re.search(r"id=(\d+)", url)
                post_id = match.group(1) if match else None
                if not post_id:
                    session_info["failed_urls"].append(url)
                    continue

                result = await crawler.arun(url=url, config=config)
                if not result.success:
                    session_info["failed_urls"].append(url)
                    continue

                raw_data = json.loads(result.extracted_content)
                if not raw_data:
                    session_info["failed_urls"].append(url)
                    continue

                data = raw_data[0]
                post = data.get("post", {})
                comments = data.get("all_comments", [])
                comment_threads = group_comments_by_thread(comments) if comments else []

                final_data = {
                    "url": url,
                    "post_id": post_id,
                    "post": post,
                    "comment_threads": comment_threads,
                    "total_threads": len(comment_threads),
                    "total_comments": len(comments),
                    "total_links": sum(
                        t.get("total_links", 0) for t in comment_threads
                    ),
                    "all_domains": list(
                        {d for t in comment_threads for d in t.get("all_domains", [])}
                    ),
                }

                with open(
                    session_dir / f"post_{post_id}.json", "w", encoding="utf-8"
                ) as f:
                    json.dump(final_data, f, indent=2, ensure_ascii=False)

                session_info["successful_crawls"] += 1
                session_info["post_ids"].append(post_id)

                print(f"✓ {post.get('title', 'Untitled')} ({len(comments)} comments)")
            except Exception as e:
                print(f"✗ Failed: {url} ({e})")
                session_info["failed_urls"].append(url)

    with open(session_dir / "session_info.json", "w", encoding="utf-8") as f:
        json.dump(session_info, f, indent=2, ensure_ascii=False)

    print(
        f"\n✔ Session complete: {session_dir} ({session_info['successful_crawls']} successful)"
    )
    return str(session_dir)


# if __name__ == "__main__":
#     urls = [
#         "https://news.ycombinator.com/item?id=42157556",
#         "https://news.ycombinator.com/item?id=44594475",
#         "https://news.ycombinator.com/item?id=44492290",
#     ]
#     asyncio.run(hn_crawl_session(urls))
