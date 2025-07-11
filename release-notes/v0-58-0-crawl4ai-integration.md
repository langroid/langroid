# Langroid Release 0.58.0

## 🎉 Major Features

### 🕷️ Crawl4AI Integration - Advanced Web Crawling with Browser Rendering

We're excited to introduce **Crawl4AI** as a new web crawling option in Langroid! This powerful crawler uses Playwright to render JavaScript-heavy websites, making it ideal for modern web applications.

#### Key Features:
- **Real Browser Rendering**: Handles dynamic content, SPAs, and JavaScript-heavy sites
- **No API Key Required**: Works locally without external dependencies
- **Multiple Extraction Strategies**:
  - CSS selector-based extraction for structured data
  - LLM-based extraction for unstructured content
  - Regex extraction for pattern matching
- **Advanced Markdown Generation**: Apply content filters to remove ads, sidebars, and irrelevant content
- **Deep Crawling**: Recursively crawl entire websites with customizable depth and filters
- **High Performance**: Optional LXML-based scraping for speed optimization

#### Installation:
```bash
pip install "langroid[crawl4ai]"
crawl4ai setup  # Note: Downloads Playwright browsers (~300MB, one-time)
crawl4ai doctor
```

#### Quick Example:
```python
from langroid.parsing.url_loader import URLLoader, Crawl4aiConfig

# Simple usage
config = Crawl4aiConfig()
loader = URLLoader(urls=["https://example.com"], crawler_config=config)
docs = loader.load()

# With extraction strategy
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy

css_strategy = JsonCssExtractionStrategy(schema={
    "name": "Articles",
    "baseSelector": "article",
    "fields": [
        {"name": "title", "selector": "h2", "type": "text"},
        {"name": "content", "selector": "p", "type": "text"}
    ]
})

config = Crawl4aiConfig(extraction_strategy=css_strategy)
loader = URLLoader(urls=["https://news.site.com"], crawler_config=config)
docs = loader.load()  # Returns structured JSON data
```

#### Using with DocChatAgent:
```python
# In chat_search.py or similar applications
python examples/docqa/chat_search.py -c crawl4ai
```

See the [full documentation](https://langroid.github.io/langroid/notes/crawl4ai/) for advanced usage including deep crawling, LLM-based extraction, and content filtering.

## 🔧 Improvements

### Enhanced URL Loader Framework
- Added `Crawl4aiConfig` to the URL loader configuration options
- Improved factory pattern to support multiple crawler backends
- Better separation between document URLs (PDF, DOCX) and web pages

### CLI Improvements
- `chat_search.py` now uses Fire instead of Typer for simpler CLI interface
- Updated help text to list all available crawlers: trafilatura, firecrawl, exa, crawl4ai

## 📚 Documentation
- Added comprehensive Crawl4AI documentation with examples
- Updated navigation in mkdocs.yml
- Added detailed examples in `examples/docqa/crawl4ai_examples.py`

## 🧪 Testing
- Added mocked tests for Crawl4AI functionality
- Added optional integration tests (skipped in CI to avoid Playwright download)
- Run integration tests locally with: `TEST_CRAWL4AI=1 pytest tests/main/test_url_loader.py::test_crawl4ai_integration`

## 🐛 Bug Fixes
- Fixed metadata extraction in crawl4ai implementation
- Improved error handling for missing crawl4ai dependencies
- Fixed import issues and duplicate code in examples

## 📦 Dependencies
- Added optional `crawl4ai>=0.6.3` dependency group
- No changes to core dependencies

## 🚀 Migration Guide
No breaking changes. To use the new Crawl4AI crawler:

1. Install the extra: `pip install "langroid[crawl4ai]"`
2. Run setup: `crawl4ai setup` (one-time Playwright download)
3. Use `Crawl4aiConfig()` instead of other crawler configs

## 🙏 Acknowledgments
Thanks to the contributors who helped improve this release, especially the integration of the powerful crawl4ai library for advanced web scraping capabilities.

---

**Full Changelog**: https://github.com/langroid/langroid/compare/v0.57.0...v0.58.0