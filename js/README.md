# Metabase Crawlers

This directory contains refactored crawlers for extracting content from Metabase documentation and glossary.

## Architecture

The crawlers have been refactored to eliminate code duplication:

### BaseCrawler (`base_crawler.js`)
- **Shared functionality**: Database connection, rate limiting, HTML fetching, markdown conversion
- **Common utilities**: URL validation, error handling, logging patterns
- **Abstract class**: Must be extended by specific crawler implementations

### MetabaseCrawler (`metabase_crawler.js`)
- **Purpose**: Crawls Metabase documentation pages
- **Data source**: Sitemap XML parsing
- **Target table**: `metabase_docs`
- **Content processing**: Full HTML to markdown conversion using Readability

### MetabaseGlossaryCrawler (`metabase_glossary_crawler.js`)
- **Purpose**: Crawls Metabase glossary terms
- **Data source**: HTML scraping from glossary index page
- **Target table**: `keyword_definitions`
- **Content processing**: Definition extraction from HTML content

## Usage

### Unified Runner (`run_crawler.js`)
The new unified runner supports both crawler types:

```bash
# Run documentation crawler
node run_crawler.js docs

# Run glossary crawler
node run_crawler.js glossary
```

### NPM Scripts
```bash
# Documentation crawler
npm run crawl
npm run crawl-docs

# Glossary crawler
npm run glossary-crawl
npm run crawl-glossary
```

### Python Integration
The `run.py` script in the project root supports both crawlers:

```bash
# Documentation crawler
uv run run.py js-crawl docs

# Glossary crawler
uv run run.py js-crawl glossary
```

## Configuration

Both crawlers use the same environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `API_KEY`: API key for the service
- `REQUESTS_PER_MINUTE`: Rate limiting (default: 50 for docs, 30 for glossary)
- `DELAY_BETWEEN_BATCHES`: Delay between requests (default: 1000ms for docs, 2000ms for glossary)

## Benefits of Refactoring

1. **Eliminated Duplication**: ~200 lines of shared code moved to base class
2. **Consistent Behavior**: Rate limiting, error handling, and logging are now uniform
3. **Easier Maintenance**: Changes to shared functionality only need to be made in one place
4. **Unified Interface**: Single runner script handles both crawler types
5. **Better Organization**: Clear separation between shared and specific functionality

## File Structure

```
js/
├── base_crawler.js              # Base class with shared functionality
├── metabase_crawler.js          # Documentation crawler
├── metabase_glossary_crawler.js # Glossary crawler
├── run_crawler.js              # Unified runner script
├── package.json                 # Updated with new scripts
└── README.md                   # This documentation
```

 