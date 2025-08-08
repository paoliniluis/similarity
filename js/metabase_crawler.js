const { BaseCrawler } = require('./base_crawler');
const { Client } = require('pg');

class MetabaseCrawler extends BaseCrawler {
    constructor(config = {}) {
        // Set default base URL for docs crawler
        config.baseUrl = "https://metabase.com";
        super(config);
        
        this.sitemapUrl = "https://metabase.com/sitemap.xml";
        
        console.log("✅ Metabase Crawler initialized for content extraction only!");
    }
    
    async fetchSitemap() {
        // Fetch the sitemap and extract URLs that contain /docs/ or /learn/
        try {
            console.log(`Fetching sitemap from ${this.sitemapUrl}...`);
            const response = await fetch(this.sitemapUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const xmlText = await response.text();
            
            // Parse XML sitemap (simple regex-based approach)
            const urls = [];
            const urlMatches = xmlText.match(/<loc>(.*?)<\/loc>/g);
            
            if (urlMatches) {
                for (const match of urlMatches) {
                    const url = match.replace(/<\/?loc>/g, '');
                    if ((url.includes('/docs/') || url.includes('/learn/')) && !url.includes('/api/')) {
                        urls.push(url);
                    }
                }
            }
            
            console.log(`Found ${urls.length} URLs containing /docs/ or /learn/`);
            return urls;
            
        } catch (error) {
            console.error(`Error fetching sitemap: ${error.message}`);
            return [];
        }
    }
    
    async processUrl(url, client) {
        // Process a single URL: fetch content, convert to markdown, and save to database
        try {
            // Check if URL already exists in database
            const existingResult = await client.query(
                'SELECT id FROM metabase_docs WHERE url = $1',
                [url]
            );
            
            if (existingResult.rows.length > 0) {
                console.log(`URL ${url} already exists in database, skipping...`);
                return { success: true, skipped: true };
            }
            
            // Apply rate limiting for content fetching
            await this.checkRateLimit();
            
            // Fetch HTML content
            const htmlContent = await this.fetchPageContent(url);
            if (!htmlContent) {
                console.log(`Failed to fetch content from ${url}`);
                return { success: false, error: 'Failed to fetch HTML content' };
            }
            
            // Convert HTML to markdown using Readability
            console.log(`Converting HTML to markdown for ${url}...`);
            const markdownContent = await this.htmlToMarkdown(htmlContent);
            
            if (!markdownContent) {
                console.log(`Failed to convert HTML to markdown for ${url}`);
                return { success: false, error: 'Failed to convert HTML to markdown' };
            }
            
            // Calculate token count (simple heuristic: 2 tokens per word)
            const tokenCount = markdownContent.trim() ? markdownContent.split(/\s+/).length * 2 : 0;
            
            // Save to database (embeddings will be processed separately by Python script)
            await client.query(
                `INSERT INTO metabase_docs (url, markdown, token_count, created_at, updated_at) 
                 VALUES ($1, $2, $3, NOW(), NOW())`,
                [url, markdownContent, tokenCount]
            );
            
            this.requestCount++;
            console.log(`✅ Content saved for ${url} (request #${this.requestCount})`);
            return { success: true, markdown: markdownContent, url };
            
        } catch (error) {
            console.error(`Error processing ${url}: ${error.message}`);
            return { success: false, error: error.message };
        }
    }
    
    async crawlAndSave() {
        // Main method to crawl sitemap, process URLs, and save to database
        console.log("Starting Metabase documentation crawl (content extraction only)...");
        
        // Connect to database
        const client = new Client(this.databaseUrl);
        
        try {
            await client.connect();
            console.log("✅ Connected to database");
            
            // Fetch URLs from sitemap
            const urls = await this.fetchSitemap();
            if (urls.length === 0) {
                console.log("No URLs found in sitemap");
                return 0;
            }
            
            // Process each URL (content extraction only)
            let successfulCount = 0;
            for (let i = 0; i < urls.length; i++) {
                const url = urls[i];
                console.log(`Processing URL ${i + 1}/${urls.length}: ${url}`);
                
                const result = await this.processUrl(url, client);
                if (result.success) {
                    successfulCount++;
                }
                
                // Add a small delay to be respectful to the server
                await this.sleep(this.rateLimit.delayBetweenRequests);
            }
            
            console.log(`\n🎉 Content extraction completed successfully!`);
            console.log(`📊 Summary:`);
            console.log(`  - Content extraction: ${successfulCount} documents`);
            console.log(`  - Requests made: ${this.requestCount}`);
            console.log(`  - Embeddings will be processed separately by Python script`);
            
            return successfulCount;
            
        } catch (error) {
            console.error(`❌ Error during crawl: ${error.message}`);
            throw error; // Re-throw the error so the caller can handle it
        } finally {
            try {
                await client.end();
            } catch (endError) {
                console.error(`Warning: Error closing database connection: ${endError.message}`);
            }
        }
    }
}

module.exports = { MetabaseCrawler }; 