const { Readability } = require('@mozilla/readability');
const { isProbablyReaderable } = require('@mozilla/readability');
const { JSDOM } = require('jsdom');
const TurndownService = require('turndown');
const { Client } = require('pg');

class BaseCrawler {
    constructor(config = {}) {
        this.baseUrl = config.baseUrl || "https://metabase.com";
        
        // Database configuration - accept full DATABASE_URL
        this.databaseUrl = config.databaseUrl;
        
        // Validate database configuration
        if (!this.databaseUrl) {
            throw new Error('DATABASE_URL is required for BaseCrawler');
        }
        
        // Validate DATABASE_URL format
        try {
            new URL(this.databaseUrl);
        } catch (error) {
            throw new Error(`Invalid DATABASE_URL: ${this.databaseUrl}`);
        }
        
        // Rate limiting configuration
        this.rateLimit = {
            requestsPerMinute: config.requestsPerMinute || 50,
            delayBetweenRequests: config.delayBetweenRequests || 1000
        };
        
        // Validate rate limiting configuration
        if (this.rateLimit.requestsPerMinute <= 0) {
            throw new Error('Rate limiting configuration must be positive values');
        }
        
        // Rate limiting state
        this.requestCount = 0;
        this.lastRequestTime = 0;
        
        // Initialize Turndown for HTML to Markdown conversion
        this.turndownService = new TurndownService({
            headingStyle: 'atx',
            codeBlockStyle: 'fenced',
            emDelimiter: '*',
            bulletListMarker: '-',
            hr: '---'
        });
        
        // Configure Turndown for better markdown output
        this.turndownService.addRule('codeBlocks', {
            filter: ['pre'],
            replacement: function(content, node) {
                const code = node.querySelector('code');
                const language = code ? code.className.replace('language-', '') : '';
                return `\n\`\`\`${language}\n${content}\n\`\`\`\n`;
            }
        });
        
        this.turndownService.addRule('tables', {
            filter: 'table',
            replacement: function(content, node) {
                return `\n${content}\n`;
            }
        });
        
        console.log(`✅ ${this.constructor.name} initialized!`);
    }
    
    async sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
    
    async checkRateLimit() {
        const now = Date.now();
        const timeSinceLastRequest = now - this.lastRequestTime;
        const minInterval = 60000 / this.rateLimit.requestsPerMinute; // Convert to milliseconds
        
        if (timeSinceLastRequest < minInterval) {
            const waitTime = minInterval - timeSinceLastRequest;
            console.log(`⏳ Rate limiting: waiting ${waitTime}ms before next request`);
            await this.sleep(waitTime);
        }
        
        this.lastRequestTime = Date.now();
    }
    
    async fetchPageContent(url) {
        // Fetch the HTML content of a page
        try {
            // Sanitize and validate URL
            let sanitizedUrl;
            try {
                const urlObj = new URL(url);
                // Only allow HTTP and HTTPS protocols
                if (!['http:', 'https:'].includes(urlObj.protocol)) {
                    throw new Error('Invalid protocol');
                }
                // Only allow metabase.com domain
                if (!urlObj.hostname.includes('metabase.com')) {
                    throw new Error('Invalid domain');
                }
                sanitizedUrl = urlObj.toString();
            } catch (error) {
                console.error(`Invalid URL: ${url}`);
                return null;
            }
            
            console.log(`Fetching content from ${sanitizedUrl}...`);
            const headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            };
            
            const response = await fetch(sanitizedUrl, { headers });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.text();
        } catch (error) {
            console.error(`Error fetching content from ${url}: ${error.message}`);
            return null;
        }
    }
    
    async htmlToMarkdown(html) {
        // Convert HTML to markdown using Readability and Turndown
        try {
            // Create a DOM document using JSDOM
            const dom = new JSDOM(html, {
                url: this.baseUrl,
                runScripts: 'outside-only',
                resources: 'usable'
            });
            
            // Check if the page is probably readable
            if (!isProbablyReaderable(dom.window.document)) {
                console.log('⚠️  Page may not be suitable for Readability processing');
            }
            
            // Create a Readability object and parse the document
            const reader = new Readability(dom.window.document, {
                debug: false,
                charThreshold: 500,
                classesToPreserve: ['code', 'pre', 'blockquote'],
                keepClasses: false
            });
            
            const article = reader.parse();
            
            if (!article) {
                console.log('❌ Readability failed to parse the document');
                return "";
            }
            
            console.log(`✅ Readability parsing completed (${article.length} characters)`);
            
            // Convert HTML content to markdown using Turndown
            const markdown = this.turndownService.turndown(article.content);
            
            // Clean up the markdown
            const cleanedMarkdown = this.cleanMarkdown(markdown);
            
            return cleanedMarkdown;
            
        } catch (error) {
            console.error(`❌ Error converting HTML to markdown: ${error.message}`);
            return "";
        }
    }
    
    cleanMarkdown(markdown) {
        // Clean and improve the generated markdown
        if (!markdown) return '';
        
        return markdown
            // Remove excessive newlines
            .replace(/\n\s*\n\s*\n/g, '\n\n')
            // Remove leading/trailing whitespace from lines
            .replace(/^\s+/gm, '')
            .replace(/\s+$/gm, '')
            // Remove empty list items
            .replace(/^\s*[-*]\s*$/gm, '')
            // Clean up code blocks
            .replace(/```\s*\n/g, '```\n')
            .replace(/\n\s*```/g, '\n```')
            // Remove any incomplete markdown at the end
            .split('\n')
            .filter(line => {
                const trimmed = line.trim();
                return trimmed && !trimmed.startsWith('```') && !trimmed.endsWith('```');
            })
            .join('\n')
            .trim();
    }
    
    async crawlAndSave() {
        // Abstract method - must be implemented by subclasses
        throw new Error('crawlAndSave() must be implemented by subclasses');
    }
}

module.exports = { BaseCrawler }; 