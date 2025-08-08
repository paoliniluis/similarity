const { BaseCrawler } = require('./base_crawler');
const { JSDOM } = require('jsdom');
const { Client } = require('pg');

class MetabaseGlossaryCrawler extends BaseCrawler {
    constructor(config = {}) {
        // Set default base URL for glossary crawler
        config.baseUrl = "https://www.metabase.com";
        super(config);
        
        this.glossaryBaseUrl = "https://www.metabase.com/glossary";
        
        console.log("✅ Metabase Glossary Crawler initialized!");
    }
    
    async fetchGlossaryUrls() {
        // Fetch the main glossary page to find all glossary term URLs
        try {
            console.log(`Fetching glossary index from ${this.glossaryBaseUrl}...`);
            const response = await fetch(this.glossaryBaseUrl);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const htmlContent = await response.text();
            const dom = new JSDOM(htmlContent);
            const document = dom.window.document;
            
            // Find all links that point to glossary terms
            const links = document.querySelectorAll('a[href^="/glossary/"]');
            const urls = [];
            
            for (const link of links) {
                const href = link.getAttribute('href');
                if (href && href.startsWith('/glossary/')) {
                    const fullUrl = `${this.baseUrl}${href}`;
                    urls.push(fullUrl);
                }
            }
            
            // Remove duplicates
            const uniqueUrls = [...new Set(urls)];
            console.log(`Found ${uniqueUrls.length} unique glossary term URLs`);
            return uniqueUrls;
            
        } catch (error) {
            console.error(`Error fetching glossary URLs: ${error.message}`);
            return [];
        }
    }
    
    extractKeywordFromUrl(url) {
        // Extract the keyword from the URL path
        // URL format: https://www.metabase.com/glossary/term-name
        const urlParts = url.split('/');
        const lastPart = urlParts[urlParts.length - 1];
        
        // Replace dashes with spaces and capitalize words
        const keyword = lastPart
            .split('-')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
        
        return keyword;
    }
    
    extractDefinitionFromContent(htmlContent) {
        try {
            const dom = new JSDOM(htmlContent);
            const document = dom.window.document;
            
            // Look for content after "What is" or "Also known as" and before "Related terms"
            let definition = '';
            
            // Find the main content area
            const mainContent = document.querySelector('main') || document.querySelector('.content') || document.querySelector('article');
            if (!mainContent) {
                console.log('Could not find main content area');
                return null;
            }
            
            // Get all text content
            const textContent = mainContent.textContent || mainContent.innerText || '';
            
            // Look for patterns like "What is [term]?" or "Also known as"
            const whatIsPattern = /what is\s+([^?]+)\?/i;
            const alsoKnownAsPattern = /also known as\s+([^.]*)/i;
            
            let startIndex = 0;
            let endIndex = textContent.length;
            
            // Find the start of the definition
            const whatIsMatch = textContent.match(whatIsPattern);
            const alsoKnownAsMatch = textContent.match(alsoKnownAsPattern);
            
            if (whatIsMatch) {
                startIndex = textContent.indexOf(whatIsMatch[0]) + whatIsMatch[0].length;
            } else if (alsoKnownAsMatch) {
                startIndex = textContent.indexOf(alsoKnownAsMatch[0]) + alsoKnownAsMatch[0].length;
            }
            
            // If we found a "What is" pattern, look for the definition after it
            if (whatIsMatch) {
                // Find the next paragraph or content after the heading
                const headingElement = mainContent.querySelector('h1, h2, h3');
                if (headingElement) {
                    let currentElement = headingElement.nextElementSibling;
                    while (currentElement && currentElement !== mainContent) {
                        const elementText = currentElement.textContent.trim();
                        
                        // Skip "Also known as" elements
                        if (elementText.toLowerCase().includes('also known as')) {
                            currentElement = currentElement.nextElementSibling;
                            continue;
                        }
                        
                        // If we find a paragraph with substantial content, that's our definition
                        if (currentElement.tagName === 'P' && elementText.length > 20) {
                            definition = elementText;
                            break;
                        }
                        
                        currentElement = currentElement.nextElementSibling;
                    }
                }
            }
            
            // If we still don't have a definition, try the old method
            if (!definition) {
                // Find the end of the definition (before "Also known as" or "Related terms")
                const alsoKnownAsIndex = textContent.toLowerCase().indexOf('also known as');
                const relatedTermsIndex = textContent.toLowerCase().indexOf('related terms');
                
                if (alsoKnownAsIndex > startIndex && (endIndex === textContent.length || alsoKnownAsIndex < endIndex)) {
                    endIndex = alsoKnownAsIndex;
                }
                
                if (relatedTermsIndex > startIndex && relatedTermsIndex < endIndex) {
                    endIndex = relatedTermsIndex;
                }
                
                // Extract the definition from the text content
                definition = textContent.substring(startIndex, endIndex).trim();
                
                // Clean up the definition
                definition = definition
                    .replace(/\s+/g, ' ') // Replace multiple spaces with single space
                    .replace(/^\s+|\s+$/g, '') // Remove leading/trailing whitespace
                    .replace(/^[.,\s]+/, '') // Remove leading punctuation
                    .replace(/[.,\s]+$/, ''); // Remove trailing punctuation
            }
            
            return definition || null;
            
        } catch (error) {
            console.error(`Error extracting definition: ${error.message}`);
            return null;
        }
    }
    
    extractSynonymsFromContent(htmlContent) {
        try {
            const dom = new JSDOM(htmlContent);
            const document = dom.window.document;
            
            // Find the first element that contains "Also known as" text
            const allElements = document.querySelectorAll('*');
            let alsoKnownAsElement = null;
            
            for (const element of allElements) {
                const elementText = element.textContent.trim();
                if (elementText.toLowerCase().includes('also known as')) {
                    // Check if this element directly contains "Also known as" text
                    // and is not a parent element that contains all text
                    if (elementText.toLowerCase() === 'also known as' || 
                        elementText.toLowerCase().startsWith('also known as')) {
                        alsoKnownAsElement = element;
                        break;
                    }
                }
            }
            
            if (!alsoKnownAsElement) {
                return [];
            }
            
            // Find the next element after the "Also known as" element
            let currentElement = alsoKnownAsElement.nextElementSibling;
            
            while (currentElement) {
                // Look for spans in this element
                const spanElements = currentElement.querySelectorAll('span');
                if (spanElements.length > 0) {
                    const firstSpan = spanElements[0];
                    const synonymText = firstSpan.textContent.trim();
                    
                    if (synonymText && synonymText.length > 0) {
                        // Clean up the synonym
                        const cleanedSynonym = synonymText
                            .replace(/^\s+|\s+$/g, '') // Remove leading/trailing whitespace
                            .replace(/^["']|["']$/g, '') // Remove quotes
                            .replace(/\s+/g, ' '); // Replace multiple spaces with single space
                        
                        return [cleanedSynonym];
                    }
                }
                
                currentElement = currentElement.nextElementSibling;
            }
            return [];
            
        } catch (error) {
            console.error(`Error extracting synonyms: ${error.message}`);
            return [];
        }
    }
    
    async processGlossaryUrl(url, client) {
        try {
            // Extract keyword from URL
            const keyword = this.extractKeywordFromUrl(url);
            
            // Check if keyword already exists in database
            const existingResult = await client.query(
                'SELECT id FROM keyword_definitions WHERE keyword = $1',
                [keyword]
            );
            
            if (existingResult.rows.length > 0) {
                console.log(`Keyword "${keyword}" already exists in database, updating with glossary definition...`);
            }
            
            // Apply rate limiting
            await this.checkRateLimit();
            
            // Fetch HTML content
            const htmlContent = await this.fetchPageContent(url);
            if (!htmlContent) {
                console.log(`Failed to fetch content from ${url}`);
                return { success: false, error: 'Failed to fetch HTML content' };
            }
            
            // Extract definition from content
            const definition = this.extractDefinitionFromContent(htmlContent);
            if (!definition) {
                console.log(`Failed to extract definition from ${url}`);
                return { success: false, error: 'Failed to extract definition' };
            }
            
            // Extract synonyms from content
            const synonyms = this.extractSynonymsFromContent(htmlContent);
            
            // Save or update keyword definition to database
            if (existingResult.rows.length > 0) {
                // Update existing entry with glossary definition
                await client.query(
                    `UPDATE keyword_definitions 
                     SET definition = $1, category = $2, updated_at = NOW() 
                     WHERE keyword = $3`,
                    [definition, 'Glossary', keyword]
                );
            } else {
                // Insert new entry
                await client.query(
                    `INSERT INTO keyword_definitions (keyword, definition, category, is_active, created_at, updated_at) 
                     VALUES ($1, $2, $3, $4, NOW(), NOW())`,
                    [keyword, definition, 'Glossary', 'true']
                );
            }
            
            // Save synonyms to database
            let synonymsSaved = 0;
            for (const synonym of synonyms) {
                try {
                    // Check if this synonym already exists for this keyword
                    const existingSynonymResult = await client.query(
                        'SELECT id FROM synonyms WHERE word = $1 AND synonym_of = $2',
                        [synonym, keyword]
                    );
                    
                    if (existingSynonymResult.rows.length === 0) {
                        await client.query(
                            `INSERT INTO synonyms (word, synonym_of, created_at, updated_at) 
                             VALUES ($1, $2, NOW(), NOW())`,
                            [synonym, keyword]
                        );
                        synonymsSaved++;
                    }
                } catch (synonymError) {
                    console.error(`Error saving synonym "${synonym}" for "${keyword}": ${synonymError.message}`);
                }
            }
            
            this.requestCount++;
            const action = existingResult.rows.length > 0 ? 'Updated' : 'Saved';
            console.log(`✅ ${action} definition for "${keyword}" with ${synonymsSaved} synonyms (request #${this.requestCount})`);
            return { success: true, keyword, definition, synonyms, url, updated: existingResult.rows.length > 0 };
            
        } catch (error) {
            console.error(`Error processing ${url}: ${error.message}`);
            return { success: false, error: error.message };
        }
    }
    
    async crawlAndSave() {
        // Main method to crawl glossary, process URLs, and save to database
        console.log("Starting Metabase glossary crawl...");
        
        // Connect to database
        const client = new Client(this.databaseUrl);
        
        try {
            await client.connect();
            console.log("✅ Connected to database");
            
            // Fetch glossary URLs
            const urls = await this.fetchGlossaryUrls();
            if (urls.length === 0) {
                console.log("No glossary URLs found");
                return 0;
            }
            
            // Process each URL
            let successfulCount = 0;
            for (let i = 0; i < urls.length; i++) {
                const url = urls[i];
                console.log(`Processing URL ${i + 1}/${urls.length}: ${url}`);
                
                const result = await this.processGlossaryUrl(url, client);
                if (result.success) {
                    successfulCount++;
                }
                
                // Add a small delay to be respectful to the server
                await this.sleep(this.rateLimit.delayBetweenRequests);
            }
            
            console.log(`\n🎉 Glossary crawl completed successfully!`);
            console.log(`📊 Summary:`);
            console.log(`  - Keywords processed: ${successfulCount}`);
            console.log(`  - Requests made: ${this.requestCount}`);
            console.log(`  - Synonyms will be extracted and saved to the synonyms table`);
            
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

module.exports = { MetabaseGlossaryCrawler }; 