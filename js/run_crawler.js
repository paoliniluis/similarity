// Load environment variables from .env file
// Look for .env in the parent directory (project root) since we're running from js/ directory
const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const { MetabaseCrawler } = require('./metabase_crawler');
const { MetabaseGlossaryCrawler } = require('./metabase_glossary_crawler');

async function main() {
    // Get command line arguments
    const args = process.argv.slice(2);
    const crawlType = args[0];
    
    if (!crawlType) {
        console.error('❌ Crawl type is required');
        console.error('Usage: node run_crawler.js <docs|glossary>');
        console.error('Examples:');
        console.error('  node run_crawler.js docs');
        console.error('  node run_crawler.js glossary');
        process.exit(1);
    }
    
    if (!['docs', 'glossary'].includes(crawlType)) {
        console.error(`❌ Invalid crawl type: ${crawlType}`);
        console.error('Available crawl types: docs, glossary');
        process.exit(1);
    }
    
    // Configuration - aligned with Python .env file
    const config = {
        // API configuration
        apiUrl: process.env.API_URL || 'http://localhost:8000',
        apiKey: process.env.API_KEY || 'your-api-key-here',
        
        // Database configuration - use DATABASE_URL directly
        databaseUrl: process.env.DATABASE_URL,
        
        // Rate limiting configuration
        requestsPerMinute: parseInt(process.env.REQUESTS_PER_MINUTE) || (crawlType === 'docs' ? 50 : 30),
        batchSize: parseInt(process.env.BATCH_SIZE) || 10,
        delayBetweenBatches: parseInt(process.env.DELAY_BETWEEN_BATCHES) || (crawlType === 'docs' ? 1000 : 2000)
    };
    
    console.log(`🚀 Starting Metabase ${crawlType === 'docs' ? 'Documentation' : 'Glossary'} Crawler`);
    console.log('=' .repeat(60));
    console.log(`Crawl Type: ${crawlType}`);
    console.log(`API URL: ${config.apiUrl}`);
    console.log(`Database URL: ${config.databaseUrl ? 'Set' : 'Not set'}`);
    console.log(`Rate Limit: ${config.requestsPerMinute} requests/minute`);
    console.log(`Batch Size: ${config.batchSize} documents per batch`);
    console.log(`Batch Delay: ${config.delayBetweenBatches}ms between batches`);
    console.log('=' .repeat(60));
    
    // Validate required configuration
    if (!config.apiKey || config.apiKey === 'your-api-key-here') {
        console.error('❌ Error: API_KEY environment variable is required');
        console.log('Please set the API_KEY environment variable with your API key');
        process.exit(1);
    }
    
    // Validate database configuration
    if (!config.databaseUrl) {
        console.error('❌ Error: DATABASE_URL environment variable is required');
        console.log('Please set the DATABASE_URL environment variable');
        process.exit(1);
    }
    
    try {
        // Test database connection first
        console.log('🔍 Testing database connection...');
        const { Client } = require('pg');
        const testClient = new Client(config.databaseUrl);
        
        await testClient.connect();
        console.log('✅ Database connection test successful');
        await testClient.end();
        
        // Create crawler instance based on type
        let crawler;
        if (crawlType === 'docs') {
            crawler = new MetabaseCrawler(config);
        } else {
            crawler = new MetabaseGlossaryCrawler(config);
        }
        
        // Run the crawler
        const processedCount = await crawler.crawlAndSave();
        
        console.log('\n' + '=' .repeat(60));
        console.log('📊 CRAWL SUMMARY');
        console.log('=' .repeat(60));
        if (crawlType === 'docs') {
            console.log(`✅ Successfully processed ${processedCount} documentation pages`);
        } else {
            console.log(`✅ Successfully processed ${processedCount} glossary keywords`);
        }
        console.log('🎉 Crawl completed successfully!');
        
        // Exit with success code
        process.exit(0);
        
    } catch (error) {
        console.error('\n' + '=' .repeat(60));
        console.error('❌ CRAWL FAILED');
        console.error('=' .repeat(60));
        console.error(`Error: ${error.message}`);
        
        // Provide helpful error messages for common issues
        if (error.message.includes('password authentication failed')) {
            console.error('\n💡 Database Connection Help:');
            console.error('   - Check your DATABASE_URL environment variable');
            console.error('   - Verify your database credentials');
            console.error('   - Ensure the database server is running');
            console.error('   - Try: export DATABASE_URL="your_connection_string"');
        } else if (error.message.includes('ECONNREFUSED')) {
            console.error('\n💡 Connection Help:');
            console.error('   - Check if the database server is running');
            console.error('   - Verify the host and port in your DATABASE_URL');
        } else if (error.message.includes('does not exist')) {
            console.error('\n💡 Database Help:');
            console.error('   - The database does not exist');
            console.error('   - Create the database or check the database name in DATABASE_URL');
        }
        
        process.exit(1);
    }
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
});

// Run the script
if (require.main === module) {
    main();
}

module.exports = { main }; 