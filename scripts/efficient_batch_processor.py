#!/usr/bin/env python3
"""
Efficient Batch Processor Script

This script demonstrates how to use the new efficient batching system that groups
multiple entities together in a single API call, reducing token usage and API calls.

Instead of:
- base prompts, id1, issue1
- base prompts, id2, issue2  
- base prompts, id3, issue3

We now do:
- base prompt + summarize all issues below
- id1, issue1
- id2, issue2
- id3, issue3

This reduces API calls from N to N/100 (where 100 is entities_per_batch).
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.batch_processor import BatchProcessor
from src.db import SessionLocal
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EfficientBatchProcessor:
    """Wrapper for efficient batch processing operations."""
    
    def __init__(self):
        self.batch_processor = BatchProcessor()
    
    async def process_table_efficiently(self, table_name: str, operation_type: str) -> Optional[str]:
        """
        Process a table using the efficient batching system.
        
        Args:
            table_name: Name of the table to process (issues, discourse_posts, metabase_docs)
            operation_type: Type of operation (summarize, questions, questions_and_concepts)
            
        Returns:
            Batch ID if successful, None otherwise
        """
        logger.info(f"Starting efficient batch processing for {operation_type} on {table_name}")
        
        try:
            # Create and submit efficient batch
            batch_id = await self.batch_processor.create_and_submit_efficient_batch(
                operation_type, table_name
            )
            
            if batch_id:
                logger.info(f"Successfully submitted efficient batch {batch_id}")
                return batch_id
            else:
                logger.info(f"No {table_name} need {operation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error in efficient batch processing: {e}")
            return None
    
    async def monitor_and_process_batch(self, batch_id: str, operation_type: str, table_name: str):
        """
        Monitor a batch and process results when complete.
        
        Args:
            batch_id: OpenAI batch ID
            operation_type: Type of operation performed
            table_name: Name of the table processed
        """
        logger.info(f"Monitoring batch {batch_id}...")
        
        try:
            # Check batch status
            status = await self.batch_processor.check_batch_status(batch_id)
            logger.info(f"Batch status: {status}")
            
            if status.get('status') == 'completed':
                # Download results
                output_file_id = status.get('output_file_id')
                if output_file_id:
                    output_file_path = await self.batch_processor.download_batch_results(
                        batch_id, output_file_id
                    )
                    
                    # Process results using the efficient processor
                    await self.batch_processor.process_efficient_batch_results(
                        batch_id, output_file_path, operation_type, table_name
                    )
                    
                    logger.info(f"Successfully processed batch {batch_id}")
                else:
                    logger.error(f"No output file ID found for batch {batch_id}")
            else:
                logger.info(f"Batch {batch_id} not yet completed. Status: {status.get('status')}")
                
        except Exception as e:
            logger.error(f"Error monitoring batch {batch_id}: {e}")
    
    def get_processing_stats(self, table_name: str, operation_type: str) -> Dict[str, Any]:
        """
        Get statistics about what needs to be processed.
        
        Args:
            table_name: Name of the table
            operation_type: Type of operation
            
        Returns:
            Dictionary with processing statistics
        """
        db = SessionLocal()
        try:
            stats = {}
            
            if operation_type == "summarize":
                if table_name == "issues":
                    count = db.execute(
                        text("SELECT COUNT(*) FROM issues WHERE llm_summary IS NULL AND body IS NOT NULL")
                    ).scalar()
                elif table_name == "discourse_posts":
                    count = db.execute(
                        text("SELECT COUNT(*) FROM discourse_posts WHERE llm_summary IS NULL AND conversation IS NOT NULL")
                    ).scalar()
                elif table_name == "metabase_docs":
                    count = db.execute(
                        text("SELECT COUNT(*) FROM metabase_docs WHERE llm_summary IS NULL AND markdown IS NOT NULL")
                    ).scalar()
                else:
                    count = 0
                    
                stats['total_items'] = count
                stats['estimated_batches'] = (count + 99) // 100  # Round up division
                
            elif operation_type in ["questions", "questions_and_concepts"]:
                if table_name == "issues":
                    count = db.execute(
                        text("""
                            SELECT COUNT(*) FROM issues 
                            WHERE id NOT IN (
                                SELECT DISTINCT source_id FROM questions WHERE source_type = 'ISSUE'
                            ) AND body IS NOT NULL AND labels::text LIKE '%feature request%'
                        """)
                    ).scalar()
                elif table_name == "discourse_posts":
                    count = db.execute(
                        text("""
                            SELECT COUNT(*) FROM discourse_posts 
                            WHERE id NOT IN (
                                SELECT DISTINCT source_id FROM questions WHERE source_type = 'DISCOURSE_POST'
                            ) AND conversation IS NOT NULL
                        """)
                    ).scalar()
                elif table_name == "metabase_docs":
                    count = db.execute(
                        text("""
                            SELECT COUNT(*) FROM metabase_docs 
                            WHERE id NOT IN (
                                SELECT DISTINCT source_id FROM questions WHERE source_type = 'METABASE_DOC'
                            ) AND markdown IS NOT NULL
                        """)
                    ).scalar()
                else:
                    count = 0
                    
                stats['total_items'] = count
                stats['estimated_batches'] = (count + 99) // 100  # Round up division
            
            return stats
            
        finally:
            db.close()

async def main():
    """Main function to demonstrate efficient batch processing."""
    
    processor = EfficientBatchProcessor()
    
    # Example: Process issues for summarization
    table_name = "issues"
    operation_type = "summarize"
    
    # Get processing statistics
    stats = processor.get_processing_stats(table_name, operation_type)
    logger.info(f"Processing stats for {table_name} {operation_type}:")
    logger.info(f"  Total items to process: {stats['total_items']}")
    logger.info(f"  Estimated batches needed: {stats['estimated_batches']}")
    logger.info(f"  Entities per batch: {processor.batch_processor.entities_per_batch}")
    
    if stats['total_items'] == 0:
        logger.info("No items need processing. Exiting.")
        return
    
    # Submit efficient batch
    batch_id = await processor.process_table_efficiently(table_name, operation_type)
    
    if batch_id:
        logger.info(f"Batch submitted successfully: {batch_id}")
        
        # In a real scenario, you would monitor this batch
        # For demonstration, we'll just log the batch ID
        logger.info("To monitor this batch, use the monitor_and_process_batch method")
        logger.info("Example:")
        logger.info(f"await processor.monitor_and_process_batch('{batch_id}', '{operation_type}', '{table_name}')")
    else:
        logger.info("No batch was submitted (no items to process)")

if __name__ == "__main__":
    asyncio.run(main())
