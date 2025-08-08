#!/usr/bin/env python3
"""
Run Efficient Batches Script

This script demonstrates how to use the new efficient batching system for all
operations and tables. It shows the dramatic reduction in API calls and token usage.

Example usage:
    python scripts/run_efficient_batches.py --table issues --operation summarize
    python scripts/run_efficient_batches.py --table all --operation all
"""

import asyncio
import argparse
import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the src directory to the path
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Use the shared path setup utility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.path_setup import setup_project_path
setup_project_path()

from src.batch_processor import BatchProcessor
from src.db import SessionLocal
from sqlalchemy import text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EfficientBatchRunner:
    """Runner for efficient batch processing operations."""
    
    def __init__(self):
        self.batch_processor = BatchProcessor()
        self.tables = ["issues", "discourse_posts", "metabase_docs"]
        self.operations = ["summarize", "questions", "questions_and_concepts"]
    
    def get_processing_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive processing statistics for all tables and operations."""
        db = SessionLocal()
        try:
            stats = {}
            
            for table_name in self.tables:
                stats[table_name] = {}
                
                for operation_type in self.operations:
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
                    
                    stats[table_name][operation_type] = {
                        'total_items': count,
                        'estimated_batches': (count + 99) // 100,  # Round up division
                        'entities_per_batch': self.batch_processor.entities_per_batch
                    }
            
            return stats
            
        finally:
            db.close()
    
    def print_processing_stats(self, stats: Dict[str, Dict[str, Any]]):
        """Print comprehensive processing statistics."""
        logger.info("=" * 80)
        logger.info("EFFICIENT BATCH PROCESSING STATISTICS")
        logger.info("=" * 80)
        
        total_items = 0
        total_batches = 0
        total_api_calls_old = 0
        total_api_calls_new = 0
        
        for table_name, table_stats in stats.items():
            logger.info(f"\n{table_name.upper()}:")
            logger.info("-" * 40)
            
            for operation_type, operation_stats in table_stats.items():
                items = operation_stats['total_items']
                batches = operation_stats['estimated_batches']
                
                # Calculate API call reduction
                api_calls_old = items  # Old way: one call per item
                api_calls_new = batches  # New way: one call per batch
                reduction = ((api_calls_old - api_calls_new) / api_calls_old * 100) if api_calls_old > 0 else 0
                
                logger.info(f"  {operation_type}:")
                logger.info(f"    Items to process: {items:,}")
                logger.info(f"    Batches needed: {batches}")
                logger.info(f"    API calls (old way): {api_calls_old:,}")
                logger.info(f"    API calls (new way): {api_calls_new:,}")
                logger.info(f"    Reduction: {reduction:.1f}%")
                
                total_items += items
                total_batches += batches
                total_api_calls_old += api_calls_old
                total_api_calls_new += api_calls_new
        
        logger.info("\n" + "=" * 80)
        logger.info("TOTAL SUMMARY:")
        logger.info(f"  Total items to process: {total_items:,}")
        logger.info(f"  Total batches needed: {total_batches:,}")
        logger.info(f"  Total API calls (old way): {total_api_calls_old:,}")
        logger.info(f"  Total API calls (new way): {total_api_calls_new:,}")
        
        if total_api_calls_old > 0:
            total_reduction = ((total_api_calls_old - total_api_calls_new) / total_api_calls_old * 100)
            logger.info(f"  Total reduction: {total_reduction:.1f}%")
            logger.info(f"  Efficiency improvement: {total_api_calls_old / total_api_calls_new:.1f}x")
        
        logger.info("=" * 80)
    
    async def run_single_batch(self, table_name: str, operation_type: str) -> Optional[str]:
        """
        Run a single efficient batch for the specified table and operation.
        
        Args:
            table_name: Name of the table to process
            operation_type: Type of operation to perform
            
        Returns:
            Batch ID if successful, None otherwise
        """
        logger.info(f"Starting efficient batch for {operation_type} on {table_name}")
        
        start_time = time.time()
        
        try:
            batch_id = await self.batch_processor.create_and_submit_efficient_batch(
                operation_type, table_name
            )
            
            if batch_id:
                elapsed_time = time.time() - start_time
                logger.info(f"Successfully submitted batch {batch_id} in {elapsed_time:.2f}s")
                return batch_id
            else:
                logger.info(f"No {table_name} need {operation_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error running efficient batch for {table_name} {operation_type}: {e}")
            return None
    
    async def run_all_batches(self, tables: List[str], operations: List[str]) -> Dict[str, str]:
        """
        Run efficient batches for multiple tables and operations in parallel.
        
        Args:
            tables: List of table names to process
            operations: List of operation types to perform
            
        Returns:
            Dictionary mapping (table, operation) to batch IDs
        """
        # Create all combinations of tables and operations
        batch_tasks = []
        task_keys = []
        
        for table_name in tables:
            for operation_type in operations:
                key = f"{table_name}_{operation_type}"
                task_keys.append(key)
                batch_tasks.append(self.run_single_batch(table_name, operation_type))
                logger.info(f"Queued batch: {key}")
        
        # Run all batches in parallel with controlled concurrency
        logger.info(f"Running {len(batch_tasks)} batches in parallel...")
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        results = {}
        for i, (key, result) in enumerate(zip(task_keys, batch_results)):
            if isinstance(result, Exception):
                logger.error(f"Error processing batch {key}: {result}")
                results[key] = None
            else:
                results[key] = result
                if result:
                    logger.info(f"✅ Batch {key} submitted: {result}")
                else:
                    logger.warning(f"⚠️ Batch {key} returned no batch ID")
        
        return results
    
    async def monitor_batches(self, batch_results: Dict[str, str]):
        """
        Monitor all submitted batches and process results when complete.
        
        Args:
            batch_results: Dictionary mapping (table, operation) to batch IDs
        """
        logger.info(f"\nMonitoring {len(batch_results)} batches...")
        
        for key, batch_id in batch_results.items():
            table_name, operation_type = key.split('_', 1)
            
            logger.info(f"Monitoring batch {batch_id} for {key}...")
            
            try:
                # Check batch status
                status = await self.batch_processor.check_batch_status(batch_id)
                logger.info(f"Batch {batch_id} status: {status.get('status')}")
                
                if status.get('status') == 'completed':
                    # Download and process results
                    output_file_id = status.get('output_file_id')
                    if output_file_id:
                        output_file_path = await self.batch_processor.download_batch_results(
                            batch_id, output_file_id
                        )
                        
                        await self.batch_processor.process_efficient_batch_results(
                            batch_id, output_file_path, operation_type, table_name
                        )
                        
                        logger.info(f"Successfully processed batch {batch_id}")
                    else:
                        logger.error(f"No output file ID found for batch {batch_id}")
                else:
                    logger.info(f"Batch {batch_id} not yet completed")
                    
            except Exception as e:
                logger.error(f"Error monitoring batch {batch_id}: {e}")

async def main():
    """Main function to run efficient batch processing."""
    
    parser = argparse.ArgumentParser(description="Run efficient batch processing")
    parser.add_argument("--table", choices=["issues", "discourse_posts", "metabase_docs", "all"], 
                       default="all", help="Table to process")
    parser.add_argument("--operation", choices=["summarize", "questions", "questions_and_concepts", "all"],
                       default="all", help="Operation to perform")
    parser.add_argument("--stats-only", action="store_true", 
                       help="Only show statistics, don't run batches")
    parser.add_argument("--monitor", action="store_true",
                       help="Monitor existing batches instead of creating new ones")
    
    args = parser.parse_args()
    
    runner = EfficientBatchRunner()
    
    # Get processing statistics
    stats = runner.get_processing_stats()
    runner.print_processing_stats(stats)
    
    if args.stats_only:
        logger.info("Statistics only mode. Exiting.")
        return
    
    # Determine which tables and operations to process
    if args.table == "all":
        tables = runner.tables
    else:
        tables = [args.table]
    
    if args.operation == "all":
        operations = runner.operations
    else:
        operations = [args.operation]
    
    logger.info(f"Processing tables: {tables}")
    logger.info(f"Processing operations: {operations}")
    
    if args.monitor:
        # For monitoring, you would need to provide batch IDs
        logger.info("Monitoring mode requires batch IDs. Use the run_single_batch method instead.")
        return
    
    # Run efficient batches
    batch_results = await runner.run_all_batches(tables, operations)
    
    if batch_results:
        logger.info(f"\nSuccessfully submitted {len(batch_results)} batches:")
        for key, batch_id in batch_results.items():
            logger.info(f"  {key}: {batch_id}")
        
        # Optionally monitor batches (in a real scenario, this would be done separately)
        logger.info("\nTo monitor these batches, use the monitor_batches method")
        logger.info("Example:")
        logger.info("await runner.monitor_batches(batch_results)")
    else:
        logger.info("No batches were submitted (no items need processing)")

if __name__ == "__main__":
    asyncio.run(main())
