#!/usr/bin/env python3
"""
Batch management CLI for OpenAI batch operations.
This script provides commands to create, submit, and monitor batch requests.
"""

import asyncio
import argparse
import logging
import sys
import os
from typing import Optional

# Use the shared path setup utility
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.path_setup import setup_project_path
setup_project_path()

from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db import SessionLocal
from src.models import BatchProcess
from src.batch_processor import BatchProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def create_and_submit_batch(operation_type: str, table_name: str):
    """Create and submit an efficient batch for the given operation and table."""
    logger.info(f"Creating and submitting efficient batch for {operation_type} on {table_name}")
    
    batch_processor = BatchProcessor()
    
    try:
        batch_id = await batch_processor.create_and_submit_batch(operation_type, table_name)
        
        if batch_id:
            logger.info(f"✅ Efficient batch created and submitted successfully: {batch_id}")
            logger.info("Use 'python scripts/batch_manager.py status' to monitor progress")
            logger.info("Note: This batch uses the new efficient batching system (100 entities per request)")
        else:
            logger.info("ℹ️  No items to process for this operation")
            
    except Exception as e:
        logger.error(f"❌ Error creating/submitting efficient batch: {e}")
        return False
    
    return True

async def check_batch_status(batch_id: Optional[str] = None):
    """Check status of a specific batch or all batches."""
    db = SessionLocal()
    try:
        if batch_id:
            # Check specific batch
            batch_process = db.query(BatchProcess).filter(BatchProcess.batch_id == batch_id).first()
            if not batch_process:
                logger.error(f"Batch {batch_id} not found")
                return
            
            batches = [batch_process]
        else:
            # Check all batches
            batches = db.query(BatchProcess).order_by(BatchProcess.created_at.desc()).limit(20).all()
        
        if not batches:
            logger.info("No batches found")
            return
        
        logger.info("Batch Status Report:")
        logger.info("=" * 100)
        
        for batch in batches:
            logger.info(f"Batch ID: {batch.batch_id}")
            logger.info(f"  Operation: {batch.operation_type}")
            logger.info(f"  Table: {batch.table_name}")
            logger.info(f"  Status: {batch.status}")
            logger.info(f"  Total Requests: {batch.total_requests}")
            logger.info(f"  Created: {batch.created_at}")
            sent_at = getattr(batch, 'sent_at', None)
            if sent_at:
                logger.info(f"  Sent: {sent_at}")
            received_at = getattr(batch, 'received_at', None)
            if received_at:
                logger.info(f"  Received: {received_at}")
            error_message = getattr(batch, 'error_message', None)
            if error_message:
                logger.info(f"  Error: {error_message}")
            logger.info("-" * 50)
            
    finally:
        db.close()

async def list_pending_items():
    """List items that need processing for each operation type."""
    db = SessionLocal()
    try:
        logger.info("Pending Items Report:")
        logger.info("=" * 50)
        
        # Check summarize operations
        for table_name in ["issues", "discourse_posts", "metabase_docs"]:
            count = 0
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
            
            logger.info(f"Summarize {table_name}: {count or 0} items")
        
        # Check questions operations
        for table_name in ["metabase_docs", "issues"]:
            count = 0
            if table_name == "metabase_docs":
                count = db.execute(
                    text("""
                        SELECT COUNT(*) FROM metabase_docs 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'METABASE_DOC'
                        ) AND markdown IS NOT NULL
                    """)
                ).scalar()
            elif table_name == "issues":
                count = db.execute(
                    text("""
                        SELECT COUNT(*) FROM issues 
                        WHERE id NOT IN (
                            SELECT DISTINCT source_id FROM questions WHERE source_type = 'ISSUE'
                        ) AND body IS NOT NULL AND labels::text LIKE '%feature request%'
                    """)
                ).scalar()
            
            logger.info(f"Questions {table_name}: {count or 0} items")
        
    finally:
        db.close()

async def main():
    """Main function to handle CLI commands."""
    parser = argparse.ArgumentParser(description="Efficient batch management for OpenAI operations")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create and submit an efficient batch')
    create_parser.add_argument('operation', choices=['summarize', 'create-questions', 'create-questions-and-concepts'],
                              help='Operation type')
    create_parser.add_argument('table', choices=['issues', 'discourse_posts', 'metabase_docs'],
                              help='Table name')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check batch status')
    status_parser.add_argument('--batch-id', help='Specific batch ID to check')
    
    # Pending command
    subparsers.add_parser('pending', help='List pending items for each operation')
    
    # Backfill command
    subparsers.add_parser('backfill-all', help='Create efficient batches for all pending operations')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        await create_and_submit_batch(args.operation, args.table)
    elif args.command == 'status':
        await check_batch_status(args.batch_id)
    elif args.command == 'pending':
        await list_pending_items()
    elif args.command == 'backfill-all':
        await backfill_all_operations()
    else:
        parser.print_help()

async def backfill_all_operations():
    """Create efficient batches for all pending operations in parallel."""
    logger.info("Starting efficient backfill of all pending operations...")
    logger.info("Using new efficient batching system (100 entities per request)")
    
    operations = [
        ('summarize', 'issues'),
        ('summarize', 'discourse_posts'),
        ('summarize', 'metabase_docs'),
        ('create-questions', 'metabase_docs'),
        ('create-questions', 'issues'),
        ('create-questions-and-concepts', 'metabase_docs'),
        ('create-questions-and-concepts', 'issues'),
    ]
    
    total = len(operations)
    
    # Create tasks for parallel execution
    tasks = []
    operation_names = []
    for operation_type, table_name in operations:
        operation_name = f"{operation_type} for {table_name}"
        operation_names.append(operation_name)
        tasks.append(create_and_submit_batch(operation_type, table_name))
        logger.info(f"Queued: {operation_name}")
    
    # Execute all operations in parallel
    logger.info(f"Running {total} operations in parallel...")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful = 0
    for i, (operation_name, result) in enumerate(zip(operation_names, results)):
        if isinstance(result, Exception):
            logger.error(f"Error in {operation_name}: {result}")
        elif result:
            successful += 1
            logger.info(f"✅ {operation_name} submitted successfully")
        else:
            logger.warning(f"⚠️ {operation_name} returned no batch ID")
    
    logger.info(f"Efficient backfill completed: {successful}/{total} operations submitted successfully")
    logger.info("Use 'python scripts/batch_manager.py status' to monitor progress")

if __name__ == "__main__":
    asyncio.run(main())
