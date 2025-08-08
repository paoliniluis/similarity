#!/usr/bin/env python3
"""
Script to clean up OpenAI files from batch operations.
This script lists and optionally deletes files that are no longer needed.
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))

from src.settings import OPENAI_API_BASE, OPENAI_API_KEY, HTTPX_TIMEOUT
from src.db import SessionLocal
from src.batch_processor import BatchProcessor
from src.models import BatchProcess
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenAIFileCleaner:
    """Manages cleanup of OpenAI files from batch operations."""
    
    def __init__(self):
        self.api_base = OPENAI_API_BASE
        self.api_key = OPENAI_API_KEY
        self.batch_processor = BatchProcessor()
        
        logger.info(f"Using API base: {self.api_base}")
        
    async def list_files(self) -> List[Dict[str, Any]]:
        """List all files in OpenAI account."""
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                response = await client.get(
                    f"{self.api_base}/v1/files",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to list files: {response.status_code} - {response.text}")
                    return []
                
                files = response.json().get('data', [])
                logger.info(f"Found {len(files)} files in OpenAI account")
                return files
                
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    async def list_batches(self) -> List[Dict[str, Any]]:
        """List all batches in OpenAI account."""
        try:
            async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
                response = await client.get(
                    f"{self.api_base}/v1/batches",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to list batches: {response.status_code} - {response.text}")
                    return []
                
                batches = response.json().get('data', [])
                logger.info(f"Found {len(batches)} batches in OpenAI account")
                return batches
                
        except Exception as e:
            logger.error(f"Error listing batches: {e}")
            return []
    
    async def cleanup_orphaned_files(self, dry_run: bool = True) -> int:
        """Clean up files that are not associated with any active batches in our database."""
        logger.info("Starting cleanup of orphaned files from database batches...")
        
        # Get all files from OpenAI
        files = await self.list_files()
        
        if not files:
            logger.info("No files found to clean up")
            return 0
        
        # Get file IDs that are associated with batches in our database
        db = SessionLocal()
        try:
            # Get all batch processes from our database
            batch_processes = db.query(BatchProcess).all()
            
            # Get file IDs that are still in use by active batches
            active_file_ids = set()
            for batch_process in batch_processes:
                status = batch_process.status
                # Only consider active batches (not completed, failed, expired, cancelled)
                if status not in ['completed', 'failed', 'expired', 'cancelled']:
                    # For active batches, we need to get the file IDs from OpenAI
                    try:
                        batch_status = await self.batch_processor.check_batch_status(batch_process.batch_id)
                        input_file_id = batch_status.get('input_file_id')
                        output_file_id = batch_status.get('output_file_id')
                        if input_file_id:
                            active_file_ids.add(input_file_id)
                        if output_file_id:
                            active_file_ids.add(output_file_id)
                    except Exception as e:
                        logger.warning(f"Could not get file IDs for batch {batch_process.batch_id}: {e}")
            
            # Find orphaned files (files not associated with active batches in our database)
            orphaned_files = []
            for file_info in files:
                file_id = file_info.get('id')
                purpose = file_info.get('purpose', 'unknown')
                
                # Only consider batch-related files (batch, batch_output)
                if file_id and purpose in ['batch', 'batch_output'] and file_id not in active_file_ids:
                    orphaned_files.append(file_info)
            
            logger.info(f"Found {len(orphaned_files)} orphaned batch files out of {len(files)} total files")
            
            if not orphaned_files:
                logger.info("No orphaned batch files to clean up")
                return 0
            
            # Display orphaned files
            logger.info("Orphaned batch files:")
            for file_info in orphaned_files:
                file_id = file_info.get('id')
                filename = file_info.get('filename', 'unknown')
                created_at = file_info.get('created_at', 'unknown')
                purpose = file_info.get('purpose', 'unknown')
                logger.info(f"  - {file_id}: {filename} ({purpose}) created at {created_at}")
            
            if dry_run:
                logger.info("DRY RUN: Files would be deleted. Use --delete to actually delete them.")
                return len(orphaned_files)
            
            # Delete orphaned files
            deleted_count = 0
            for file_info in orphaned_files:
                file_id = file_info.get('id')
                filename = file_info.get('filename', 'unknown')
                
                try:
                    if await self.batch_processor.delete_file(file_id):
                        logger.info(f"Deleted file {file_id}: {filename}")
                        deleted_count += 1
                    else:
                        logger.warning(f"Failed to delete file {file_id}: {filename}")
                except Exception as e:
                    logger.error(f"Error deleting file {file_id}: {e}")
            
            logger.info(f"Successfully deleted {deleted_count} out of {len(orphaned_files)} orphaned batch files")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up orphaned files: {e}")
            return 0
        finally:
            db.close()
    
    async def cleanup_completed_batches(self, dry_run: bool = True) -> int:
        """Clean up files from completed batches that are recorded in our database."""
        logger.info("Starting cleanup of completed batch files from database...")
        
        # Get completed batches from our database
        db = SessionLocal()
        try:
            completed_batches = db.query(BatchProcess).filter(
                BatchProcess.status.in_(['completed', 'failed', 'expired', 'cancelled'])
            ).all()
            
            logger.info(f"Found {len(completed_batches)} completed batch processes in database")
            
            if not completed_batches:
                logger.info("No completed batches to clean up")
                return 0
            
            deleted_count = 0
            for batch_process in completed_batches:
                batch_id = batch_process.batch_id
                status = batch_process.status
                operation_type = batch_process.operation_type
                table_name = batch_process.table_name
                
                logger.info(f"Processing batch {batch_id} (status: {status}, operation: {operation_type}, table: {table_name})")
                
                if dry_run:
                    logger.info(f"DRY RUN: Would delete files for batch {batch_id}")
                    deleted_count += 1
                else:
                    try:
                        if await self.batch_processor.delete_batch_files(batch_id):
                            logger.info(f"Deleted files for batch {batch_id}")
                            deleted_count += 1
                        else:
                            logger.warning(f"Failed to delete files for batch {batch_id}")
                    except Exception as e:
                        logger.error(f"Error deleting files for batch {batch_id}: {e}")
            
            if dry_run:
                logger.info(f"DRY RUN: Would delete files for {deleted_count} completed batches from database")
            else:
                logger.info(f"Successfully deleted files for {deleted_count} completed batches from database")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up completed batches from database: {e}")
            return 0
        finally:
            db.close()

async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Clean up OpenAI files from batch operations")
    parser.add_argument("--list", action="store_true", help="List all files and batches")
    parser.add_argument("--cleanup-orphaned", action="store_true", help="Clean up orphaned files")
    parser.add_argument("--cleanup-completed", action="store_true", help="Clean up files from completed batches")
    parser.add_argument("--delete", action="store_true", help="Actually delete files (default is dry run)")
    parser.add_argument("--all", action="store_true", help="Run all cleanup operations")
    
    args = parser.parse_args()
    
    cleaner = OpenAIFileCleaner()
    
    if args.list or not any([args.cleanup_orphaned, args.cleanup_completed, args.all]):
        logger.info("Listing files and batches...")
        files = await cleaner.list_files()
        batches = await cleaner.list_batches()
        
        logger.info(f"Total files: {len(files)}")
        logger.info(f"Total batches: {len(batches)}")
        
        # Show batch status distribution
        status_counts = {}
        for batch in batches:
            status = batch.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        logger.info("Batch status distribution:")
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count}")
    
    if args.cleanup_orphaned or args.all:
        await cleaner.cleanup_orphaned_files(dry_run=not args.delete)
    
    if args.cleanup_completed or args.all:
        await cleaner.cleanup_completed_batches(dry_run=not args.delete)

if __name__ == "__main__":
    asyncio.run(main())
