#!/usr/bin/env python3
"""
Batch monitoring worker for OpenAI batch operations.
This worker continuously monitors submitted batches and processes completed ones.
"""

import asyncio
import logging
import time
from typing import List
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.db import SessionLocal
from src.models import BatchProcess
from src.batch_processor import BatchProcessor
from src.settings import WORKER_POLL_INTERVAL_SECONDS, WORKER_BACKOFF_SECONDS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchMonitorWorker:
    """Worker that monitors OpenAI batch operations."""

    def __init__(self, check_interval: int = WORKER_POLL_INTERVAL_SECONDS):
        self.check_interval = int(check_interval)
        self.batch_processor = BatchProcessor()
        self.running = False

    async def monitor_batches(self):
        """Monitor all pending batches."""
        self.running = True
        logger.info("Starting batch monitor worker...")
        
        while self.running:
            try:
                await self._check_pending_batches()
                logger.info(f"Sleeping for {self.check_interval} seconds...")
                await asyncio.sleep(self.check_interval)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in batch monitor: {e}")
                await asyncio.sleep(WORKER_BACKOFF_SECONDS)

    async def _check_pending_batches(self):
        """Check all batches that are in 'sent' status."""
        db = SessionLocal()
        try:
            # Get all batches that are sent but not completed
            pending_batches = db.query(BatchProcess).filter(
                BatchProcess.status.in_(['sent', 'in_progress', 'finalizing'])
            ).all()

            if not pending_batches:
                logger.info("No pending batches to check")
                return

            logger.info(f"Checking {len(pending_batches)} pending batches...")

            for batch_process in pending_batches:
                try:
                    await self._check_batch(db, batch_process)
                except Exception as e:
                    logger.error(f"Error checking batch {batch_process.batch_id}: {e}")
                    # Continue with the next batch instead of stopping
                    try:
                        db.execute(
                            text("UPDATE batch_processes SET status = 'error', error_message = :error WHERE id = :id"),
                            {"error": str(e), "id": batch_process.id}
                        )
                    except Exception as db_error:
                        logger.error(f"Failed to update batch {batch_process.batch_id} error status: {db_error}")

            db.commit()

        except Exception as e:
            logger.error(f"Error in _check_pending_batches: {e}")
            db.rollback()
        finally:
            db.close()

    async def _check_batch(self, db: Session, batch_process: BatchProcess):
        """Check a specific batch and process if completed."""
        batch_id = str(batch_process.batch_id)
        operation_type = str(batch_process.operation_type)
        table_name = str(batch_process.table_name)
        
        logger.info(f"Checking batch {batch_id} (operation: {operation_type}, table: {table_name})")
        
        try:
            # Get batch status from OpenAI
            batch_status = await self.batch_processor.check_batch_status(batch_id)
            
            current_status = batch_status.get('status', 'unknown')
            logger.info(f"Batch {batch_id} status: {current_status}")

            # Update status in database
            if current_status != batch_process.status:
                db.execute(
                    text("UPDATE batch_processes SET status = :status, updated_at = NOW() WHERE id = :id"),
                    {"status": current_status, "id": batch_process.id}
                )
                batch_process.status = current_status
                logger.info(f"Updated batch {batch_id} status to {current_status}")

            # If completed, download and process results
            if current_status == 'completed':
                await self._process_completed_batch(db, batch_process, batch_status)
            elif current_status in ['failed', 'expired', 'cancelled']:
                error_message = batch_status.get('errors', {}).get('message', f"Batch {current_status}")
                db.execute(
                    text("UPDATE batch_processes SET error_message = :error WHERE id = :id"),
                    {"error": error_message, "id": batch_process.id}
                )
                logger.error(f"Batch {batch_id} failed: {error_message}")

        except Exception as e:
            logger.error(f"Error checking batch {batch_id}: {e}")
            # Don't re-raise the exception - just log it and update status
            try:
                db.execute(
                    text("UPDATE batch_processes SET status = 'error', error_message = :error WHERE id = :id"),
                    {"error": str(e), "id": batch_process.id}
                )
            except Exception as db_error:
                logger.error(f"Failed to update batch {batch_id} error status: {db_error}")

    async def _process_completed_batch(self, db: Session, batch_process: BatchProcess, batch_status: dict):
        """Process a completed batch."""
        batch_id = str(batch_process.batch_id)
        operation_type = str(batch_process.operation_type)
        table_name = str(batch_process.table_name)
        
        logger.info(f"Processing completed batch {batch_id}")
        
        try:
            # Get output file ID
            output_file_id = batch_status.get('output_file_id')
            if not output_file_id:
                raise Exception("No output file ID in completed batch")

            # Download results
            output_file_path = await self.batch_processor.download_batch_results(
                batch_id, 
                output_file_id
            )

            # Process results and update database
            await self.batch_processor.process_batch_results(
                batch_id,
                output_file_path,
                operation_type,
                table_name
            )

            # Delete batch files from OpenAI after successful processing
            try:
                await self.batch_processor.delete_batch_files(batch_id)
            except Exception as e:
                logger.warning(f"Failed to delete batch files for {batch_id}: {e}")

            logger.info(f"Successfully processed completed batch {batch_id}")

        except Exception as e:
            logger.error(f"Error processing completed batch {batch_id}: {e}")
            db.execute(
                text("UPDATE batch_processes SET status = 'processing_failed', error_message = :error WHERE id = :id"),
                {"error": str(e), "id": batch_process.id}
            )

    def stop(self):
        """Stop the monitor worker."""
        self.running = False
        logger.info("Batch monitor worker stopped")

async def main():
    """Main function to run the batch monitor worker."""
    worker = BatchMonitorWorker()
    
    try:
        await worker.monitor_batches()
    except KeyboardInterrupt:
        logger.info("Shutting down batch monitor worker...")
        worker.stop()

if __name__ == "__main__":
    asyncio.run(main())
