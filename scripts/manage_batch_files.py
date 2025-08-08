#!/usr/bin/env python3
"""
Script to manage batch files using LiteLLM files endpoints.
Lists and optionally deletes files from batch processes.
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

from src.settings import OPENAI_API_BASE, OPENAI_API_KEY
from src.db import SessionLocal
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchFileManager:
    """Manages batch files using LiteLLM files endpoints."""
    
    def __init__(self):
        # Use configured API base and key (defaults defined in settings)
        self.api_base = str(OPENAI_API_BASE).rstrip("/")
        self.api_key = str(OPENAI_API_KEY) if OPENAI_API_KEY else None
        
        # Warn if no API key is configured when talking to OpenAI
        if not self.api_key and "localhost" not in str(self.api_base):
            logger.warning("No API key found. This may fail unless using a local proxy.")
        
        logger.info(f"Using API base: {self.api_base}")
        if self.api_key:
            logger.info("API key configured")
        else:
            logger.info("No API key (using proxy mode)")
    
    async def test_connection(self) -> bool:
        """Test if we can connect to the API."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try a simple health check first
                health_url = (
                    f"{self.api_base}/health"
                    if "localhost" in str(self.api_base)
                    else f"{self.api_base}/v1/models"
                )
                response = await client.get(
                    health_url,
                    headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
                )
                logger.info(f"Health check response: {response.status_code}")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    async def list_files(self, purpose: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all files, optionally filtered by purpose.
        
        Args:
            purpose: Optional purpose filter (e.g., 'batch', 'fine-tune')
            
        Returns:
            List of file objects
        """
        logger.info("Listing files...")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {}
                if purpose:
                    params['purpose'] = purpose
                
                response = await client.get(
                    f"{self.api_base}/v1/files",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    params=params
                )
                
                if response.status_code != 200:
                    error_msg = f"Failed to list files: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return []
                
                result = response.json()
                files = result.get('data', [])
                logger.info(f"Found {len(files)} files")
                return files
                
        except httpx.TimeoutException:
            logger.error("Timeout while listing files")
            return []
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []
    
    async def get_file_details(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get details for a specific file.
        
        Args:
            file_id: The file ID to get details for
            
        Returns:
            File details dict or None if error
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.api_base}/v1/files/{file_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get file details for {file_id}: {response.status_code} - {response.text}")
                    return None
                
                return response.json()
                
        except Exception as e:
            logger.error(f"Error getting file details for {file_id}: {e}")
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by ID.
        
        Args:
            file_id: The file ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.delete(
                    f"{self.api_base}/v1/files/{file_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                
                if response.status_code not in [200, 204]:
                    logger.error(f"Failed to delete file {file_id}: {response.status_code} - {response.text}")
                    return False
                
                logger.info(f"Successfully deleted file: {file_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {e}")
            return False
    
    def get_batch_processes_from_db(self) -> List[Dict[str, Any]]:
        """Get batch processes from database for correlation."""
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT id, batch_id, provider, operation_type, table_name, 
                           total_requests, status, sent_at, received_at,
                           input_file_path, output_file_path
                    FROM batch_processes 
                    ORDER BY sent_at DESC
                """)
            ).fetchall()
            
            return [dict(row._mapping) for row in result]
            
        except Exception as e:
            logger.error(f"Error getting batch processes from database: {e}")
            return []
        finally:
            db.close()
    
    def format_file_size(self, bytes_size: int) -> str:
        """Format file size in human readable format."""
        size = float(bytes_size)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def format_timestamp(self, timestamp: int) -> str:
        """Format Unix timestamp to readable date."""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)
    
    async def list_and_display_files(self, purpose: Optional[str] = None, show_details: bool = False):
        """List files and display them in a formatted way."""
        files = await self.list_files(purpose)
        
        if not files:
            print("No files found.")
            return files
        
        print(f"\n{'='*80}")
        print(f"Found {len(files)} files")
        print(f"{'='*80}")
        
        # Get batch processes for correlation
        batch_processes = self.get_batch_processes_from_db()
        batch_lookup = {bp['batch_id']: bp for bp in batch_processes}
        
        for i, file_obj in enumerate(files, 1):
            file_id = file_obj.get('id', 'N/A')
            filename = file_obj.get('filename', 'N/A')
            purpose = file_obj.get('purpose', 'N/A')
            bytes_size = file_obj.get('bytes', 0)
            created_at = file_obj.get('created_at', 0)
            
            print(f"\n{i}. File ID: {file_id}")
            print(f"   Filename: {filename}")
            print(f"   Purpose: {purpose}")
            print(f"   Size: {self.format_file_size(bytes_size)}")
            print(f"   Created: {self.format_timestamp(created_at)}")
            
            # Try to correlate with batch processes
            related_batches = []
            for batch_id, batch_info in batch_lookup.items():
                if (batch_info.get('input_file_path') and filename in batch_info['input_file_path']) or \
                   (batch_info.get('output_file_path') and filename in batch_info['output_file_path']):
                    related_batches.append(batch_info)
            
            if related_batches:
                print(f"   Related batches:")
                for batch in related_batches:
                    print(f"     - {batch['batch_id']} ({batch['operation_type']} {batch['table_name']}) - {batch['status']}")
            
            if show_details:
                details = await self.get_file_details(file_id)
                if details:
                    print(f"   Details: {details}")
        
        return files
    
    async def delete_files_interactive(self, files: List[Dict[str, Any]]):
        """Interactively delete files."""
        if not files:
            print("No files to delete.")
            return
        
        print(f"\n{'='*50}")
        print("INTERACTIVE FILE DELETION")
        print(f"{'='*50}")
        
        for i, file_obj in enumerate(files, 1):
            file_id = file_obj.get('id', 'N/A')
            filename = file_obj.get('filename', 'N/A')
            purpose = file_obj.get('purpose', 'N/A')
            bytes_size = file_obj.get('bytes', 0)
            
            print(f"\nFile {i}/{len(files)}:")
            print(f"  ID: {file_id}")
            print(f"  Name: {filename}")
            print(f"  Purpose: {purpose}")
            print(f"  Size: {self.format_file_size(bytes_size)}")
            
            while True:
                choice = input("Delete this file? (y/n/q to quit): ").strip().lower()
                if choice == 'y':
                    success = await self.delete_file(file_id)
                    if success:
                        print("✓ File deleted successfully")
                    else:
                        print("✗ Failed to delete file")
                    break
                elif choice == 'n':
                    print("Skipped")
                    break
                elif choice == 'q':
                    print("Quitting deletion process")
                    return
                else:
                    print("Please enter 'y', 'n', or 'q'")
    
    async def delete_files_by_pattern(self, files: List[Dict[str, Any]], pattern: str):
        """Delete files matching a pattern."""
        matching_files = []
        for file_obj in files:
            filename = file_obj.get('filename', '')
            if pattern.lower() in filename.lower():
                matching_files.append(file_obj)
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            return
        
        print(f"Found {len(matching_files)} files matching pattern '{pattern}':")
        for file_obj in matching_files:
            print(f"  - {file_obj.get('filename', 'N/A')} ({file_obj.get('id', 'N/A')})")
        
        confirm = input(f"\nDelete these {len(matching_files)} files? (y/N): ").strip().lower()
        if confirm == 'y':
            deleted_count = 0
            for file_obj in matching_files:
                file_id = file_obj.get('id')
                if file_id and await self.delete_file(file_id):
                    deleted_count += 1
            
            print(f"Deleted {deleted_count}/{len(matching_files)} files")
        else:
            print("Deletion cancelled")
    
    async def delete_old_files(self, files: List[Dict[str, Any]], days_old: int):
        """Delete files older than specified days."""
        import time
        
        cutoff_timestamp = time.time() - (days_old * 24 * 60 * 60)
        old_files = []
        
        for file_obj in files:
            created_at = file_obj.get('created_at', 0)
            if created_at < cutoff_timestamp:
                old_files.append(file_obj)
        
        if not old_files:
            print(f"No files found older than {days_old} days")
            return
        
        print(f"Found {len(old_files)} files older than {days_old} days:")
        for file_obj in old_files:
            filename = file_obj.get('filename', 'N/A')
            created_at = file_obj.get('created_at', 0)
            print(f"  - {filename} (created: {self.format_timestamp(created_at)})")
        
        confirm = input(f"\nDelete these {len(old_files)} old files? (y/N): ").strip().lower()
        if confirm == 'y':
            deleted_count = 0
            for file_obj in old_files:
                file_id = file_obj.get('id')
                if file_id and await self.delete_file(file_id):
                    deleted_count += 1
            
            print(f"Deleted {deleted_count}/{len(old_files)} files")
        else:
            print("Deletion cancelled")

async def main():
    parser = argparse.ArgumentParser(description="Manage batch files using LiteLLM files endpoints")
    parser.add_argument('--list', action='store_true', help='List all files')
    parser.add_argument('--purpose', type=str, help='Filter files by purpose (e.g., batch)')
    parser.add_argument('--details', action='store_true', help='Show detailed file information')
    parser.add_argument('--delete-interactive', action='store_true', help='Interactively delete files')
    parser.add_argument('--delete-pattern', type=str, help='Delete files matching pattern')
    parser.add_argument('--delete-older-than', type=int, help='Delete files older than N days')
    parser.add_argument('--delete-all-batch', action='store_true', help='Delete all batch purpose files (with confirmation)')
    parser.add_argument('--test-connection', action='store_true', help='Test API connection')
    parser.add_argument('--use-openai-direct', action='store_true', help='Use OpenAI API directly instead of proxy')
    parser.add_argument('--openai-api-key', type=str, help='OpenAI API key (if using direct mode)')
    
    args = parser.parse_args()
    
    if not any([args.list, args.delete_interactive, args.delete_pattern, 
                args.delete_older_than, args.delete_all_batch, args.test_connection]):
        parser.print_help()
        return
    
    manager = BatchFileManager()
    
    # Override settings if using direct OpenAI
    if args.use_openai_direct:
        manager.api_base = "https://api.openai.com"
        if args.openai_api_key:
            manager.api_key = args.openai_api_key
        elif not manager.api_key:
            print("Error: --openai-api-key required when using --use-openai-direct")
            return
        api_key_str = str(manager.api_key)
        masked_key = f"...{api_key_str[-4:]}" if len(api_key_str) > 4 else "short"
        print(f"Using direct OpenAI API with key: {masked_key}")
    
    try:
        # Test connection if requested or if performing operations
        if args.test_connection or any([args.list, args.delete_interactive, 
                                       args.delete_pattern, args.delete_older_than, args.delete_all_batch]):
            print("Testing API connection...")
            connection_ok = await manager.test_connection()
            if not connection_ok:
                print("❌ Connection test failed. Check your API configuration.")
                print(f"API Base: {manager.api_base}")
                print(f"API Key configured: {'Yes' if manager.api_key else 'No'}")
                if not args.test_connection:
                    print("Continuing anyway...")
            else:
                print("✅ Connection test successful")
            
            if args.test_connection:
                return
        
        if args.list or args.delete_interactive or args.delete_pattern or args.delete_older_than:
            files = await manager.list_and_display_files(args.purpose, args.details)
            
            if args.delete_interactive:
                await manager.delete_files_interactive(files)
            elif args.delete_pattern:
                await manager.delete_files_by_pattern(files, args.delete_pattern)
            elif args.delete_older_than:
                await manager.delete_old_files(files, args.delete_older_than)
        
        elif args.delete_all_batch:
            files = await manager.list_files('batch')
            if files:
                print(f"Found {len(files)} batch files:")
                for file_obj in files:
                    print(f"  - {file_obj.get('filename', 'N/A')} ({manager.format_file_size(file_obj.get('bytes', 0))})")
                
                confirm = input(f"\nDelete ALL {len(files)} batch files? (y/N): ").strip().lower()
                if confirm == 'y':
                    deleted_count = 0
                    for file_obj in files:
                        file_id = file_obj.get('id')
                        if file_id and await manager.delete_file(file_id):
                            deleted_count += 1
                    
                    print(f"Deleted {deleted_count}/{len(files)} batch files")
                else:
                    print("Deletion cancelled")
            else:
                print("No batch files found")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
