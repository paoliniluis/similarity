#!/usr/bin/env python3
"""
Alternative batch file manager that works with local files when API is unavailable.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.db import SessionLocal
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalBatchFileManager:
    """Manages batch files locally when API is not available."""
    
    def __init__(self):
        self.batch_dir = Path(__file__).parent.parent / "batch"
        self.batch_dir.mkdir(exist_ok=True)
        
        # Create sent and received subdirectories
        self.sent_dir = self.batch_dir / "sent"
        self.received_dir = self.batch_dir / "received"
        self.sent_dir.mkdir(exist_ok=True)
        self.received_dir.mkdir(exist_ok=True)
    
    def list_local_files(self) -> List[Dict[str, Any]]:
        """List all files in the batch directory and subdirectories."""
        files = []
        
        # Scan sent directory
        for file_path in self.sent_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "type": "sent"
                })
        
        # Scan received directory
        for file_path in self.received_dir.glob("*"):
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    "path": str(file_path),
                    "name": file_path.name,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "type": "received"
                })
        
        return sorted(files, key=lambda x: x["created"], reverse=True)
    
    def get_batch_processes_from_db(self) -> List[Dict[str, Any]]:
        """Get batch processes from database."""
        db = SessionLocal()
        try:
            result = db.execute(
                text("""
                    SELECT id, batch_id, provider, operation_type, table_name, 
                           total_requests, status, sent_at, received_at,
                           input_file_path, output_file_path, error_message
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
    
    def format_timestamp(self, timestamp: float) -> str:
        """Format timestamp to readable date."""
        try:
            return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        except:
            return str(timestamp)
    
    def analyze_file_content(self, file_path: Path) -> Dict[str, Any]:
        """Analyze the content of a batch file."""
        analysis = {
            "type": "unknown",
            "requests": 0,
            "operation": "unknown",
            "table": "unknown"
        }
        
        try:
            # Determine file type from extension and name
            if file_path.suffix == '.jsonl':
                if 'summarize' in file_path.name:
                    analysis["operation"] = "summarize"
                elif 'questions' in file_path.name:
                    analysis["operation"] = "questions"
                
                # Extract table name from filename
                for table in ['issues', 'discourse_posts', 'metabase_docs']:
                    if table in file_path.name:
                        analysis["table"] = table
                        break
                
                # Count requests
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        analysis["requests"] = sum(1 for line in f if line.strip())
                    
                    analysis["type"] = "input_batch" if "results_" not in file_path.name else "output_batch"
        except Exception as e:
            logger.warning(f"Error analyzing file {file_path}: {e}")
        
        return analysis
    
    def list_and_display_files(self, pattern: str | None = None):
        """List and display local batch files."""
        files = self.list_local_files()
        
        if pattern:
            files = [f for f in files if pattern.lower() in f["name"].lower()]
        
        if not files:
            print("No local batch files found.")
            return files
        
        # Get batch processes for correlation
        batch_processes = self.get_batch_processes_from_db()
        
        print(f"\n{'='*100}")
        print(f"Found {len(files)} local batch files")
        print(f"{'='*100}")
        
        for i, file_info in enumerate(files, 1):
            file_path = Path(file_info["path"])
            analysis = self.analyze_file_content(file_path)
            
            print(f"\n{i}. {file_info['name']}")
            print(f"   Path: {file_info['path']}")
            print(f"   Type: {file_info.get('type', 'unknown')} ({analysis['type']})")
            print(f"   Size: {self.format_file_size(file_info['size'])}")
            print(f"   Created: {self.format_timestamp(file_info['created'])}")
            print(f"   Modified: {self.format_timestamp(file_info['modified'])}")
            print(f"   Operation: {analysis['operation']}")
            print(f"   Table: {analysis['table']}")
            print(f"   Requests: {analysis['requests']}")
            
            # Find related batch process
            related_batch = None
            for bp in batch_processes:
                if (bp.get('input_file_path') and file_info['name'] in bp['input_file_path']) or \
                   (bp.get('output_file_path') and file_info['name'] in bp['output_file_path']):
                    related_batch = bp
                    break
            
            if related_batch:
                print(f"   Batch ID: {related_batch['batch_id']}")
                print(f"   Status: {related_batch['status']}")
                if related_batch.get('error_message'):
                    print(f"   Error: {related_batch['error_message']}")
        
        return files
    
    def delete_files_interactive(self, files: List[Dict[str, Any]]):
        """Interactively delete local files."""
        if not files:
            print("No files to delete.")
            return
        
        print(f"\n{'='*50}")
        print("INTERACTIVE FILE DELETION")
        print(f"{'='*50}")
        
        for i, file_info in enumerate(files, 1):
            file_path = Path(file_info["path"])
            
            print(f"\nFile {i}/{len(files)}:")
            print(f"  Name: {file_info['name']}")
            print(f"  Size: {self.format_file_size(file_info['size'])}")
            print(f"  Path: {file_info['path']}")
            
            while True:
                choice = input("Delete this file? (y/n/q to quit): ").strip().lower()
                if choice == 'y':
                    try:
                        file_path.unlink()
                        print("✓ File deleted successfully")
                        break
                    except Exception as e:
                        print(f"✗ Failed to delete file: {e}")
                        break
                elif choice == 'n':
                    print("Skipped")
                    break
                elif choice == 'q':
                    print("Quitting deletion process")
                    return
                else:
                    print("Please enter 'y', 'n', or 'q'")
    
    def delete_files_by_pattern(self, files: List[Dict[str, Any]], pattern: str):
        """Delete files matching a pattern."""
        matching_files = [f for f in files if pattern.lower() in f["name"].lower()]
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            return
        
        print(f"Found {len(matching_files)} files matching pattern '{pattern}':")
        for file_info in matching_files:
            print(f"  - {file_info['name']} ({self.format_file_size(file_info['size'])})")
        
        confirm = input(f"\nDelete these {len(matching_files)} files? (y/N): ").strip().lower()
        if confirm == 'y':
            deleted_count = 0
            for file_info in matching_files:
                try:
                    Path(file_info["path"]).unlink()
                    deleted_count += 1
                    print(f"✓ Deleted: {file_info['name']}")
                except Exception as e:
                    print(f"✗ Failed to delete {file_info['name']}: {e}")
            
            print(f"Deleted {deleted_count}/{len(matching_files)} files")
        else:
            print("Deletion cancelled")
    
    def delete_old_files(self, files: List[Dict[str, Any]], days_old: int):
        """Delete files older than specified days."""
        import time
        
        cutoff_timestamp = time.time() - (days_old * 24 * 60 * 60)
        old_files = [f for f in files if f["created"] < cutoff_timestamp]
        
        if not old_files:
            print(f"No files found older than {days_old} days")
            return
        
        print(f"Found {len(old_files)} files older than {days_old} days:")
        for file_info in old_files:
            print(f"  - {file_info['name']} (created: {self.format_timestamp(file_info['created'])})")
        
        confirm = input(f"\nDelete these {len(old_files)} old files? (y/N): ").strip().lower()
        if confirm == 'y':
            deleted_count = 0
            for file_info in old_files:
                try:
                    Path(file_info["path"]).unlink()
                    deleted_count += 1
                    print(f"✓ Deleted: {file_info['name']}")
                except Exception as e:
                    print(f"✗ Failed to delete {file_info['name']}: {e}")
            
            print(f"Deleted {deleted_count}/{len(old_files)} files")
        else:
            print("Deletion cancelled")
    
    def show_batch_status(self):
        """Show status of all batch processes."""
        batch_processes = self.get_batch_processes_from_db()
        
        if not batch_processes:
            print("No batch processes found in database.")
            return
        
        print(f"\n{'='*120}")
        print(f"Batch Processes Status ({len(batch_processes)} total)")
        print(f"{'='*120}")
        
        status_counts = {}
        for bp in batch_processes:
            status = bp.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
            
            print(f"\nBatch ID: {bp.get('batch_id', 'N/A')}")
            print(f"Operation: {bp.get('operation_type', 'N/A')} - {bp.get('table_name', 'N/A')}")
            print(f"Status: {bp.get('status', 'N/A')}")
            print(f"Requests: {bp.get('total_requests', 'N/A')}")
            print(f"Sent: {bp.get('sent_at', 'N/A')}")
            if bp.get('received_at'):
                print(f"Received: {bp['received_at']}")
            if bp.get('error_message'):
                print(f"Error: {bp['error_message']}")
            if bp.get('input_file_path'):
                input_exists = Path(bp['input_file_path']).exists() if bp['input_file_path'] else False
                print(f"Input file: {bp['input_file_path']} ({'exists' if input_exists else 'missing'})")
            if bp.get('output_file_path'):
                output_exists = Path(bp['output_file_path']).exists() if bp['output_file_path'] else False
                print(f"Output file: {bp['output_file_path']} ({'exists' if output_exists else 'missing'})")
        
        print(f"\n{'='*60}")
        print("Status Summary:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

def main():
    parser = argparse.ArgumentParser(description="Manage local batch files")
    parser.add_argument('--list', action='store_true', help='List all local batch files')
    parser.add_argument('--pattern', type=str, help='Filter files by pattern')
    parser.add_argument('--delete-interactive', action='store_true', help='Interactively delete files')
    parser.add_argument('--delete-pattern', type=str, help='Delete files matching pattern')
    parser.add_argument('--delete-older-than', type=int, help='Delete files older than N days')
    parser.add_argument('--status', action='store_true', help='Show batch process status from database')
    parser.add_argument('--clean-orphans', action='store_true', help='Delete files not referenced in database')
    
    args = parser.parse_args()
    
    if not any([args.list, args.delete_interactive, args.delete_pattern, 
                args.delete_older_than, args.status, args.clean_orphans]):
        parser.print_help()
        return
    
    manager = LocalBatchFileManager()
    
    try:
        if args.status:
            manager.show_batch_status()
            return
        
        if args.list or args.delete_interactive or args.delete_pattern or args.delete_older_than or args.clean_orphans:
            files = manager.list_and_display_files(args.pattern)
            
            if args.delete_interactive:
                manager.delete_files_interactive(files)
            elif args.delete_pattern:
                manager.delete_files_by_pattern(files, args.delete_pattern)
            elif args.delete_older_than:
                manager.delete_old_files(files, args.delete_older_than)
            elif args.clean_orphans:
                # Delete files not referenced in database
                batch_processes = manager.get_batch_processes_from_db()
                referenced_files = set()
                for bp in batch_processes:
                    if bp.get('input_file_path'):
                        referenced_files.add(Path(bp['input_file_path']).name)
                    if bp.get('output_file_path'):
                        referenced_files.add(Path(bp['output_file_path']).name)
                
                orphan_files = [f for f in files if f['name'] not in referenced_files]
                
                if orphan_files:
                    print(f"\nFound {len(orphan_files)} orphan files (not referenced in database):")
                    for file_info in orphan_files:
                        print(f"  - {file_info['name']}")
                    
                    confirm = input(f"\nDelete these {len(orphan_files)} orphan files? (y/N): ").strip().lower()
                    if confirm == 'y':
                        deleted_count = 0
                        for file_info in orphan_files:
                            try:
                                Path(file_info["path"]).unlink()
                                deleted_count += 1
                                print(f"✓ Deleted: {file_info['name']}")
                            except Exception as e:
                                print(f"✗ Failed to delete {file_info['name']}: {e}")
                        
                        print(f"Deleted {deleted_count}/{len(orphan_files)} orphan files")
                    else:
                        print("Deletion cancelled")
                else:
                    print("No orphan files found.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
