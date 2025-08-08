#!/usr/bin/env python3
"""
Update batch file paths in database to reflect new directory structure.
"""

import sys
from pathlib import Path
from sqlalchemy import text

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.db import SessionLocal

def update_batch_file_paths():
    """Update batch file paths in database to use new sent/received structure."""
    db = SessionLocal()
    
    try:
        # Update input file paths (sent files)
        result = db.execute(
            text("""
                UPDATE batch_processes 
                SET input_file_path = REPLACE(input_file_path, '/batch/', '/batch/sent/')
                WHERE input_file_path LIKE '%/batch/%' 
                AND input_file_path NOT LIKE '%/batch/sent/%'
            """)
        )
        input_updated = result.rowcount
        print(f"Updated {input_updated} input file paths")
        
        # Update output file paths (received files)
        result = db.execute(
            text("""
                UPDATE batch_processes 
                SET output_file_path = REPLACE(output_file_path, '/batch/', '/batch/received/')
                WHERE output_file_path LIKE '%/batch/%' 
                AND output_file_path NOT LIKE '%/batch/received/%'
            """)
        )
        output_updated = result.rowcount
        print(f"Updated {output_updated} output file paths")
        
        db.commit()
        print("Database updated successfully")
        
    except Exception as e:
        print(f"Error updating database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    update_batch_file_paths()
