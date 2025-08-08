"""
Shared path utilities to eliminate repetitive sys.path.append statements.
This should be imported by scripts that need to access the src module.
"""
import sys
from pathlib import Path

def setup_project_path():
    """Add the src directory to Python path if not already present."""
    project_root = Path(__file__).parent.parent
    src_path = str(project_root / "src")
    
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# Automatically setup path when this module is imported
setup_project_path()
