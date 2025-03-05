import os
import sys
from pathlib import Path

def setup_project_path():
    """
    Sets up the project path for imports.
    Should be called at the beginning of notebooks.
    """
    # Get the absolute path to the project root (where this file is located)
    project_root = str(Path(__file__).parent)
    
    # Add to Python path if not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added project root to Python path: {project_root}")
    
    return project_root

# Optional: Automatically set up the path when this module is imported
setup_project_path() 