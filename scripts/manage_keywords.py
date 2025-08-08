#!/usr/bin/env python3
"""
Command-line interface for managing keyword definitions.
Allows adding, updating, deleting, and listing keyword definitions.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.keyword_service import KeywordService
from src.db import SessionLocal

def add_keyword(keyword: str, definition: str, category: Optional[str] = None):
    """Add a new keyword definition."""
    service = KeywordService()
    
    if service.add_keyword_definition(keyword, definition, category):
        print(f"✓ Successfully added keyword: {keyword}")
    else:
        print(f"✗ Failed to add keyword: {keyword} (may already exist)")
        sys.exit(1)

def update_keyword(keyword: str, definition: str, category: Optional[str] = None):
    """Update an existing keyword definition."""
    service = KeywordService()
    
    if service.update_keyword_definition(keyword, definition, category):
        print(f"✓ Successfully updated keyword: {keyword}")
    else:
        print(f"✗ Failed to update keyword: {keyword} (not found)")
        sys.exit(1)

def delete_keyword(keyword: str):
    """Delete a keyword definition."""
    service = KeywordService()
    
    if service.delete_keyword_definition(keyword):
        print(f"✓ Successfully deleted keyword: {keyword}")
    else:
        print(f"✗ Failed to delete keyword: {keyword} (not found)")
        sys.exit(1)

def toggle_keyword(keyword: str):
    """Toggle the active status of a keyword definition."""
    service = KeywordService()
    
    if service.toggle_keyword_status(keyword):
        print(f"✓ Successfully toggled keyword: {keyword}")
    else:
        print(f"✗ Failed to toggle keyword: {keyword} (not found)")
        sys.exit(1)

def list_keywords(category: Optional[str] = None, show_inactive: bool = False):
    """List keyword definitions."""
    service = KeywordService()
    db = SessionLocal()
    
    try:
        keywords = service.list_keywords(db, category)
        
        if not keywords:
            print("No keyword definitions found.")
            return
        
        # Filter by active status if needed
        if not show_inactive:
            keywords = [kw for kw in keywords if kw['is_active'] == 'true']
        
        if not keywords:
            print("No active keyword definitions found.")
            return
        
        print(f"\nKeyword Definitions ({len(keywords)} found):")
        print("=" * 80)
        
        # Group by category
        categories = {}
        for kw in keywords:
            cat = kw.get('category', 'General')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(kw)
        
        for category_name, category_keywords in sorted(categories.items()):
            print(f"\n--- {category_name} ---")
            for kw in category_keywords:
                status = "✓" if kw['is_active'] == 'true' else "✗"
                print(f"{status} {kw['keyword']}")
                print(f"    Definition: {kw['definition']}")
                print(f"    Created: {kw['created_at']}")
                print(f"    Updated: {kw['updated_at']}")
                print()
    
    finally:
        db.close()

def search_keywords(search_term: str):
    """Search for keywords containing the search term."""
    service = KeywordService()
    db = SessionLocal()
    
    try:
        keywords = service.list_keywords(db)
        
        # Filter by search term
        matching_keywords = []
        for kw in keywords:
            if (search_term.lower() in kw['keyword'].lower() or 
                search_term.lower() in kw['definition'].lower() or
                (kw.get('category') and search_term.lower() in kw['category'].lower())):
                matching_keywords.append(kw)
        
        if not matching_keywords:
            print(f"No keywords found matching '{search_term}'")
            return
        
        print(f"\nSearch Results for '{search_term}' ({len(matching_keywords)} found):")
        print("=" * 80)
        
        for kw in matching_keywords:
            status = "✓" if kw['is_active'] == 'true' else "✗"
            category = f" [{kw['category']}]" if kw.get('category') else ""
            print(f"{status} {kw['keyword']}{category}")
            print(f"    Definition: {kw['definition']}")
            print()
    finally:
        db.close()


def show_stats():
    """Show statistics about keyword definitions."""
    service = KeywordService()
    db = SessionLocal()
    
    try:
        keywords = service.list_keywords(db)
        
        total_keywords = len(keywords)
        active_keywords = len([kw for kw in keywords if kw['is_active'] == 'true'])
        inactive_keywords = total_keywords - active_keywords
        
        print(f"\nKeyword Definitions Statistics:")
        print(f"  Total keywords: {total_keywords}")
        print(f"  Active keywords: {active_keywords}")
        print(f"  Inactive keywords: {inactive_keywords}")
        
        if total_keywords > 0:
            # Show categories
            categories = set(kw.get('category') for kw in keywords if kw.get('category'))
            if categories:
                print(f"  Categories: {', '.join(sorted(cat for cat in categories if cat is not None))}")
            
            # Show most recent keywords
            recent_keywords = sorted(keywords, key=lambda x: x['updated_at'], reverse=True)[:5]
            print(f"\nMost recently updated keywords:")
            for kw in recent_keywords:
                status = "✓" if kw['is_active'] == 'true' else "✗"
                category = f" [{kw['category']}]" if kw.get('category') else ""
                print(f"  {status} {kw['keyword']}{category}")
    
    finally:
        db.close()

def main():
    """Main entry point for the keyword management script."""
    parser = argparse.ArgumentParser(
        description="Manage keyword definitions for LLM context injection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_keywords.py --add "Metabase" "An open-source BI platform" --category "Product"
  python manage_keywords.py --update "Metabase" "Updated definition here"
  python manage_keywords.py --delete "Metabase"
  python manage_keywords.py --list
  python manage_keywords.py --list --category "Product"
  python manage_keywords.py --search "dashboard"
  python manage_keywords.py --stats
        """
    )
    
    # Action arguments
    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--add", nargs=2, metavar=("KEYWORD", "DEFINITION"), 
                             help="Add a new keyword definition")
    action_group.add_argument("--update", nargs=2, metavar=("KEYWORD", "DEFINITION"), 
                             help="Update an existing keyword definition")
    action_group.add_argument("--delete", metavar="KEYWORD", 
                             help="Delete a keyword definition")
    action_group.add_argument("--toggle", metavar="KEYWORD", 
                             help="Toggle the active status of a keyword")
    action_group.add_argument("--list", action="store_true", 
                             help="List all keyword definitions")
    action_group.add_argument("--search", metavar="TERM", 
                             help="Search for keywords containing the term")
    action_group.add_argument("--stats", action="store_true", 
                             help="Show statistics about keyword definitions")
    
    # Optional arguments
    parser.add_argument("--category", metavar="CATEGORY", 
                       help="Category for the keyword (used with --add or --update)")
    parser.add_argument("--show-inactive", action="store_true", 
                       help="Show inactive keywords when listing")
    
    args = parser.parse_args()
    
    try:
        if args.add:
            keyword, definition = args.add
            add_keyword(keyword, definition, args.category)
        
        elif args.update:
            keyword, definition = args.update
            update_keyword(keyword, definition, args.category)
        
        elif args.delete:
            delete_keyword(args.delete)
        
        elif args.toggle:
            toggle_keyword(args.toggle)
        
        elif args.list:
            list_keywords(args.category, args.show_inactive)
        
        elif args.search:
            search_keywords(args.search)
        
        elif args.stats:
            show_stats()
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 