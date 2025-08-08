#!/usr/bin/env python3
"""
Script to find potential duplicate issues based on similarity scores.
For every open issue, finds other similar issues (open or closed) with similarity > 80%.
Generates a markdown report with findings.
"""

import argparse
from datetime import datetime
from typing import List, Tuple, Dict
from sqlalchemy.orm import Session
from sqlalchemy import text
from src.db import SessionLocal
from src.models import Issue
from src.settings import GITHUB_REPO_OWNER, GITHUB_REPO_NAME

def find_similar_issues_for_issue(db: Session, target_issue: Issue, min_similarity: float = 0.8) -> List[Tuple[int, str, str, str, float]]:
    """
    Find similar issues for a given target issue.
    
    Args:
        db: Database session
        target_issue: The issue to find duplicates for
        min_similarity: Minimum similarity threshold (default 0.8 for 80%)
    
    Returns:
        List of tuples: (number, title, state, url, similarity_score)
    """
    
    # Get embeddings for the target issue
    has_issue_embedding = target_issue.issue_embedding is not None
    has_summary_embedding = target_issue.summary_embedding is not None
    
    if not has_issue_embedding and not has_summary_embedding:
        print(f"Warning: Issue #{target_issue.number} has no embeddings. Skipping.")
        return []
    
    # Use the best available embedding - prefer summary if available, otherwise use issue
    target_embedding = target_issue.summary_embedding if has_summary_embedding else target_issue.issue_embedding
    
    if target_embedding is None:
        return []
    
    # Convert embedding to SQL format
    embedding_str = ','.join(str(v) for v in target_embedding)
    embedding_sql = f"'[{embedding_str}]'::vector"
    
    # SQL query to find similar issues (excluding the target issue itself)
    sql = f"""
    WITH issue_sim AS (
        SELECT number, title, state, 1 - (issue_embedding <=> {embedding_sql}) AS similarity
        FROM issues
        WHERE issue_embedding IS NOT NULL 
        AND number != {target_issue.number}
        ORDER BY issue_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    summary_sim AS (
        SELECT number, title, state, 1 - (summary_embedding <=> {embedding_sql}) AS similarity
        FROM issues
        WHERE summary_embedding IS NOT NULL 
        AND number != {target_issue.number}
        ORDER BY summary_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    all_sim AS (
        SELECT * FROM issue_sim
        UNION ALL
        SELECT * FROM summary_sim
    )
    SELECT number, title, state, MAX(similarity) AS similarity
    FROM all_sim
    WHERE similarity >= {min_similarity}
    GROUP BY number, title, state
    ORDER BY similarity DESC
    LIMIT 20;
    """
    
    result = db.execute(text(sql))
    similar_issues = []
    
    for row in result:
        url = f"https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{row.number}"
        similar_issues.append((
            row.number,
            row.title,
            row.state,
            url,
            float(row.similarity)
        ))
    
    return similar_issues

def generate_markdown_report(duplicate_findings: Dict[Tuple[int, str], List[Tuple[int, str, str, str, float]]], output_file: str = "duplicate_issues_report.md") -> None:
    """
    Generate a markdown report with duplicate findings.
    
    Args:
        duplicate_findings: Dictionary mapping (issue_number, issue_title) to list of similar issues
        output_file: Output file path for the markdown report
    """
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Duplicate Issues Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Repository**: {GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}\n\n")
        f.write("This report shows open issues that have potential duplicates (similarity > 80%) among all issues (open or closed).\n\n")
        f.write("---\n\n")
        
        if not duplicate_findings:
            f.write("🎉 **No potential duplicates found!**\n\n")
            f.write("All open issues appear to be unique based on the 80% similarity threshold.\n")
            return
        
        total_open_issues_with_duplicates = len(duplicate_findings)
        total_duplicate_pairs = sum(len(duplicates) for duplicates in duplicate_findings.values())
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Open issues with potential duplicates**: {total_open_issues_with_duplicates}\n")
        f.write(f"- **Total potential duplicate relationships found**: {total_duplicate_pairs}\n\n")
        f.write("---\n\n")
        
        for (issue_number, issue_title), similar_issues in duplicate_findings.items():
            # Escape markdown special characters in titles
            safe_title = issue_title.replace('[', '\\[').replace(']', '\\]').replace('|', '\\|')
            
            f.write(f"## [{safe_title}](https://github.com/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues/{issue_number}) (#{issue_number})\n\n")
            
            f.write("**Potential duplicates:**\n\n")
            
            for similar_number, similar_title, similar_state, similar_url, similarity_score in similar_issues:
                # Escape markdown special characters
                safe_similar_title = similar_title.replace('[', '\\[').replace(']', '\\]').replace('|', '\\|')
                
                # Add state badge
                state_badge = "🟢 OPEN" if similar_state.lower() == 'open' else "🔴 CLOSED"
                
                f.write(f"- [{safe_similar_title}]({similar_url}) (#{similar_number}) - {state_badge} - **{similarity_score:.1%} similarity**\n")
            
            f.write("\n---\n\n")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Find potential duplicate issues and generate a report.")
    parser.add_argument("--min-similarity", type=float, default=0.8, 
                       help="Minimum similarity threshold (default: 0.8 for 80%%)")
    parser.add_argument("--output", "-o", type=str, default="duplicate_issues_report.md",
                       help="Output file for the markdown report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    print("🔍 Starting duplicate issue detection...")
    print(f"📊 Similarity threshold: {args.min_similarity:.1%}")
    
    # Connect to database
    db = SessionLocal()
    
    try:
        # Get all open issues
        open_issues = db.query(Issue).filter(Issue.state == 'open').all()
        print(f"📋 Found {len(open_issues)} open issues to analyze")
        
        if not open_issues:
            print("ℹ️  No open issues found in the database.")
            return
        
        duplicate_findings = {}
        issues_with_duplicates = 0
        
        for i, issue in enumerate(open_issues, 1):
            if args.verbose:
                print(f"🔎 Analyzing issue #{issue.number}: {str(issue.title)[:60]}..." + ("..." if len(str(issue.title)) > 60 else ""))
            elif i % 10 == 0:
                print(f"📈 Progress: {i}/{len(open_issues)} issues analyzed")
            
            similar_issues = find_similar_issues_for_issue(db, issue, args.min_similarity)
            
            if similar_issues:
                # Count open vs closed similar issues
                open_count = sum(1 for _, _, state, _, _ in similar_issues if state.lower() == 'open')
                closed_count = len(similar_issues) - open_count
                
                # Skip if all similar issues are closed (the open issue is effectively unique)
                if open_count == 0:
                    if args.verbose:
                        if len(similar_issues) == 1:
                            print(f"   ✅ Found 1 potential duplicate but it's closed - likely resolved, skipping")
                        else:
                            print(f"   ✅ Found {len(similar_issues)} potential duplicates but all are closed - issue is unique, skipping")
                else:
                    duplicate_findings[(issue.number, issue.title)] = similar_issues
                    issues_with_duplicates += 1
                    
                    if args.verbose:
                        print(f"   ⚠️  Found {len(similar_issues)} potential duplicates ({open_count} open, {closed_count} closed)")
        
        print(f"\n✅ Analysis complete!")
        print(f"📊 Summary:")
        print(f"   - Total open issues analyzed: {len(open_issues)}")
        print(f"   - Open issues with potential duplicates: {issues_with_duplicates}")
        print(f"   - Total potential duplicate relationships: {sum(len(dups) for dups in duplicate_findings.values())}")
        
        # Generate the markdown report
        print(f"\n📝 Generating markdown report: {args.output}")
        generate_markdown_report(duplicate_findings, args.output)
        print(f"✅ Report saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    main()