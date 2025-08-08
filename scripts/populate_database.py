import requests
import time
import argparse
import json
from datetime import datetime
from typing import List, Dict, Any, Generator, Optional
from tqdm import tqdm
from sqlalchemy.orm import Session
from sqlalchemy import or_, MetaData, text
from src.db import SessionLocal, engine
from src.models import Issue, DiscoursePost
from src.settings import GITHUB_TOKEN, GITHUB_REPO_OWNER, GITHUB_REPO_NAME, DISCOURSE_BASE_URL, DISCOURSE_API_KEY, DISCOURSE_API_USERNAME, DISCOURSE_MAX_PAGES
from src.llm_analyzer import LLMAnalyzer
from src.embedding_service import get_embedding_service
from src.text_utils import combine_discourse_posts, get_topic_creator_username, combine_all_discourse_posts, calculate_token_count


def refresh_sqlalchemy_metadata():
    """Force SQLAlchemy to refresh its metadata cache."""
    try:
        # Clear any cached metadata
        metadata = MetaData()
        metadata.reflect(bind=engine)
        print("SQLAlchemy metadata refreshed successfully")
    except Exception as e:
        print(f"Warning: Could not refresh SQLAlchemy metadata: {e}")


def stream_github_issue_pages() -> Generator[List[Dict[str, Any]], None, None]:
    """
    Fetches all issues from the specified GitHub repository page by page.
    Handles pagination and primary rate limiting.
    Yields each page of issues as a list of dictionaries.
    """
    url = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}/issues"
    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"token {GITHUB_TOKEN}"
    else:
        print("Warning: GITHUB_TOKEN not set. Making unauthenticated requests.")
    
    params = {"state": "all", "per_page": 100}

    page_num = 1
    while url:
        try:
            print(f"\nFetching page {page_num}...")
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            page_of_issues = response.json()
            
            if not page_of_issues:
                print("No more issues found on this page. Concluding fetch.")
                break
            
            print(f"Found {len(page_of_issues)} issues on page {page_num}.")
            yield page_of_issues
            
            page_num += 1

            if 'next' in response.links:
                url = response.links['next']['url']
                params = {}  # Clear params as they are in the next URL
            else:
                url = None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching issues: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 403:
                reset_time = int(e.response.headers.get('X-RateLimit-Reset', time.time() + 60))
                wait_time = max(0, reset_time - time.time())
                print(f"Rate limit exceeded. Waiting for {wait_time:.0f} seconds.")
                time.sleep(wait_time + 5)
            else:
                print("Stopping fetch due to an unrecoverable error.")
                break

def serialize_reported_version(reported_version):
    """
    Converts reported_version to a string format suitable for database storage.
    Handles both string and dict formats from LLM.
    """
    if reported_version is None:
        return None
    elif isinstance(reported_version, dict):
        return json.dumps(reported_version)
    elif isinstance(reported_version, str):
        return reported_version
    else:
        # Fallback: convert to string
        return str(reported_version)

def fetch_and_save_issues(db: Session) -> int:
    """
    Fetches issues from GitHub and saves them to the database without any processing.
    Returns the number of new or updated issues saved.
    """
    embedding_service = get_embedding_service()
    total_new_or_updated = 0
    for page_of_issues in stream_github_issue_pages():
        for issue_data in page_of_issues:
            if "pull_request" in issue_data:
                continue

            existing_issue = db.query(Issue).filter(Issue.number == issue_data['number']).first()
            
            updated_at_value = getattr(existing_issue, 'updated_at', None) if existing_issue else None
            if existing_issue and updated_at_value and datetime.fromisoformat(updated_at_value.isoformat()) >= datetime.fromisoformat(issue_data['updated_at'].replace('Z', '')):
                continue

            issue_payload = {
                "number": issue_data['number'], "title": issue_data['title'],
                "body": issue_data['body'] or "", "state": issue_data['state'],
                "created_at": datetime.fromisoformat(issue_data['created_at'].replace('Z', '')),
                "updated_at": datetime.fromisoformat(issue_data['updated_at'].replace('Z', '')),
                "labels": [label['name'] for label in issue_data['labels']],
                "user_login": issue_data['user']['login'],
                 "issue_embedding": embedding_service.create_embedding(
                    f"{issue_data['title']}\n{issue_data['body'] or ''}"
                ),
                # Extract milestone information for fixed_in_version
                "fixed_in_version": issue_data.get('milestone', {}).get('title') if issue_data.get('milestone') else None,
                # Calculate token count for body field
                "token_count": calculate_token_count(issue_data['body'] or ""),
            }
            
            # Debug: Print when milestone data is found
            if issue_payload["fixed_in_version"]:
                print(f"Issue #{issue_data['number']} has milestone: {issue_payload['fixed_in_version']}")
            
            if existing_issue:
                for key, value in issue_payload.items():
                    setattr(existing_issue, key, value)
            else:
                db.add(Issue(**issue_payload))
            
            total_new_or_updated += 1
        db.commit()
    return total_new_or_updated

def process_unprocessed_issues(db: Session) -> int:
    """
    Processes issues that have been fetched but not yet analyzed by the LLM.
    Uses batch processing to efficiently handle multiple issues per LLM call.
    Returns the number of issues processed.
    """
    llm_analyzer = LLMAnalyzer()
    embedding_service = get_embedding_service()

    # First, count how many unprocessed issues we have (lightweight query)
    total_unprocessed = db.query(Issue).filter(Issue.llm_summary.is_(None)).count()
    
    if total_unprocessed == 0:
        print("No unprocessed issues to analyze.")
        return 0

    print(f"Found {total_unprocessed} issues to process with LLM and embeddings.")
    
    # Process issues in batches, but now batch the LLM calls too
    fetch_batch_size = 50  # How many issues to fetch from DB at once
    processed_count = 0
    
    for offset in tqdm(range(0, total_unprocessed, fetch_batch_size), desc="Processing batches"):
        # Get a batch of unprocessed issues from database
        issues_batch = db.query(Issue).filter(
            Issue.llm_summary.is_(None)
        ).offset(offset).limit(fetch_batch_size).all()
        
        if not issues_batch:
            break
            
        print(f"Processing batch {offset//fetch_batch_size + 1}: {len(issues_batch)} issues")
        
        # Create batches for LLM processing based on token count
        llm_batches = create_llm_batches(issues_batch, llm_analyzer)
        
        # Track tokens used in this minute for rate limiting
        minute_start_time = time.time()
        tokens_this_minute = 0
        
        for batch_num, llm_batch in enumerate(llm_batches):
            print(f"  LLM batch {batch_num + 1}/{len(llm_batches)}: {len(llm_batch)} issues")
            
            # Note: Each batch is designed to stay under 1M tokens per API call
            # No need for per-minute token rate limiting - just respect RPM limits
            
            # Prepare data for batch LLM call
            issues_for_llm = []
            for issue in llm_batch:
                issues_for_llm.append((
                    issue.id,           # Use database ID for mapping
                    issue.title,
                    issue.body or "",
                    issue.labels,
                    issue.state
                ))
            
            # Call LLM with batch of issues
            try:
                batch_results = llm_analyzer.analyze_issues_batch(issues_for_llm)
                
                # Update each issue with LLM results and create embeddings
                for issue in llm_batch:
                    if issue.id in batch_results:
                        analysis = batch_results[getattr(issue, "id", 0)]
                        
                        db.execute(text("UPDATE issues SET llm_summary = :summary WHERE id = :issue_id"), {"summary": analysis.get("summary"), "issue_id": issue.id})
                        db.execute(text("UPDATE issues SET reported_version = :version WHERE id = :issue_id"), {"version": serialize_reported_version(analysis.get("reported_version")), "issue_id": issue.id})
                        db.execute(text("UPDATE issues SET stack_trace_file = :file WHERE id = :issue_id"), {"file": analysis.get("stack_trace_file"), "issue_id": issue.id})
                        # Note: fixed_in_version comes from GitHub API milestone data
                        
                        # Create embedding if we have a summary
                        if getattr(issue, "llm_summary", None):
                            embedding = embedding_service.create_embedding(str(issue.llm_summary))
                            if embedding:
                                db.execute(text("UPDATE issues SET summary_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        
                        # Commit immediately after processing each issue
                        try:
                            db.commit()
                            processed_count += 1
                            if processed_count % 10 == 0:
                                print(f"    Processed {processed_count} issues so far...")
                        except Exception as commit_error:
                            print(f"    Error committing issue {getattr(issue, 'number', 'unknown')}: {commit_error}")
                            db.rollback()
                            continue
                    else:
                        print(f"    Warning: No LLM result for issue {issue.number} (ID: {issue.id})")
                        
            except Exception as e:
                print(f"  Error in LLM batch processing: {e}")
                # Fallback to individual processing for this batch
                print("  Falling back to individual processing...")
                for issue in llm_batch:
                    try:
                        analysis = llm_analyzer.analyze_issue(str(issue.title), str(issue.body), getattr(issue, "labels", []) or [], str(issue.state))
                        
                        db.execute(text("UPDATE issues SET llm_summary = :summary WHERE id = :issue_id"), {"summary": analysis.get("summary"), "issue_id": issue.id})
                        db.execute(text("UPDATE issues SET reported_version = :version WHERE id = :issue_id"), {"version": serialize_reported_version(analysis.get("reported_version")), "issue_id": issue.id})
                        db.execute(text("UPDATE issues SET stack_trace_file = :file WHERE id = :issue_id"), {"file": analysis.get("stack_trace_file"), "issue_id": issue.id})
                        
                        if getattr(issue, "llm_summary", None):
                            embedding = embedding_service.create_embedding(str(issue.llm_summary))
                            if embedding:
                                db.execute(text("UPDATE issues SET summary_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
                        
                        try:
                            db.commit()
                            processed_count += 1
                        except Exception as commit_error:
                            print(f"    Error committing issue {getattr(issue, 'number', 'unknown')}: {commit_error}")
                            db.rollback()
                            continue
                            
                    except Exception as individual_error:
                        print(f"    Error processing individual issue {getattr(issue, 'number', 'unknown')}: {individual_error}")
                        # Rollback the session in case of any error to ensure it's in a clean state
                        db.rollback()
                        continue

    return processed_count

def regenerate_issue_embeddings(db: Session) -> int:
    """
    Resets ALL issue embeddings to NULL and regenerates them from scratch.
    This is useful when the embedding model has been changed and you want a complete refresh.
    Uses GPU if available for faster processing.
    """
    from src.utils import get_device
    device = get_device()
    print(f"Using device: {device}")
    
    # First, reset all embeddings to NULL
    total_issues = db.query(Issue).count()
    if total_issues == 0:
        print("No issues found in the database.")
        return 0
    
    print(f"Resetting ALL {total_issues} issue embeddings to NULL...")
    db.query(Issue).update({Issue.issue_embedding: None})
    db.commit()
    print("All embeddings reset to NULL. Starting regeneration...")
    
    # Now generate embeddings for all issues
    embedding_service = get_embedding_service()
    regenerated_count = 0
    failed_count = 0
    
    # Process all issues one by one
    all_issues = db.query(Issue).all()
    
    for issue in tqdm(all_issues, desc="Regenerating All Embeddings"):
        try:
            text_to_embed = f"{issue.title}\n{issue.body or ''}"
            embedding = embedding_service.create_embedding(text_to_embed)
            
            # Validate embedding result
            if embedding is None:
                print(f"Warning: Failed to create embedding for issue #{issue.number} (embedding is None)")
                failed_count += 1
                continue
                
            if not embedding or len(embedding) == 0:
                print(f"Warning: Invalid embedding shape for issue #{issue.number}")
                failed_count += 1
                continue
            
            # Update and commit immediately
            db.execute(text("UPDATE issues SET issue_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
            db.commit()
            regenerated_count += 1
            
        except Exception as e:
            print(f"Error processing issue #{issue.number}: {str(e)}")
            failed_count += 1
            try:
                db.rollback()
            except:
                pass
            continue
    
    print(f"Full regeneration complete: {regenerated_count} successful, {failed_count} failed")
    return regenerated_count

def generate_issue_embeddings(db: Session) -> int:
    """
    Generates embeddings only for issues that currently have null embeddings.
    This is useful for incremental updates when adding new issues or fixing failed embeddings.
    Uses GPU if available for faster processing.
    """
    from src.utils import get_device
    device = get_device()
    print(f"Using device: {device}")
    
    embedding_service = get_embedding_service()
    
    # Only count and process issues with null embeddings
    null_embedding_issues = db.query(Issue).filter(Issue.issue_embedding == None).count()
    
    if null_embedding_issues == 0:
        print("No issues with null embeddings found. All issues already have embeddings.")
        return 0

    print(f"Found {null_embedding_issues} issues with null embeddings to process.")
    
    generated_count = 0
    failed_count = 0
    
    # Process issues one by one - no batching needed for local embedding generation
    issues_with_null_embeddings = db.query(Issue).filter(Issue.issue_embedding == None).all()
    
    for issue in tqdm(issues_with_null_embeddings, desc="Generating Missing Embeddings"):
        try:
            text_to_embed = f"{issue.title}\n{issue.body or ''}"
            embedding = embedding_service.create_embedding(text_to_embed)
            
            # Validate embedding result
            if embedding is None:
                print(f"Warning: Failed to create embedding for issue #{issue.number} (embedding is None)")
                failed_count += 1
                continue
                
            if not embedding or len(embedding) == 0:
                print(f"Warning: Invalid embedding shape for issue #{issue.number}")
                failed_count += 1
                continue
            
            # Update and commit immediately
            db.execute(text("UPDATE issues SET issue_embedding = :embedding WHERE id = :issue_id"), {"embedding": str(embedding), "issue_id": issue.id})
            db.commit()
            generated_count += 1
            
        except Exception as e:
            print(f"Error processing issue #{issue.number}: {str(e)}")
            failed_count += 1
            try:
                db.rollback()
            except:
                pass
            continue
    
    print(f"Generation complete: {generated_count} successful, {failed_count} failed")
    return generated_count

def create_llm_batches(issues: List[Issue], llm_analyzer: LLMAnalyzer) -> List[List[Issue]]:
    """
    Creates simple batches of issues for LLM processing.
    Since batch processing won't exceed 1M tokens, we use a simple fixed-size batching.
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.constants import DEFAULT_BATCH_SIZE
    batch_size = DEFAULT_BATCH_SIZE
    batches = []
    for i in range(0, len(issues), batch_size):
        batches.append(issues[i:i + batch_size])
    return batches

def get_discourse_headers() -> Dict[str, str]:
    """Returns headers for Discourse API requests."""
    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'MetabaseDuplicateFinder/1.0'
    }
    
    if DISCOURSE_API_KEY and DISCOURSE_API_USERNAME:
        headers['Api-Key'] = str(DISCOURSE_API_KEY)
        headers['Api-Username'] = str(DISCOURSE_API_USERNAME)
    
    return headers

def stream_discourse_topic_pages() -> Generator[List[Dict[str, Any]], None, None]:
    """
    Fetches all topics from Discourse page by page.
    Handles pagination and rate limiting properly using Discourse pagination metadata.
    Yields each page of topics as a list of dictionaries.
    """
    url = f"{DISCOURSE_BASE_URL}/latest.json"
    headers = get_discourse_headers()
    params = {"order": "desc"}

    page_num = 0
    topics_per_page = 30  # Discourse default per page
    last_page_topic_count = None
    
    while True:
        try:
            print(f"\nFetching discourse page {page_num + 1}...")
            if page_num > 0:
                params["page"] = str(page_num)
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Get topics and pagination info
            topic_list = data.get('topic_list', {})
            topics = topic_list.get('topics', [])
            
            # Check if we have topics
            if not topics:
                print("No topics found on this page. Concluding fetch.")
                break
            
            print(f"Found {len(topics)} topics on page {page_num + 1}.")
            
            # Check for pagination metadata
            more_topics_url = topic_list.get('more_topics_url')
            per_page = topic_list.get('per_page', topics_per_page)
            
            # Yield current page
            yield topics
            
            # Determine if we should continue
            should_continue = False
            
            # Method 1: Check for more_topics_url (most reliable)
            if more_topics_url:
                print(f"More topics available (more_topics_url present)")
                should_continue = True
            # Method 2: Check if we got a full page
            elif len(topics) >= per_page:
                print(f"Full page received ({len(topics)} topics), likely more pages available")
                should_continue = True
            # Method 3: Check if this page has fewer topics than the last (declining pattern)
            elif last_page_topic_count is not None and len(topics) < last_page_topic_count:
                print(f"Received fewer topics than last page ({len(topics)} vs {last_page_topic_count}), likely last page")
                should_continue = False
            # Method 4: If this is a small page on first fetch, might be the only page
            elif page_num == 0 and len(topics) < per_page:
                print(f"First page with fewer than {per_page} topics, likely the only page")
                should_continue = False
            else:
                # Default to continue if we're unsure (safer to over-fetch than under-fetch)
                print(f"Uncertain about pagination, continuing to be safe")
                should_continue = True
            
            if not should_continue:
                print("Concluding fetch based on pagination analysis.")
                break
            
            # Safety check to prevent infinite loops
            if page_num >= DISCOURSE_MAX_PAGES - 1:
                print(f"Reached maximum page limit ({DISCOURSE_MAX_PAGES}). Stopping to prevent infinite loop.")
                break
            
            last_page_topic_count = len(topics)
            page_num += 1
            
            # Add delay to be respectful to the API
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching discourse topics: {e}")
            if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 403:
                print("Access denied. Check your Discourse API credentials.")
            elif isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                print("Page not found - likely reached end of available pages.")
            break

def get_topic_conversation(topic_id: int, slug: str) -> Optional[Dict[str, Any]]:
    """Fetches the full conversation for a given topic."""
    url = f"{DISCOURSE_BASE_URL}/t/{slug}/{topic_id}.json"
    headers = get_discourse_headers()
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching topic {topic_id}: {e}")
        return None

def fetch_and_save_discourse_posts(db: Session) -> int:
    """
    Fetches discourse posts and saves them to the database with embeddings.
    Returns the number of new or updated posts saved.
    """
    total_new_or_updated = 0
    
    for page_of_topics in stream_discourse_topic_pages():
        for topic_data in page_of_topics:
            topic_id = topic_data['id']
            slug = topic_data['slug']
            
            existing_post = db.query(DiscoursePost).filter(DiscoursePost.topic_id == topic_id).first()
            
            # Skip if we already have this topic
            if existing_post:
                continue

            print(f"Processing topic #{topic_id}: {topic_data.get('fancy_title', topic_data.get('title', ''))}")
            
            # Get the full conversation
            conversation_data = get_topic_conversation(topic_id, slug)
            if not conversation_data:
                print(f"Failed to fetch conversation for topic #{topic_id}")
                continue
            
            # Extract posts and combine them (all posts, not just creator)
            posts = conversation_data.get('post_stream', {}).get('posts', [])
            title = topic_data.get('fancy_title', topic_data.get('title', ''))
            
            # Combine all posts from the topic
            conversation = combine_all_discourse_posts(title, posts)
            
            # Analyze the conversation using LLM
            llm_analyzer = LLMAnalyzer()
            analysis_result = llm_analyzer.analyze_discourse_conversation(conversation)
            
            post_payload = {
                "topic_id": topic_id,
                "title": title,
                "conversation": conversation,
                "created_at": datetime.fromisoformat(topic_data['created_at'].replace('Z', '')),
                "slug": slug,
                "llm_summary": analysis_result.get("llm_summary", ""),
                "type_of_topic": analysis_result.get("type_of_topic"),
                "solution": analysis_result.get("solution"),
                "version": analysis_result.get("version"),
                "reference": analysis_result.get("reference"),
                # Calculate token count for conversation field
                "token_count": calculate_token_count(conversation),
            }
            
            try:
                new_post = DiscoursePost(**post_payload)
                db.add(new_post)
                db.commit()
                total_new_or_updated += 1
                print(f"  Successfully saved discourse post #{topic_id}")
            except Exception as e:
                print(f"  Error saving discourse post #{topic_id}: {e}")
                db.rollback()
                continue
            
            # Add delay to be respectful to the API
            time.sleep(0.5)
        
        # Commit after each page
        db.commit()
    
    return total_new_or_updated

def generate_discourse_embeddings(db: Session) -> int:
    """
    Generates embeddings only for discourse posts that currently have null embeddings.
    """
    from src.utils import get_device
    device = get_device()
    print(f"Using device: {device}")
    
    embedding_service = get_embedding_service()
    
    # Only count and process posts with null embeddings
    null_embedding_posts = db.query(DiscoursePost).filter(DiscoursePost.conversation_embedding == None).count()
    
    if null_embedding_posts == 0:
        print("No discourse posts with null embeddings found. All posts already have embeddings.")
        return 0

    print(f"Found {null_embedding_posts} discourse posts with null embeddings to process.")
    
    generated_count = 0
    failed_count = 0
    
    # Process posts one by one
    posts_with_null_embeddings = db.query(DiscoursePost).filter(DiscoursePost.conversation_embedding == None).all()
    
    for post in tqdm(posts_with_null_embeddings, desc="Generating Missing Discourse Embeddings"):
        try:
            embedding = embedding_service.create_embedding(str(post.conversation))
            
            # Validate embedding result
            if embedding is None:
                print(f"Warning: Failed to create embedding for discourse post #{post.topic_id} (embedding is None)")
                failed_count += 1
                continue
                
            if not embedding or len(embedding) == 0:
                print(f"Warning: Invalid embedding shape for discourse post #{post.topic_id}")
                failed_count += 1
                continue
            
            # Update and commit immediately
            db.execute(text("UPDATE discourse_posts SET conversation_embedding = :embedding WHERE id = :post_id"), {"embedding": str(embedding), "post_id": post.id})
            db.commit()
            generated_count += 1
            
        except Exception as e:
            print(f"Error processing discourse post #{post.topic_id}: {str(e)}")
            failed_count += 1
            try:
                db.rollback()
            except:
                pass
            continue
    
    print(f"Discourse embedding generation complete: {generated_count} successful, {failed_count} failed")
    return generated_count



def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Populate the database with GitHub issues and Discourse posts.")
    parser.add_argument("--fetch-and-embed", action="store_true", help="Fetch issues and create title+body embeddings.")
    parser.add_argument("--summarize-and-embed", action="store_true", help="Use LLM to summarize issues and create summary embeddings.")
    parser.add_argument("--regenerate-embeddings", action="store_true", help="Reset ALL embeddings to NULL and regenerate from scratch.")
    parser.add_argument("--generate-embeddings", action="store_true", help="Generate embeddings only for issues that currently have NULL embeddings.")
    parser.add_argument("--fetch-discourse", action="store_true", help="Fetch discourse posts and create embeddings.")
    parser.add_argument("--generate-discourse-embeddings", action="store_true", help="Generate embeddings only for discourse posts that currently have NULL embeddings.")

    args = parser.parse_args()

    # Force SQLAlchemy metadata refresh to handle schema changes
    refresh_sqlalchemy_metadata()

    # Always initialize the database
    db = SessionLocal()

    if args.fetch_and_embed:
        print("--- Starting: Fetch and Embed Mode ---")
        count = fetch_and_save_issues(db)
        print(f"Finished fetching and embedding. {count} new or updated issues were saved.")
    elif args.summarize_and_embed:
        print("--- Starting: Summarize and Embed Mode ---")
        count = process_unprocessed_issues(db)
        print(f"Finished summarizing and embedding. {count} issues were updated.")
    elif args.regenerate_embeddings:
        print("--- Starting: Regenerate Embeddings Mode (Full Reset) ---")
        count = regenerate_issue_embeddings(db)
        print(f"Finished regenerating embeddings. {count} issues were updated.")
    elif args.generate_embeddings:
        print("--- Starting: Generate Embeddings Mode (Incremental) ---")
        count = generate_issue_embeddings(db)
        print(f"Finished generating embeddings. {count} issues were updated.")
    elif args.fetch_discourse:
        print("--- Starting: Fetch Discourse Posts Mode ---")
        count = fetch_and_save_discourse_posts(db)
        print(f"Finished fetching discourse posts. {count} new posts were saved.")
    elif args.generate_discourse_embeddings:
        print("--- Starting: Generate Discourse Embeddings Mode ---")
        count = generate_discourse_embeddings(db)
        print(f"Finished generating discourse embeddings. {count} posts were updated.")
    else:
        print("--- Starting: Full Sync (Fetch, Summarize, and Embed) ---")
        fetch_count = fetch_and_save_issues(db)
        print(f"Finished fetching and embedding. {fetch_count} new or updated issues were saved.")
        process_count = process_unprocessed_issues(db)
        print(f"Finished summarizing and embedding. {process_count} issues were updated.")

    db.close()

if __name__ == "__main__":
    main()
