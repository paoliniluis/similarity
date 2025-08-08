import html
import re
from typing import List, Dict, Any, Optional

def sanitize_sql_content(text: str) -> str:
    """
    Sanitizes text content to remove potential SQL injection patterns.
    
    Args:
        text: Text that might contain SQL-like content
        
    Returns:
        Sanitized text safe for database storage and processing
    """
    if not text:
        return ""
    
    # Common SQL keywords and patterns to neutralize (case-insensitive)
    sql_patterns = [
        # SQL DML statements
        r'\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE)\b',
        # SQL injection patterns
        r'\b(UNION|OR\s+1\s*=\s*1|AND\s+1\s*=\s*1)\b',
        r'(--|/\*|\*/)',  # SQL comments
        r'(\';|\';\s*--)',  # Common injection endings (fixed parenthesis)
        # Database functions that could be dangerous
        r'\b(EXEC|EXECUTE|sp_|xp_)\b',
        # SQL operators in dangerous contexts
        r'(\'\s*(OR|AND)\s*\'\w*\'\s*=\s*\'\w*)',
    ]
    
    sanitized_text = text
    for pattern in sql_patterns:
        # Replace SQL patterns with neutral text, preserving readability
        sanitized_text = re.sub(pattern, '[SQL_CONTENT_REMOVED]', sanitized_text, flags=re.IGNORECASE)
    
    # Remove potential hex/binary injection patterns
    sanitized_text = re.sub(r'0x[0-9a-fA-F]+', '[HEX_REMOVED]', sanitized_text)
    
    return sanitized_text

def get_topic_creator_username(topic_data: Dict[str, Any], posts: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extracts the topic creator's username from Discourse topic and post data.
    
    Args:
        topic_data: Topic data from /latest.json
        posts: Posts data from /t/<slug>/<id>.json
        
    Returns:
        Creator's username or None if not found
    """
    creator_username = None
    
    # Method 1: Get from first post (most reliable)
    if posts:
        creator_username = posts[0].get('username')
    
    # Method 2: If not found, try to get from topic data (posters info)
    if not creator_username:
        posters = topic_data.get('posters', [])
        if posters:
            # Find the original poster (usually marked with 'Original Poster' or first in list)
            for poster in posters:
                if poster.get('description') == 'Original Poster':
                    creator_username = poster.get('user', {}).get('username')
                    break
            # Fallback: use first poster if no 'Original Poster' found
            if not creator_username and posters:
                creator_username = posters[0].get('user', {}).get('username')
    
    return creator_username

def decode_discourse_text(encoded_text: str) -> str:
    """
    Decodes unicode escapes and removes HTML tags from Discourse post text,
    while preserving links for potential use with LLMs.
    
    Args:
        encoded_text: The encoded text from Discourse API (cooked parameter)
        
    Returns:
        Clean text suitable for embedding generation with preserved links
    """
    if not encoded_text:
        return ""
    
    # First decode unicode escapes (like \u003Cp\u003E becomes <p>)
    try:
        decoded = encoded_text.encode().decode('unicode_escape')
    except UnicodeDecodeError:
        decoded = encoded_text
    
    # Preserve links by extracting them before removing HTML tags
    # Find all <a> tags and store their href and text content
    link_pattern = r'<a\s+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
    links = re.findall(link_pattern, decoded)
    
    # Replace links with a placeholder that includes the URL
    def replace_link(match):
        href = match.group(1)
        link_text = match.group(2)
        return f"[LINK: {link_text} -> {href}]"
    
    # Replace all links with our format
    text = re.sub(link_pattern, replace_link, decoded)
    
    # Remove remaining HTML tags using regex
    # This handles the common HTML tags found in Discourse posts
    text = re.sub(r'<[^>]+>', '', text)
    
    # Decode HTML entities (like &amp; &lt; &gt; etc.)
    text = html.unescape(text)
    
    # Remove NUL bytes (0x00) that PostgreSQL cannot handle
    text = text.replace('\x00', '')
    
    # Remove other problematic control characters (0x01-0x08, 0x0B, 0x0C, 0x0E-0x1F)
    text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F]', '', text)
    
    # Sanitize potential SQL injection content
    text = sanitize_sql_content(text)
    
    # Clean up extra whitespace and newlines
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
    text = re.sub(r' +', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    return text

def combine_discourse_posts(title: str, posts: list, creator_username: Optional[str] = None) -> str:
    """
    Combines the title and posts from the topic creator in a Discourse topic into a single text.
    
    Args:
        title: The topic title
        posts: List of post dictionaries from Discourse API
        creator_username: Username of the topic creator (if None, includes all posts)
        
    Returns:
        Combined text suitable for embedding (only creator's posts)
    """
    if not posts:
        return title
    
    # Start with the title
    full_text = f"Title: {title}\n\n"
    
    # Filter and add only posts from the creator
    creator_post_count = 0
    for post in posts:
        # Check if this post is from the topic creator
        post_username = post.get('username', '')
        
        # If creator_username is specified, only include posts from the creator
        if creator_username and post_username != creator_username:
            continue
            
        cooked = post.get('cooked', '')
        if cooked:
            post_text = decode_discourse_text(cooked)
            if post_text.strip():
                creator_post_count += 1
                full_text += f"Creator Post {creator_post_count}:\n{post_text}\n\n"
    
    # If no creator posts found, just return the title
    if creator_post_count == 0 and creator_username:
        return f"Title: {title}\n\n(No posts from topic creator found)"
    
    return full_text.strip()

def combine_all_discourse_posts(title: str, posts: list) -> str:
    """
    Combines the title and all posts from a Discourse topic into a single text.
    
    Args:
        title: The topic title
        posts: List of post dictionaries from Discourse API
        
    Returns:
        Combined text suitable for embedding (all posts in chronological order)
    """
    if not posts:
        return title
    
    # Start with the title
    full_text = f"Title: {title}\n\n"
    
    # Add all posts in chronological order
    for i, post in enumerate(posts, 1):
        cooked = post.get('cooked', '')
        if cooked:
            post_text = decode_discourse_text(cooked)
            if post_text.strip():
                username = post.get('username', 'Unknown')
                full_text += f"Post {i} (by {username}):\n{post_text}\n\n"
    
    return full_text.strip()

def calculate_token_count(text: str) -> int:
    """
    Calculate the approximate token count for given text.
    Uses a simple heuristic of 2 tokens per word.
    
    Args:
        text: The text to count tokens for
        
    Returns:
        Estimated token count (words * 2)
    """
    if not text or not text.strip():
        return 0
    
    # Basic word count: split on whitespace and count non-empty strings
    words = [word for word in text.split() if word.strip()]
    
    # Apply the 2 tokens per word heuristic
    return len(words) * 2