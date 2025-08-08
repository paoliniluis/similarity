import argparse
import secrets
from sqlalchemy import text, or_
from src.db import engine, Base, SessionLocal
from src.models import ApiKey, ChatSession, ChatSessionEntity, DiscoursePost, Issue, MetabaseDoc, Question, SourceType, KeywordDefinition, Synonym, BatchProcess
from src.text_utils import calculate_token_count

def enable_vector_extension():
    """Enables the pgvector extension in the database."""
    print("Enabling the 'vector' extension...")
    with engine.connect() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        connection.commit()
    print("Vector extension enabled successfully.")

def recreate_database():
    """Drops all tables and recreates them based on the current models."""
    print("Dropping all tables...")
    Base.metadata.drop_all(bind=engine)
    print("Recreating all tables...")
    Base.metadata.create_all(bind=engine)
    print("Database has been reset successfully.")

def recreate_issues_table():
    """Drops and recreates only the issues table."""
    print("Dropping issues table...")
    Issue.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating issues table...")
    Issue.__table__.create(bind=engine, checkfirst=True)
    print("Issues table has been recreated successfully.")

def recreate_discourse_table():
    """Drops and recreates only the discourse_posts table."""
    print("Dropping discourse_posts table...")
    DiscoursePost.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating discourse_posts table...")
    DiscoursePost.__table__.create(bind=engine, checkfirst=True)
    print("Discourse posts table has been recreated successfully.")

def add_api_key(description: str):
    """Generates a new API key and adds it to the database."""
    db = SessionLocal()
    new_key = secrets.token_urlsafe(32)
    api_key_record = ApiKey(key=new_key, description=description)
    db.add(api_key_record)
    db.commit()
    print(f"New API Key created successfully!")
    print(f"  Description: {description}")
    print(f"  Key: {new_key}")
    db.close()

def clear_discourse_posts():
    """Clears all discourse posts from the database."""
    db = SessionLocal()
    count = db.query(DiscoursePost).count()
    if count == 0:
        print("No discourse posts found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} discourse posts? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(DiscoursePost).delete()
    db.commit()
    print(f"Successfully deleted {count} discourse posts.")
    db.close()

def show_discourse_stats():
    """Shows statistics about discourse posts in the database."""
    db = SessionLocal()
    total_posts = db.query(DiscoursePost).count()
    posts_with_embeddings = db.query(DiscoursePost).filter(DiscoursePost.conversation_embedding.isnot(None)).count()
    posts_with_summaries = db.query(DiscoursePost).filter(DiscoursePost.llm_summary.isnot(None)).count()
    posts_with_summary_embeddings = db.query(DiscoursePost).filter(DiscoursePost.summary_embedding.isnot(None)).count()
    posts_without_embeddings = total_posts - posts_with_embeddings
    
    print(f"Discourse Posts Statistics:")
    print(f"  Total posts: {total_posts}")
    print(f"  Posts with conversation embeddings: {posts_with_embeddings}")
    print(f"  Posts with LLM summaries: {posts_with_summaries}")
    print(f"  Posts with summary embeddings: {posts_with_summary_embeddings}")
    print(f"  Posts without embeddings: {posts_without_embeddings}")
    
    if total_posts > 0:
        latest_post = db.query(DiscoursePost).order_by(DiscoursePost.created_at.desc()).first()
        oldest_post = db.query(DiscoursePost).order_by(DiscoursePost.created_at.asc()).first()
        print(f"  Date range: {oldest_post.created_at.date() if oldest_post and getattr(oldest_post, "created_at", None) else "N/A"} to {latest_post.created_at.date() if latest_post and getattr(latest_post, "created_at", None) else "N/A"}")
    
    db.close()

def recreate_metabase_docs_table():
    """Drops and recreates only the metabase_docs table."""
    print("Dropping metabase_docs table...")
    MetabaseDoc.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating metabase_docs table...")
    MetabaseDoc.__table__.create(bind=engine, checkfirst=True)
    print("Metabase docs table has been recreated successfully.")

def clear_metabase_docs():
    """Clears all Metabase documentation from the database."""
    db = SessionLocal()
    count = db.query(MetabaseDoc).count()
    if count == 0:
        print("No Metabase documentation found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} Metabase documentation entries? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(MetabaseDoc).delete()
    db.commit()
    print(f"Successfully deleted {count} Metabase documentation entries.")
    db.close()

def show_metabase_docs_stats():
    """Shows statistics about Metabase documentation in the database."""
    db = SessionLocal()
    total_docs = db.query(MetabaseDoc).count()
    docs_with_embeddings = db.query(MetabaseDoc).filter(
        or_(MetabaseDoc.markdown_embedding.isnot(None), MetabaseDoc.summary_embedding.isnot(None))
    ).count()
    docs_without_embeddings = total_docs - docs_with_embeddings
    
    print(f"Metabase Documentation Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  Documents with embeddings: {docs_with_embeddings}")
    print(f"  Documents without embeddings: {docs_without_embeddings}")
    
    if total_docs > 0:
        latest_doc = db.query(MetabaseDoc).order_by(MetabaseDoc.created_at.desc()).first()
        oldest_doc = db.query(MetabaseDoc).order_by(MetabaseDoc.created_at.asc()).first()
        print(f"  Date range: {oldest_doc.created_at.date() if oldest_doc and getattr(oldest_doc, "created_at", None) else "N/A"} to {latest_doc.created_at.date() if latest_doc and getattr(latest_doc, "created_at", None) else "N/A"}")
        
        # Show some example URLs
        docs = db.query(MetabaseDoc).limit(5).all()
        print(f"  Sample URLs:")
        for doc in docs:
            print(f"    - {doc.url}")
    
    db.close()

def recreate_questions_table():
    """Drops and recreates only the questions table."""
    print("Dropping questions table...")
    Question.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating questions table...")
    Question.__table__.create(bind=engine, checkfirst=True)
    print("Questions table has been recreated successfully.")

def clear_questions():
    """Clears all questions from the database."""
    db = SessionLocal()
    count = db.query(Question).count()
    if count == 0:
        print("No questions found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} questions? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(Question).delete()
    db.commit()
    print(f"Successfully deleted {count} questions.")
    db.close()

def show_questions_stats():
    """Shows statistics about questions in the database."""
    db = SessionLocal()
    total_questions = db.query(Question).count()
    questions_with_question_embeddings = db.query(Question).filter(Question.question_embedding.isnot(None)).count()
    questions_with_answer_embeddings = db.query(Question).filter(Question.answer_embedding.isnot(None)).count()
    questions_without_embeddings = total_questions - questions_with_question_embeddings
    
    # Count questions by source type
    metabase_questions = db.query(Question).filter(Question.source_type == SourceType.METABASE_DOC).count()
    issue_questions = db.query(Question).filter(Question.source_type == SourceType.ISSUE).count()
    discourse_questions = db.query(Question).filter(Question.source_type == SourceType.DISCOURSE_POST).count()
    
    print(f"Questions Statistics:")
    print(f"  Total questions: {total_questions}")
    print(f"  Questions with question embeddings: {questions_with_question_embeddings}")
    print(f"  Questions with answer embeddings: {questions_with_answer_embeddings}")
    print(f"  Questions without embeddings: {questions_without_embeddings}")
    print(f"  Questions by source type:")
    print(f"    - Metabase docs: {metabase_questions}")
    print(f"    - Issues: {issue_questions}")
    print(f"    - Discourse posts: {discourse_questions}")
    
    if total_questions > 0:
        latest_question = db.query(Question).order_by(Question.created_at.desc()).first()
        oldest_question = db.query(Question).order_by(Question.created_at.asc()).first()
        print(f"  Date range: {oldest_question.created_at.date() if oldest_question and getattr(oldest_question, "created_at", None) else "N/A"} to {latest_question.created_at.date() if latest_question and getattr(latest_question, "created_at", None) else "N/A"}")
        
        # Show some example questions by source type
        print(f"  Sample Questions by Source Type:")
        
        metabase_samples = db.query(Question).filter(Question.source_type == SourceType.METABASE_DOC).limit(2).all()
        if metabase_samples:
            print(f"    Metabase Docs:")
            for q in metabase_samples:
                print(f"      - Q: {q.question[:50]}...")
                print(f"        A: {q.answer[:50]}...")
        
        issue_samples = db.query(Question).filter(Question.source_type == SourceType.ISSUE).limit(2).all()
        if issue_samples:
            print(f"    Issues:")
            for q in issue_samples:
                print(f"      - Q: {q.question[:50]}...")
                print(f"        A: {q.answer[:50]}...")
        
        discourse_samples = db.query(Question).filter(Question.source_type == SourceType.DISCOURSE_POST).limit(2).all()
        if discourse_samples:
            print(f"    Discourse Posts:")
            for q in discourse_samples:
                print(f"      - Q: {q.question[:50]}...")
                print(f"        A: {q.answer[:50]}...")
    
    db.close()

def recreate_chat_sessions_table():
    """Drops and recreates only the chat_sessions table (and dependent chat_session_entities)."""
    print("Dropping chat_session_entities table first (due to foreign key dependency)...")
    ChatSessionEntity.__table__.drop(bind=engine, checkfirst=True)
    print("Dropping chat_sessions table...")
    ChatSession.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating chat_sessions table...")
    ChatSession.__table__.create(bind=engine, checkfirst=True)
    print("Recreating chat_session_entities table...")
    ChatSessionEntity.__table__.create(bind=engine, checkfirst=True)
    print("Chat sessions and entities tables have been recreated successfully.")

def clear_chat_sessions():
    """Clears all chat sessions from the database."""
    db = SessionLocal()
    count = db.query(ChatSession).count()
    if count == 0:
        print("No chat sessions found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} chat sessions? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(ChatSession).delete()
    db.commit()
    print(f"Successfully deleted {count} chat sessions.")
    db.close()

def show_chat_sessions_stats():
    """Shows statistics about chat sessions in the database."""
    db = SessionLocal()
    total_sessions = db.query(ChatSession).count()
    sessions_with_sources = db.query(ChatSession).filter(ChatSession.sources.isnot(None)).count()
    sessions_with_response = db.query(ChatSession).filter(ChatSession.response.isnot(None)).count()
    sessions_with_prompt = db.query(ChatSession).filter(ChatSession.prompt.isnot(None)).count()
    sessions_with_errors = db.query(ChatSession).filter(ChatSession.response.like("Error: %")).count()
    
    # Get entity statistics
    total_entities = db.query(ChatSessionEntity).count()
    entity_types = db.query(ChatSessionEntity.entity_type).distinct().all()
    entity_type_list = [et[0] for et in entity_types]
    
    print(f"Chat Sessions Statistics:")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Sessions with sources: {sessions_with_sources}")
    print(f"  Sessions with response: {sessions_with_response}")
    print(f"  Sessions with prompt: {sessions_with_prompt}")
    print(f"  Sessions with errors: {sessions_with_errors}")
    print(f"  Total entities injected: {total_entities}")
    print(f"  Entity types: {', '.join(entity_type_list) if entity_type_list else 'None'}")
    
    if total_sessions > 0:
        latest_session = db.query(ChatSession).order_by(ChatSession.created_at.desc()).first()
        oldest_session = db.query(ChatSession).order_by(ChatSession.created_at.asc()).first()
        print(f"  Date range: {oldest_session.created_at.date() if oldest_session and getattr(oldest_session, "created_at", None) else "N/A"} to {latest_session.created_at.date() if latest_session and getattr(latest_session, "created_at", None) else "N/A"}")
        
        # Show some example sessions
        print(f"  Sample Sessions:")
        recent_sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).limit(3).all()
        for session in recent_sessions:
            print(f"    - Chat ID: {session.chat_id}")
            print(f"      Request: {session.user_request[:50]}...")
            print(f"      Sources: {len(getattr(session, "sources", [])) if getattr(session, "sources", None) else 0} sources")
            print(f"      Response: {str(getattr(session, "response", ""))[:50] if getattr(session, "response", None) else 'None'}...")
            print(f"      Prompt: {str(getattr(session, "prompt", ""))[:100] if getattr(session, "prompt", None) else 'None'}...")
            print(f"      Created: {session.created_at}")
            
            # Show entities for this session
            session_entities = db.query(ChatSessionEntity).filter(ChatSessionEntity.chat_id == session.id).all()
            if session_entities:
                print(f"      Entities injected: {len(session_entities)}")
                for entity in session_entities[:3]:  # Show first 3 entities
                    print(f"        - {entity.entity_type}: {entity.entity_id} (score: {entity.similarity_score})")
            else:
                print(f"      Entities injected: 0")
            print()
    
    db.close()

def recreate_chat_session_entities_table():
    """Drops and recreates only the chat_session_entities table."""
    print("Dropping chat_session_entities table...")
    ChatSessionEntity.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating chat_session_entities table...")
    ChatSessionEntity.__table__.create(bind=engine, checkfirst=True)
    print("Chat session entities table has been recreated successfully.")

def clear_chat_session_entities():
    """Clears all chat session entities from the database."""
    db = SessionLocal()
    count = db.query(ChatSessionEntity).count()
    if count == 0:
        print("No chat session entities found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} chat session entities? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(ChatSessionEntity).delete()
    db.commit()
    print(f"Successfully deleted {count} chat session entities.")
    db.close()

def show_chat_session_entities_stats():
    """Shows statistics about chat session entities in the database."""
    db = SessionLocal()
    total_entities = db.query(ChatSessionEntity).count()
    
    # Get entity type breakdown
    entity_types = db.query(ChatSessionEntity.entity_type).distinct().all()
    entity_type_list = [et[0] for et in entity_types]
    
    print(f"Chat Session Entities Statistics:")
    print(f"  Total entities: {total_entities}")
    print(f"  Entity types: {', '.join(entity_type_list) if entity_type_list else 'None'}")
    
    if total_entities > 0:
        # Show breakdown by entity type
        for entity_type in entity_type_list:
            count = db.query(ChatSessionEntity).filter(ChatSessionEntity.entity_type == entity_type).count()
            print(f"    {entity_type}: {count}")
        
        # Show some example entities
        print(f"\nSample entities:")
        sample_entities = db.query(ChatSessionEntity).limit(5).all()
        for entity in sample_entities:
            print(f"  - {entity.entity_type}: {entity.entity_id} (score: {entity.similarity_score})")
            print(f"    URL: {entity.entity_url or 'N/A'}")
            print(f"    Created: {entity.created_at}")
    
    db.close()

def recreate_keyword_definitions_table():
    """Drops and recreates only the keyword_definitions table."""
    print("Dropping keyword_definitions table...")
    KeywordDefinition.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating keyword_definitions table...")
    KeywordDefinition.__table__.create(bind=engine, checkfirst=True)
    print("Keyword definitions table has been recreated successfully.")

def clear_keyword_definitions():
    """Clears all keyword definitions from the database."""
    db = SessionLocal()
    count = db.query(KeywordDefinition).count()
    if count == 0:
        print("No keyword definitions found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} keyword definitions? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(KeywordDefinition).delete()
    db.commit()
    print(f"Successfully deleted {count} keyword definitions.")
    db.close()

def show_keyword_definitions_stats():
    """Shows statistics about keyword definitions in the database."""
    db = SessionLocal()
    total_keywords = db.query(KeywordDefinition).count()
    active_keywords = db.query(KeywordDefinition).filter(KeywordDefinition.is_active.is_(True)).count()
    inactive_keywords = total_keywords - active_keywords
    keywords_with_embeddings = db.query(KeywordDefinition).filter(KeywordDefinition.keyword_embedding.isnot(None)).count()
    keywords_without_embeddings = total_keywords - keywords_with_embeddings
    
    print(f"Keyword Definitions Statistics:")
    print(f"  Total keywords: {total_keywords}")
    print(f"  Active keywords: {active_keywords}")
    print(f"  Inactive keywords: {inactive_keywords}")
    print(f"  Keywords with embeddings: {keywords_with_embeddings}")
    print(f"  Keywords without embeddings: {keywords_without_embeddings}")
    
    if total_keywords > 0:
        # Show categories
        categories = db.query(KeywordDefinition.category).distinct().all()
        category_list = [cat[0] for cat in categories if cat[0] is not None]
        if category_list:
            print(f"  Categories: {', '.join(category_list)}")
        
        # Show some examples
        print("\nSample keywords:")
        sample_keywords = db.query(KeywordDefinition).limit(5).all()
        for kw in sample_keywords:
            status = "✓" if bool(getattr(kw, "is_active", False)) else "✗"
            embedding_status = "✓" if kw.keyword_embedding is not None else "✗"
            category = f" [{getattr(kw, 'category', None)}]" if getattr(kw, 'category', None) else ""
            print(f"  {status} {embedding_status} {kw.keyword}{category}: {kw.definition[:50]}...")

def migrate_keyword_is_active_to_boolean():
    """Convert existing keyword_definitions.is_active string values to boolean."""
    db = SessionLocal()
    try:
        db.execute(text("UPDATE keyword_definitions SET is_active = TRUE WHERE is_active IN ('true', 'True', '1', 't')"))
        db.execute(text("UPDATE keyword_definitions SET is_active = FALSE WHERE is_active IN ('false', 'False', '0', 'f')"))
        db.commit()
        print("Migrated keyword_definitions.is_active to boolean values.")
    except Exception as e:
        db.rollback()
        print(f"Migration failed: {e}")
    finally:
        db.close()
    
    db.close()

def recreate_synonyms_table():
    """Drops and recreates only the synonyms table."""
    print("Dropping synonyms table...")
    Synonym.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating synonyms table...")
    Synonym.__table__.create(bind=engine, checkfirst=True)
    print("Synonyms table has been recreated successfully.")

def clear_synonyms():
    """Clears all synonyms from the database."""
    db = SessionLocal()
    count = db.query(Synonym).count()
    if count == 0:
        print("No synonyms found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} synonyms? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(Synonym).delete()
    db.commit()
    print(f"Successfully deleted {count} synonyms.")
    db.close()

def show_synonyms_stats():
    """Shows statistics about synonyms in the database."""
    db = SessionLocal()
    total_synonyms = db.query(Synonym).count()
    
    print(f"Synonyms Statistics:")
    print(f"  Total synonyms: {total_synonyms}")
    
    if total_synonyms > 0:
        # Show unique words and their synonyms
        unique_words = db.query(Synonym.word).distinct().count()
        unique_synonym_of = db.query(Synonym.synonym_of).distinct().count()
        print(f"  Unique words: {unique_words}")
        print(f"  Unique synonym groups: {unique_synonym_of}")
        
        # Show some examples
        print("\nSample synonyms:")
        sample_synonyms = db.query(Synonym).limit(10).all()
        for syn in sample_synonyms:
            print(f"  {syn.word} -> {syn.synonym_of}")
    
    db.close()

def recreate_batch_processes_table():
    """Drops and recreates only the batch_processes table."""
    print("Dropping batch_processes table...")
    BatchProcess.__table__.drop(bind=engine, checkfirst=True)
    print("Recreating batch_processes table...")
    BatchProcess.__table__.create(bind=engine, checkfirst=True)
    print("Batch processes table has been recreated successfully.")

def clear_batch_processes():
    """Clears all batch processes from the database."""
    db = SessionLocal()
    count = db.query(BatchProcess).count()
    if count == 0:
        print("No batch processes found in database.")
        db.close()
        return
    
    confirmation = input(f"Are you sure you want to delete all {count} batch processes? (y/N): ")
    if confirmation.lower() != 'y':
        print("Operation cancelled.")
        db.close()
        return
    
    db.query(BatchProcess).delete()
    db.commit()
    print(f"Successfully deleted {count} batch processes.")
    db.close()

def show_batch_processes_stats():
    """Shows statistics about batch processes in the database."""
    db = SessionLocal()
    total_processes = db.query(BatchProcess).count()
    
    print(f"Batch Processes Statistics:")
    print(f"  Total batch processes: {total_processes}")
    
    if total_processes > 0:
        # Show status breakdown
        status_counts = {}
        operation_counts = {}
        table_counts = {}
        
        for process in db.query(BatchProcess).all():
            status = process.status
            operation = process.operation_type
            table = process.table_name
            
            status_counts[status] = status_counts.get(status, 0) + 1
            operation_counts[operation] = operation_counts.get(operation, 0) + 1
            table_counts[table] = table_counts.get(table, 0) + 1
        
        print(f"\nStatus breakdown:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count}")
        
        print(f"\nOperation type breakdown:")
        for operation, count in sorted(operation_counts.items()):
            print(f"  {operation}: {count}")
        
        print(f"\nTable breakdown:")
        for table, count in sorted(table_counts.items()):
            print(f"  {table}: {count}")
        
        # Show recent processes
        print(f"\nRecent batch processes:")
        recent_processes = db.query(BatchProcess).order_by(BatchProcess.created_at.desc()).limit(5).all()
        for process in recent_processes:
            print(f"  {process.batch_id} - {process.operation_type} on {process.table_name} ({process.status})")
    
    db.close()

def add_sample_keywords():
    """Adds some sample keyword definitions for testing."""
    from src.keyword_service import KeywordService
    
    service = KeywordService()
    
    sample_keywords = [
        ("Metabase", "An open-source business intelligence and analytics platform that allows users to explore and visualize data from various databases and data sources.", "Product"),
        ("Dashboard", "A visual display of key metrics and data visualizations that provides an overview of business performance and insights.", "Product"),
        ("Query Builder", "A visual interface in Metabase that allows users to build database queries without writing SQL code.", "Product"),
        ("Pulse", "A feature in Metabase that sends automated reports and insights via email on a scheduled basis.", "Product"),
        ("Alerts", "Notifications that trigger when certain conditions are met in your data, helping users stay informed about important changes.", "Product"),
        ("Data Source", "Any database, API, or file that contains data that can be connected to and analyzed in Metabase.", "Technical"),
        ("SQL", "Structured Query Language, a standard language for accessing and manipulating databases.", "Technical"),
        ("ETL", "Extract, Transform, Load - the process of extracting data from various sources, transforming it, and loading it into a target system.", "Technical"),
        ("API", "Application Programming Interface - a set of rules that allows different software applications to communicate with each other.", "Technical"),
        ("BI", "Business Intelligence - the process of analyzing data to help organizations make better business decisions.", "Technical")
    ]
    
    added_count = 0
    for keyword, definition, category in sample_keywords:
        if service.add_keyword_definition(keyword, definition, category):
            added_count += 1
            print(f"Added: {keyword}")
        else:
            print(f"Skipped (already exists): {keyword}")
    
    print(f"\nAdded {added_count} sample keyword definitions.")



def main():
    """Main entry point for the database management script."""
    parser = argparse.ArgumentParser(description="Manage the application database.")
    parser.add_argument("--recreate", action="store_true", help="Drop all tables and recreate them.")
    parser.add_argument("--recreate-issues", action="store_true", help="Drop and recreate only the issues table.")
    parser.add_argument("--recreate-discourse", action="store_true", help="Drop and recreate only the discourse_posts table.")
    parser.add_argument("--add-api-key", type=str, metavar="DESCRIPTION", help="Add a new API key with the given description.")
    parser.add_argument("--enable-vector", action="store_true", help="Enable the pgvector extension.")
    parser.add_argument("--clear-discourse", action="store_true", help="Clear all discourse posts from the database.")
    parser.add_argument("--discourse-stats", action="store_true", help="Show discourse posts statistics.")
    parser.add_argument("--recreate-metabase-docs", action="store_true", help="Drop and recreate only the metabase_docs table.")
    parser.add_argument("--clear-metabase-docs", action="store_true", help="Clear all Metabase documentation from the database.")
    parser.add_argument("--metabase-docs-stats", action="store_true", help="Show Metabase documentation statistics.")
    parser.add_argument("--recreate-questions", action="store_true", help="Drop and recreate only the questions table.")
    parser.add_argument("--clear-questions", action="store_true", help="Clear all questions from the database.")
    parser.add_argument("--questions-stats", action="store_true", help="Show questions statistics.")
    parser.add_argument("--recreate-chat-sessions", action="store_true", help="Drop and recreate only the chat_sessions table.")
    parser.add_argument("--clear-chat-sessions", action="store_true", help="Clear all chat sessions from the database.")
    parser.add_argument("--chat-sessions-stats", action="store_true", help="Show chat sessions statistics.")
    parser.add_argument("--recreate-chat-session-entities", action="store_true", help="Drop and recreate only the chat_session_entities table.")
    parser.add_argument("--clear-chat-session-entities", action="store_true", help="Clear all chat session entities from the database.")
    parser.add_argument("--chat-session-entities-stats", action="store_true", help="Show chat session entities statistics.")
    parser.add_argument("--recreate-keyword-definitions", action="store_true", help="Drop and recreate only the keyword_definitions table.")
    parser.add_argument("--clear-keyword-definitions", action="store_true", help="Clear all keyword definitions from the database.")
    parser.add_argument("--keyword-definitions-stats", action="store_true", help="Show keyword definitions statistics.")
    parser.add_argument("--add-sample-keywords", action="store_true", help="Add some sample keyword definitions for testing.")
    parser.add_argument("--recreate-synonyms", action="store_true", help="Drop and recreate only the synonyms table.")
    parser.add_argument("--clear-synonyms", action="store_true", help="Clear all synonyms from the database.")
    parser.add_argument("--synonyms-stats", action="store_true", help="Show synonyms statistics.")
    parser.add_argument("--recreate-batch-processes", action="store_true", help="Drop and recreate only the batch_processes table.")
    parser.add_argument("--clear-batch-processes", action="store_true", help="Clear all batch processes from the database.")
    parser.add_argument("--batch-processes-stats", action="store_true", help="Show batch processes statistics.")
    parser.add_argument("--migrate-keyword-boolean", action="store_true", help="Migrate keyword_definitions.is_active strings to boolean.")

    args = parser.parse_args()

    if args.recreate:
        recreate_database()
    elif args.recreate_issues:
        recreate_issues_table()
    elif args.recreate_discourse:
        recreate_discourse_table()
    elif args.recreate_metabase_docs:
        recreate_metabase_docs_table()
    elif args.add_api_key:
        add_api_key(args.add_api_key)
    elif args.enable_vector:
        enable_vector_extension()
    elif args.clear_discourse:
        clear_discourse_posts()
    elif args.discourse_stats:
        show_discourse_stats()
    elif args.clear_metabase_docs:
        clear_metabase_docs()
    elif args.metabase_docs_stats:
        show_metabase_docs_stats()
    elif args.recreate_questions:
        recreate_questions_table()
    elif args.clear_questions:
        clear_questions()
    elif args.questions_stats:
        show_questions_stats()
    elif args.recreate_chat_sessions:
        recreate_chat_sessions_table()
    elif args.clear_chat_sessions:
        clear_chat_sessions()
    elif args.chat_sessions_stats:
        show_chat_sessions_stats()
    elif args.recreate_chat_session_entities:
        recreate_chat_session_entities_table()
    elif args.clear_chat_session_entities:
        clear_chat_session_entities()
    elif args.chat_session_entities_stats:
        show_chat_session_entities_stats()
    elif args.recreate_keyword_definitions:
        recreate_keyword_definitions_table()
    elif args.clear_keyword_definitions:
        clear_keyword_definitions()
    elif args.keyword_definitions_stats:
        show_keyword_definitions_stats()
    elif args.add_sample_keywords:
        add_sample_keywords()
    elif args.recreate_synonyms:
        recreate_synonyms_table()
    elif args.clear_synonyms:
        clear_synonyms()
    elif args.synonyms_stats:
        show_synonyms_stats()
    elif args.recreate_batch_processes:
        recreate_batch_processes_table()
    elif args.clear_batch_processes:
        clear_batch_processes()
    elif args.batch_processes_stats:
        show_batch_processes_stats()
    elif args.migrate_keyword_boolean:
        migrate_keyword_is_active_to_boolean()
    else:
        print("No action specified. Use --recreate, --recreate-issues, --recreate-discourse, --recreate-metabase-docs, --recreate-questions, --recreate-chat-sessions, --recreate-chat-session-entities, --recreate-keyword-definitions, --recreate-synonyms, --recreate-batch-processes, --add-api-key, --enable-vector, --clear-discourse, --discourse-stats, --clear-metabase-docs, --metabase-docs-stats, --clear-questions, --questions-stats, --clear-chat-sessions, --chat-sessions-stats, --clear-chat-session-entities, --chat-session-entities-stats, --clear-keyword-definitions, --keyword-definitions-stats, --clear-synonyms, --synonyms-stats, --clear-batch-processes, --batch-processes-stats, or --add-sample-keywords.")

if __name__ == "__main__":
    main()
