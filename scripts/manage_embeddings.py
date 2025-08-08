#!/usr/bin/env python3
"""
Embedding and LLM response management script for GitHub Duplicate Issue Finder.
This script provides functionality to delete embeddings and LLM responses from various tables.
"""

import sys
import logging
import argparse

# Use the shared path setup utility
from path_setup import setup_project_path
setup_project_path()

from sqlalchemy import text
from src.db import SessionLocal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages embeddings and LLM responses in the database."""
    
    def __init__(self):
        """Initialize the embedding manager."""
        self.db = SessionLocal()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
    
    def delete_all_embeddings(self) -> int:
        """Delete all embeddings from all tables."""
        try:
            logger.info("🗑️  Deleting all embeddings from all tables...")
            
            # Delete markdown embeddings from metabase_docs
            result1 = self.db.execute(
                text("UPDATE metabase_docs SET markdown_embedding = NULL, summary_embedding = NULL")
            )
            
            # Delete embeddings from issues
            result2 = self.db.execute(
                text("UPDATE issues SET title_embedding = NULL, issue_embedding = NULL, summary_embedding = NULL")
            )
            
            # Delete embeddings from discourse_posts
            result3 = self.db.execute(
                text("UPDATE discourse_posts SET conversation_embedding = NULL, summary_embedding = NULL")
            )
            
            # Delete all questions (which contain embeddings)
            result4 = self.db.execute(text("DELETE FROM questions"))
            
            self.db.commit()
            
            logger.info("✅ All embeddings deleted successfully")
            return 1
            
        except Exception as e:
            logger.error(f"❌ Error deleting all embeddings: {e}")
            self.db.rollback()
            return 0
    
    def delete_embeddings_from_table(self, table_name: str) -> int:
        """Delete embeddings from a specific table."""
        try:
            logger.info(f"🗑️  Deleting embeddings from table: {table_name}")
            
            if table_name == "metabase_docs":
                result = self.db.execute(
                    text("UPDATE metabase_docs SET markdown_embedding = NULL, summary_embedding = NULL")
                )
                # Also delete questions for metabase docs
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'METABASE_DOC'")
                )
            elif table_name == "issues":
                result = self.db.execute(
                    text("UPDATE issues SET title_embedding = NULL, issue_embedding = NULL, summary_embedding = NULL")
                )
                # Also delete questions for issues
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'ISSUE'")
                )
            elif table_name == "discourse_posts":
                result = self.db.execute(
                    text("UPDATE discourse_posts SET conversation_embedding = NULL, summary_embedding = NULL")
                )
                # Also delete questions for discourse posts
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'DISCOURSE_POST'")
                )
            elif table_name == "questions":
                result = self.db.execute(text("DELETE FROM questions"))
            else:
                logger.error(f"❌ Unknown table: {table_name}")
                return 0
            
            self.db.commit()
            logger.info(f"✅ Embeddings deleted from table: {table_name}")
            return 1
            
        except Exception as e:
            logger.error(f"❌ Error deleting embeddings from table {table_name}: {e}")
            self.db.rollback()
            return 0
    
    def delete_embeddings_from_id(self, table_name: str, record_id: int) -> int:
        """Delete embeddings from a specific ID in a table."""
        try:
            logger.info(f"🗑️  Deleting embeddings from {table_name} ID: {record_id}")
            
            if table_name == "metabase_docs":
                result = self.db.execute(
                    text("UPDATE metabase_docs SET markdown_embedding = NULL, summary_embedding = NULL WHERE id = :id"),
                    {"id": record_id}
                )
                # Also delete questions for this specific doc
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'METABASE_DOC' AND source_id = :id"),
                    {"id": record_id}
                )
            elif table_name == "issues":
                result = self.db.execute(
                    text("UPDATE issues SET title_embedding = NULL, issue_embedding = NULL, summary_embedding = NULL WHERE id = :id"),
                    {"id": record_id}
                )
                # Also delete questions for this specific issue
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'ISSUE' AND source_id = :id"),
                    {"id": record_id}
                )
            elif table_name == "discourse_posts":
                result = self.db.execute(
                    text("UPDATE discourse_posts SET conversation_embedding = NULL, summary_embedding = NULL WHERE id = :id"),
                    {"id": record_id}
                )
                # Also delete questions for this specific discourse post
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'DISCOURSE_POST' AND source_id = :id"),
                    {"id": record_id}
                )
            elif table_name == "questions":
                result = self.db.execute(
                    text("DELETE FROM questions WHERE id = :id"),
                    {"id": record_id}
                )
            else:
                logger.error(f"❌ Unknown table: {table_name}")
                return 0
            
            self.db.commit()
            logger.info(f"✅ Embeddings deleted from {table_name} ID: {record_id}")
            return 1
            
        except Exception as e:
            logger.error(f"❌ Error deleting embeddings from {table_name} ID {record_id}: {e}")
            self.db.rollback()
            return 0
    
    def delete_all_llm_responses(self) -> int:
        """Delete all LLM responses from all tables."""
        try:
            logger.info("🗑️  Deleting all LLM responses from all tables...")
            
            # Delete LLM summaries from metabase_docs
            result1 = self.db.execute(
                text("UPDATE metabase_docs SET llm_summary = NULL")
            )
            
            # Delete LLM summaries from issues
            result2 = self.db.execute(
                text("UPDATE issues SET llm_summary = NULL")
            )
            
            # Delete LLM summaries from discourse_posts
            result3 = self.db.execute(
                text("UPDATE discourse_posts SET llm_summary = NULL")
            )
            
            # Delete all questions (which are LLM responses)
            result4 = self.db.execute(text("DELETE FROM questions"))
            
            self.db.commit()
            
            logger.info("✅ All LLM responses deleted successfully")
            return 1
            
        except Exception as e:
            logger.error(f"❌ Error deleting all LLM responses: {e}")
            self.db.rollback()
            return 0
    
    def delete_llm_responses_from_table(self, table_name: str) -> int:
        """Delete LLM responses from a specific table."""
        try:
            logger.info(f"🗑️  Deleting LLM responses from table: {table_name}")
            
            if table_name == "metabase_docs":
                result = self.db.execute(
                    text("UPDATE metabase_docs SET llm_summary = NULL")
                )
                # Also delete questions for metabase docs
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'METABASE_DOC'")
                )
            elif table_name == "issues":
                result = self.db.execute(
                    text("UPDATE issues SET llm_summary = NULL")
                )
                # Also delete questions for issues
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'ISSUE'")
                )
            elif table_name == "discourse_posts":
                result = self.db.execute(
                    text("UPDATE discourse_posts SET llm_summary = NULL")
                )
                # Also delete questions for discourse posts
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'DISCOURSE_POST'")
                )
            elif table_name == "questions":
                result = self.db.execute(text("DELETE FROM questions"))
            else:
                logger.error(f"❌ Unknown table: {table_name}")
                return 0
            
            self.db.commit()
            logger.info(f"✅ LLM responses deleted from table: {table_name}")
            return 1
            
        except Exception as e:
            logger.error(f"❌ Error deleting LLM responses from table {table_name}: {e}")
            self.db.rollback()
            return 0
    
    def delete_llm_responses_from_id(self, table_name: str, record_id: int) -> int:
        """Delete LLM responses from a specific ID in a table."""
        try:
            logger.info(f"🗑️  Deleting LLM responses from {table_name} ID: {record_id}")
            
            if table_name == "metabase_docs":
                result = self.db.execute(
                    text("UPDATE metabase_docs SET llm_summary = NULL WHERE id = :id"),
                    {"id": record_id}
                )
                # Also delete questions for this specific doc
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'METABASE_DOC' AND source_id = :id"),
                    {"id": record_id}
                )
            elif table_name == "issues":
                result = self.db.execute(
                    text("UPDATE issues SET llm_summary = NULL WHERE id = :id"),
                    {"id": record_id}
                )
                # Also delete questions for this specific issue
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'ISSUE' AND source_id = :id"),
                    {"id": record_id}
                )
            elif table_name == "discourse_posts":
                result = self.db.execute(
                    text("UPDATE discourse_posts SET llm_summary = NULL WHERE id = :id"),
                    {"id": record_id}
                )
                # Also delete questions for this specific discourse post
                self.db.execute(
                    text("DELETE FROM questions WHERE source_type = 'DISCOURSE_POST' AND source_id = :id"),
                    {"id": record_id}
                )
            elif table_name == "questions":
                result = self.db.execute(
                    text("DELETE FROM questions WHERE id = :id"),
                    {"id": record_id}
                )
            else:
                logger.error(f"❌ Unknown table: {table_name}")
                return 0
            
            self.db.commit()
            logger.info(f"✅ LLM responses deleted from {table_name} ID: {record_id}")
            return 1
            
        except Exception as e:
            logger.error(f"❌ Error deleting LLM responses from {table_name} ID {record_id}: {e}")
            self.db.rollback()
            return 0
    
    def get_table_stats(self) -> dict:
        """Get statistics about embeddings and LLM responses in each table."""
        try:
            stats = {}
            
            # Metabase docs stats
            result = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(markdown_embedding) as with_embeddings,
                    COUNT(llm_summary) as with_llm_summaries,
                    COUNT(CASE WHEN markdown_embedding IS NOT NULL AND llm_summary IS NOT NULL THEN 1 END) as with_both
                FROM metabase_docs
            """)).fetchone()
            
            stats['metabase_docs'] = {
                'total': result[0] if result else 0 if result else 0,
                'with_embeddings': result[1] if result else 0,
                'with_llm_summaries': result[2] if result else 0,
                'with_both': result[3] if result else 0
            }
            
            # Issues stats
            result = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(title_embedding) as with_title_embeddings,
                    COUNT(issue_embedding) as with_body_embeddings,
                    COUNT(llm_summary) as with_llm_summaries,
                    COUNT(CASE WHEN title_embedding IS NOT NULL AND issue_embedding IS NOT NULL AND llm_summary IS NOT NULL THEN 1 END) as with_all
                FROM issues
            """)).fetchone()
            
            stats['issues'] = {
                'total': result[0] if result else 0 if result else 0,
                'with_title_embeddings': result[1] if result else 0,
                'with_body_embeddings': result[2] if result else 0,
                'with_llm_summaries': result[3] if result else 0,
                'with_all': result[4] if result else 0
            }
            
            # Discourse posts stats
            result = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(conversation_embedding) as with_embeddings,
                    COUNT(llm_summary) as with_llm_summaries,
                    COUNT(CASE WHEN conversation_embedding IS NOT NULL AND llm_summary IS NOT NULL THEN 1 END) as with_both
                FROM discourse_posts
            """)).fetchone()
            
            stats['discourse_posts'] = {
                'total': result[0] if result else 0 if result else 0,
                'with_embeddings': result[1] if result else 0,
                'with_llm_summaries': result[2] if result else 0,
                'with_both': result[3] if result else 0
            }
            
            # Questions stats
            result = self.db.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(question_embedding) as with_question_embeddings,
                    COUNT(answer_embedding) as with_answer_embeddings,
                    COUNT(CASE WHEN question_embedding IS NOT NULL AND answer_embedding IS NOT NULL THEN 1 END) as with_both_embeddings
                FROM questions
            """)).fetchone()
            
            stats['questions'] = {
                'total': result[0] if result else 0 if result else 0,
                'with_question_embeddings': result[1] if result else 0,
                'with_answer_embeddings': result[2] if result else 0,
                'with_both_embeddings': result[3] if result else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting table stats: {e}")
            return {}

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Manage embeddings and LLM responses')
    parser.add_argument('action', choices=['delete-embeddings', 'delete-llm', 'stats'],
                       help='Action to perform')
    parser.add_argument('--table', choices=['metabase_docs', 'issues', 'discourse_posts', 'questions'],
                       help='Specific table to operate on')
    parser.add_argument('--id', type=int, help='Specific record ID to operate on')
    parser.add_argument('--all', action='store_true', help='Operate on all tables')
    
    args = parser.parse_args()
    
    with EmbeddingManager() as manager:
        if args.action == 'stats':
            stats = manager.get_table_stats()
            print("\n📊 Database Statistics:")
            print("=" * 50)
            for table, data in stats.items():
                print(f"\n{table}:")
                print(f"  Total records: {data['total']}")
                if 'with_title_embeddings' in data:
                    print(f"  With title embeddings: {data['with_title_embeddings']}")
                    print(f"  With body embeddings: {data['with_body_embeddings']}")
                    print(f"  With LLM summaries: {data['with_llm_summaries']}")
                    print(f"  With all: {data['with_all']}")
                elif 'with_embeddings' in data:
                    print(f"  With embeddings: {data['with_embeddings']}")
                    print(f"  With LLM summaries: {data['with_llm_summaries']}")
                    print(f"  With both: {data['with_both']}")
                if 'with_question_embeddings' in data:
                    print(f"  With question embeddings: {data['with_question_embeddings']}")
                    print(f"  With answer embeddings: {data['with_answer_embeddings']}")
                    print(f"  With both embeddings: {data['with_both_embeddings']}")
        
        elif args.action == 'delete-embeddings':
            if args.all:
                manager.delete_all_embeddings()
            elif args.table and args.id:
                manager.delete_embeddings_from_id(args.table, args.id)
            elif args.table:
                manager.delete_embeddings_from_table(args.table)
            else:
                print("❌ Please specify --all, --table, or --table with --id")
        
        elif args.action == 'delete-llm':
            if args.all:
                manager.delete_all_llm_responses()
            elif args.table and args.id:
                manager.delete_llm_responses_from_id(args.table, args.id)
            elif args.table:
                manager.delete_llm_responses_from_table(args.table)
            else:
                print("❌ Please specify --all, --table, or --table with --id")

if __name__ == "__main__":
    main() 