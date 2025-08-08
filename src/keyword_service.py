"""
Keyword definition service for managing specialized terminology and injecting context into LLM calls.
"""

import logging
import re
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from .db import SessionLocal
from .models import KeywordDefinition
import datetime

logger = logging.getLogger(__name__)

class KeywordService:
    """Service for managing keyword definitions and injecting them into LLM calls."""
    
    def __init__(self):
        """Initialize the keyword service."""
        pass
    
    def _generate_plural_forms(self, word: str) -> List[str]:
        """Generate potential plural forms of a word."""
        plurals = []
        
        if word.endswith('y'):
            plurals.append(f"{word[:-1]}ies")
        elif not word.endswith('s'):
            plurals.append(f"{word}s")
            
        return plurals
    
    def _check_keyword_match(self, keyword: str, message_lower: str, found_keywords: set, 
                           target_keyword: Optional[str] = None) -> bool:
        """
        Check if a keyword (and its variations) matches in the message.
        
        Args:
            keyword: The keyword to check
            message_lower: Lowercase version of the message
            found_keywords: Set of already found keywords
            target_keyword: The keyword to check in found_keywords (for synonyms)
            
        Returns:
            True if match found and not already processed
        """
        check_keyword = target_keyword or keyword
        
        if check_keyword in found_keywords:
            return False
        
        # Check exact match
        if keyword in message_lower:
            return True
        
        # Check plural forms
        plurals = self._generate_plural_forms(keyword)
        return any(plural in message_lower for plural in plurals)
    
    def get_active_keywords(self, db: Session) -> List[Dict[str, str]]:
        """
        Get all active keyword definitions from the database.
        
        Args:
            db: Database session
            
        Returns:
            List of dictionaries with keyword and definition
        """
        try:
            keywords = db.query(KeywordDefinition).filter(
                KeywordDefinition.is_active.is_(True)
            ).all()
            
            return [
                {
                    'keyword': str(kw.keyword),
                    'definition': str(kw.definition), 
                    'category': str(kw.category) if kw.category is not None else ""
                }
                for kw in keywords
            ]
        except Exception as e:
            logger.error(f"Error fetching keyword definitions: {e}")
            return []
    
    def get_relevant_keywords(self, message: str, db: Session) -> List[Dict[str, str]]:
        """
        Get keywords that are relevant to the specific message using pattern matching.
        
        Args:
            message: The message text to check for keywords
            db: Database session
            
        Returns:
            List of relevant keyword definitions
        """
        try:
            # Get all active keywords
            keywords = db.query(KeywordDefinition).filter(
                KeywordDefinition.is_active.is_(True)
            ).all()
            
            if not keywords:
                return []
            
            # Convert message to lowercase for case-insensitive matching
            message_lower = message.lower()
            
            # Get synonyms
            from .models import Synonym
            synonyms = db.query(Synonym).all()
            
            # Build keyword to definition mapping
            keyword_map = {}
            for kw in keywords:
                keyword_map[kw.keyword.lower()] = {
                    'keyword': kw.keyword,
                    'definition': kw.definition,
                    'category': kw.category
                }
            
            # Build synonym to keyword mapping
            synonym_map = {}
            for syn in synonyms:
                synonym_map[syn.word.lower()] = syn.synonym_of.lower()
            
            # Find relevant keywords
            relevant_keywords = []
            found_keywords = set()  # To avoid duplicates
            
            # Check each keyword and its variations
            for kw in keywords:
                keyword = kw.keyword.lower()
                
                if self._check_keyword_match(keyword, message_lower, found_keywords):
                    relevant_keywords.append(keyword_map[keyword])
                    found_keywords.add(keyword)
            
            # Check synonyms
            for synonym_word, original_keyword in synonym_map.items():
                if original_keyword not in keyword_map:
                    continue
                    
                if self._check_keyword_match(synonym_word, message_lower, found_keywords, original_keyword):
                    relevant_keywords.append(keyword_map[original_keyword])
                    found_keywords.add(original_keyword)
            
            logger.debug(f"Found {len(relevant_keywords)} relevant keywords out of {len(keywords)} total keywords")
            logger.debug(f"Relevant keywords: {[kw['keyword'] for kw in relevant_keywords]}")
            
            return relevant_keywords
            
        except Exception as e:
            logger.error(f"Error finding relevant keywords: {e}")
            return []
    
    def inject_relevant_keywords_into_prompt(self, prompt: str, db: Session) -> str:
        """
        Inject only relevant keyword definitions into a prompt for LLM context.
        
        Args:
            prompt: The original prompt
            db: Database session
            
        Returns:
            Enhanced prompt with relevant keyword definitions
        """
        relevant_keywords = self.get_relevant_keywords(prompt, db)
        
        if not relevant_keywords:
            return prompt
        
        # Group keywords by category for better organization
        keywords_by_category = {}
        for kw in relevant_keywords:
            category = kw.get('category', 'General')
            if category not in keywords_by_category:
                keywords_by_category[category] = []
            keywords_by_category[category].append(kw)
        
        # Build the keyword context section
        keyword_context = "\n\nIMPORTANT CONTEXT - Relevant Specialized Terminology:\n"
        keyword_context += "The following terms are mentioned in your request. Please use these definitions:\n\n"
        
        for category, category_keywords in keywords_by_category.items():
            if category != 'General':
                keyword_context += f"--- {category} ---\n"
            
            for kw in category_keywords:
                keyword_context += f"• {kw['keyword']}: {kw['definition']}\n"
            
            keyword_context += "\n"
        
        keyword_context += "Please consider these definitions when generating your response.\n"
        
        # Insert the keyword context at the beginning of the prompt
        return keyword_context + "\n" + prompt
    
    def add_keyword_definition(self, keyword: str, definition: str, category: Optional[str] = None) -> bool:
        """
        Add a new keyword definition to the database.
        
        Args:
            keyword: The keyword to define
            definition: The definition/meaning of the keyword
            category: Optional category for organization
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            # Check if keyword already exists
            existing = db.query(KeywordDefinition).filter(
                KeywordDefinition.keyword == keyword
            ).first()
            
            if existing:
                logger.warning(f"Keyword '{keyword}' already exists")
                return False
            
            new_keyword = KeywordDefinition(
                keyword=keyword,
                definition=definition,
                category=category,
                is_active=True
            )
            
            db.add(new_keyword)
            db.commit()
            logger.info(f"Added keyword definition: {keyword}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding keyword definition: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def update_keyword_definition(self, keyword: str, definition: str, category: Optional[str] = None) -> bool:
        """
        Update an existing keyword definition.
        
        Args:
            keyword: The keyword to update
            definition: The new definition
            category: Optional new category
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            keyword_def = db.query(KeywordDefinition).filter(
                KeywordDefinition.keyword == keyword
            ).first()
            
            if not keyword_def:
                logger.warning(f"Keyword '{keyword}' not found")
                return False
            
            # Update using SQL to avoid SQLAlchemy column assignment issues
            update_data = {
                KeywordDefinition.definition: definition, 
                KeywordDefinition.updated_at: datetime.datetime.utcnow()
            }
            if category is not None:
                update_data[KeywordDefinition.category] = category
                
            db.query(KeywordDefinition).filter(
                KeywordDefinition.keyword == keyword
            ).update(update_data)
            
            db.commit()
            logger.info(f"Updated keyword definition: {keyword}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating keyword definition: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def delete_keyword_definition(self, keyword: str) -> bool:
        """
        Delete a keyword definition from the database.
        
        Args:
            keyword: The keyword to delete
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            keyword_def = db.query(KeywordDefinition).filter(
                KeywordDefinition.keyword == keyword
            ).first()
            
            if not keyword_def:
                logger.warning(f"Keyword '{keyword}' not found")
                return False
            
            db.delete(keyword_def)
            db.commit()
            logger.info(f"Deleted keyword definition: {keyword}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting keyword definition: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def toggle_keyword_status(self, keyword: str) -> bool:
        """
        Toggle the active status of a keyword definition.
        
        Args:
            keyword: The keyword to toggle
            
        Returns:
            True if successful, False otherwise
        """
        db = SessionLocal()
        try:
            keyword_def = db.query(KeywordDefinition).filter(
                KeywordDefinition.keyword == keyword
            ).first()
            
            if not keyword_def:
                logger.warning(f"Keyword '{keyword}' not found")
                return False
            
            # Get current status and toggle it
            current_status = bool(keyword_def.is_active)
            new_status = not current_status
            
            # Update using query to avoid column assignment issues
            db.query(KeywordDefinition).filter(
                KeywordDefinition.keyword == keyword
            ).update({
                KeywordDefinition.is_active: new_status,
                KeywordDefinition.updated_at: datetime.datetime.utcnow()
            })
            
            db.commit()
            status = "activated" if new_status == 'true' else "deactivated"
            logger.info(f"{status.capitalize()} keyword definition: {keyword}")
            return True
            
        except Exception as e:
            logger.error(f"Error toggling keyword status: {e}")
            db.rollback()
            return False
        finally:
            db.close()
    
    def inject_keywords_into_prompt(self, prompt: str, db: Session) -> str:
        """
        Inject active keyword definitions into a prompt for LLM context.
        
        Args:
            prompt: The original prompt
            db: Database session
            
        Returns:
            Enhanced prompt with keyword definitions
        """
        keywords = self.get_active_keywords(db)
        
        if not keywords:
            return prompt
        
        # Group keywords by category for better organization
        keywords_by_category = {}
        for kw in keywords:
            category = kw.get('category', 'General')
            if category not in keywords_by_category:
                keywords_by_category[category] = []
            keywords_by_category[category].append(kw)
        
        # Build the keyword context section
        keyword_context = "\n\nIMPORTANT CONTEXT - Specialized Terminology:\n"
        keyword_context += "Please use the following definitions when analyzing the content:\n\n"
        
        for category, category_keywords in keywords_by_category.items():
            if category != 'General':
                keyword_context += f"--- {category} ---\n"
            
            for kw in category_keywords:
                keyword_context += f"• {kw['keyword']}: {kw['definition']}\n"
            
            keyword_context += "\n"
        
        keyword_context += "Please consider these definitions when generating your response.\n"
        
        # Insert the keyword context at the beginning of the prompt
        return keyword_context + "\n" + prompt
    
    def list_keywords(self, db: Session, category: Optional[str] = None) -> List[Dict]:
        """
        List all keyword definitions, optionally filtered by category.
        
        Args:
            db: Database session
            category: Optional category filter
            
        Returns:
            List of keyword definitions
        """
        try:
            query = db.query(KeywordDefinition)
            
            if category:
                query = query.filter(KeywordDefinition.category == category)
            
            keywords = query.order_by(KeywordDefinition.keyword).all()
            
            return [
                {
                    'id': kw.id,
                    'keyword': kw.keyword,
                    'definition': kw.definition,
                    'category': kw.category,
                    'is_active': kw.is_active,
                    'created_at': kw.created_at,
                    'updated_at': kw.updated_at
                }
                for kw in keywords
            ]
        except Exception as e:
            logger.error(f"Error listing keywords: {e}")
            return [] 