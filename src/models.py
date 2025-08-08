from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, Enum, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from .db import Base
from .settings import EMBEDDING_DIM
import datetime
import enum
import uuid

class SourceType(enum.Enum):
    """Enum for different source types that can have questions."""
    METABASE_DOC = "metabase_doc"
    ISSUE = "issue"
    DISCOURSE_POST = "discourse_post"

class KeywordDefinition(Base):
    """
    SQLAlchemy model for keyword definitions used to provide context to LLM calls.
    """
    __tablename__ = 'keyword_definitions'

    id = Column(Integer, primary_key=True, index=True)
    keyword = Column(String, nullable=False, index=True)
    definition = Column(Text, nullable=False)
    category = Column(String, nullable=True)  # Optional category for organization
    is_active = Column(Boolean, default=True)
    keyword_embedding = Column(Vector(768), nullable=True)  # 768-dimensional embedding for keyword + definition + synonyms
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Synonym(Base):
    """
    SQLAlchemy model for word synonyms used for future features.
    """
    __tablename__ = 'synonyms'

    id = Column(Integer, primary_key=True, index=True)
    word = Column(String, nullable=False, index=True)
    synonym_of = Column(String, nullable=False, index=True)
    word_embedding = Column(Vector(768), nullable=True)  # 768-dimensional embedding for the word
    synonym_embedding = Column(Vector(768), nullable=True)  # 768-dimensional embedding for the synonym relationship
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Issue(Base):
    """
    SQLAlchemy model for a GitHub issue.
    """
    __tablename__ = 'issues'

    id = Column(Integer, primary_key=True, index=True)
    number = Column(Integer, unique=True, index=True)
    title = Column(String, nullable=False)
    body = Column(Text, nullable=True)
    state = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    labels = Column(JSON, nullable=True)
    user_login = Column(String, nullable=False)
    llm_summary = Column(Text, nullable=True)
    
    # New fields for advanced analysis
    reported_version = Column(String, nullable=True)
    stack_trace_file = Column(String, nullable=True)
    fixed_in_version = Column(String, nullable=True)
    token_count = Column(Integer, nullable=True)  # Token count for body field

    # Vector columns for embeddings
    title_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding for title
    issue_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding for body
    summary_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding for LLM summary
    
    # Note: Questions are linked via source_id + source_type, not a foreign key relationship

class DiscoursePost(Base):
    """
    SQLAlchemy model for a Discourse post/topic.
    """
    __tablename__ = 'discourse_posts'

    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    conversation = Column(Text, nullable=False)  # Using Text for large content
    created_at = Column(DateTime, nullable=False)
    slug = Column(String, nullable=False)
    llm_summary = Column(Text, nullable=True)  # LLM-generated summary
    type_of_topic = Column(String, nullable=True)  # bug, help, or feature_request
    solution = Column(Text, nullable=True)  # Solution extracted by LLM
    version = Column(String, nullable=True)  # Version mentioned in conversation
    reference = Column(String, nullable=True)  # URL reference mentioned in conversation
    token_count = Column(Integer, nullable=True)  # Token count for conversation field
    
    # Vector columns for embeddings (same dimensions as issues)
    conversation_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    summary_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding of LLM summary
    solution_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding of solution
    
    # Note: Questions are linked via source_id + source_type, not a foreign key relationship

class MetabaseDoc(Base):
    """
    SQLAlchemy model for Metabase documentation pages.
    """
    __tablename__ = 'metabase_docs'

    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=False)
    markdown = Column(Text, nullable=False)  # Using Text for large markdown content
    llm_summary = Column(Text, nullable=True)  # LLM-generated summary
    token_count = Column(Integer, nullable=True)  # Token count for markdown field
    markdown_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding of markdown content
    summary_embedding = Column(Vector(EMBEDDING_DIM), nullable=True)  # Embedding of LLM summary
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Note: Questions are linked via source_id + source_type, not a foreign key relationship

class Question(Base):
    """
    SQLAlchemy model for individual questions and answers extracted from various sources.
    """
    __tablename__ = 'questions'

    id = Column(Integer, primary_key=True, index=True)
    source_type = Column(Enum(SourceType), nullable=False, index=True)
    source_id = Column(Integer, nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    question_embedding = Column(Vector(768), nullable=True)  # 768-dimensional embedding for question
    answer_embedding = Column(Vector(768), nullable=True)    # 768-dimensional embedding for answer
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Note: Source documents are accessed via queries using source_id + source_type
    # SQLAlchemy relationships don't work well with this polymorphic design
    
    def get_source(self, db_session):
        """Get the source object based on source_type using database session."""
        from sqlalchemy import and_
        
        if str(self.source_type) == str(SourceType.METABASE_DOC):
            return db_session.query(MetabaseDoc).filter(MetabaseDoc.id == self.source_id).first()
        elif str(self.source_type) == str(SourceType.ISSUE):
            return db_session.query(Issue).filter(Issue.id == self.source_id).first()
        elif str(self.source_type) == str(SourceType.DISCOURSE_POST):
            return db_session.query(DiscoursePost).filter(DiscoursePost.id == self.source_id).first()
        return None

class ApiKey(Base):
    """
    SQLAlchemy model for an API key.
    """
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.utcnow) 

class ChatSession(Base):
    """
    SQLAlchemy model for tracking chat endpoint interactions.
    """
    __tablename__ = 'chat_sessions'

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, nullable=False, index=True)  # Chat ID passed by client to identify conversation (allows multiple sessions per conversation)
    user_request = Column(Text, nullable=False)  # Original user request
    prompt = Column(Text, nullable=True)  # Full prompt sent to the LLM
    sources = Column(JSON, nullable=True)  # Sources from /v2/similar endpoint
    response = Column(Text, nullable=True)  # Slow model response
    tokens_sent = Column(Integer, nullable=True)  # Prompt tokens sent to LLM
    tokens_received = Column(Integer, nullable=True)  # Completion tokens received from LLM
    cache_hit = Column(Boolean, nullable=True, default=False)  # Whether response was served from cache
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationship to chat session entities
    entities = relationship("ChatSessionEntity", back_populates="chat_session")

class ChatSessionEntity(Base):
    """
    SQLAlchemy model for tracking entity types and IDs that get injected into the context for each chat session.
    """
    __tablename__ = 'chat_session_entities'

    id = Column(Integer, primary_key=True, index=True)
    chat_id = Column(Integer, ForeignKey('chat_sessions.id'), nullable=False, index=True)  # References the specific session
    entity_type = Column(String, nullable=False, index=True)  # 'metabase_doc', 'question_answer', 'keyword', etc.
    entity_id = Column(Integer, nullable=False, index=True)  # ID of the entity in its respective table
    entity_url = Column(String, nullable=True)  # URL of the entity if available
    similarity_score = Column(Float, nullable=True)  # Similarity score as float (null for keywords)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationship to chat session
    chat_session = relationship("ChatSession", back_populates="entities")

class BatchProcess(Base):
    """
    SQLAlchemy model for tracking OpenAI batch processing operations.
    """
    __tablename__ = 'batch_processes'

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(String, nullable=False, unique=True, index=True)  # OpenAI batch ID
    provider = Column(String, nullable=False, default="openai")  # Provider name (openai, litellm, etc.)
    operation_type = Column(String, nullable=False, index=True)  # summarize, create-questions, create-questions-and-concepts
    table_name = Column(String, nullable=False, index=True)  # issues, discourse_posts, metabase_docs
    total_requests = Column(Integer, nullable=False, default=0)  # Number of requests in the batch
    sent_at = Column(DateTime, nullable=True)  # When the batch was sent
    received_at = Column(DateTime, nullable=True)  # When the batch results were received
    status = Column(String, nullable=False, default="created")  # created, sent, completed, failed, cancelled
    input_file_path = Column(String, nullable=True)  # Path to the input JSONL file
    output_file_path = Column(String, nullable=True)  # Path to the output JSONL file
    error_message = Column(Text, nullable=True)  # Error message if failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow) 