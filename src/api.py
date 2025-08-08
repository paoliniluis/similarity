from fastapi import FastAPI, Depends, Security, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
from typing import List, Optional, Dict, Any
import logging
import re
import html
import uuid
import asyncio

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.db import get_db, SessionLocal
from src.models import Issue, DiscoursePost, MetabaseDoc, Question, ChatSession, ChatSessionEntity
from src.embedding_service import get_embedding_service
from src.similarity_query_builder import SimilarityQueryBuilder
# Removed unused imports from src.api_utils
from src.security import get_api_key
from src.llm_client import llm_client
from src.reranker_service import get_reranker_service
from src.utils import get_device
from src import settings
from src.prompts import get_api_chat_system_prompt, get_api_context_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="GitHub Duplicate Issue Finder API",
    description="An API to find semantically similar GitHub issues stored in a PostgreSQL database.",
    version="1.0.0",
)

# Auto-detect best available device
device = get_device()
logger.info(f"🚀 Initializing models with device: {device}")

# Use the new embedding service instead of SemanticAnalyzer
embedding_service = get_embedding_service()
query_builder = SimilarityQueryBuilder()

# Initialize reranker service (local or API) if enabled
reranker_service = get_reranker_service()

class SearchRequest(BaseModel):
    """Request model for the similarity search endpoint."""
    text: str = Field(..., description="The text to search for similar issues.")
    state: Optional[str] = Field(None, description="Filter by issue state: 'open', 'closed', or leave empty for all issues.")

    @field_validator('state')
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        """Validate that state parameter is one of the allowed values."""
        if v is not None and v.lower() not in ["open", "closed"]:
            raise ValueError("State parameter must be either 'open' or 'closed'")
        return v.lower() if v else None

class SimilarIssueResponse(BaseModel):
    """Response model for a single similar issue, including detailed similarity scores."""
    number: int = Field(..., description="The GitHub issue number.")
    title: str = Field(..., description="The title of the GitHub issue.")
    state: str = Field(..., description="The state of the GitHub issue.")
    url: str = Field(..., description="The GitHub issue URL.")
    similarity_score: float = Field(..., description="Overall similarity score based on title, body, and summary embeddings.")

class SimilarDiscourseResponse(BaseModel):
    """Response model for a single similar discourse post."""
    id: int = Field(..., description="The discourse post ID.")
    title: str = Field(..., description="The title of the discourse post.")
    url: str = Field(..., description="The discourse post URL.")
    similarity_score: float = Field(..., description="Similarity score based on conversation and summary embeddings.")

class SimilarDocumentationResponse(BaseModel):
    """Response model for a single similar documentation page."""
    id: int = Field(..., description="The documentation page ID.")
    url: str = Field(..., description="The documentation page URL.")
    similarity_score: float = Field(..., description="Similarity score based on markdown content and summary embeddings.")

class SimilarQuestionResponse(BaseModel):
    """Response model for a single similar question."""
    id: int = Field(..., description="The question ID.")
    question: str = Field(..., description="The question text.")
    answer: str = Field(..., description="The answer text.")
    url: str = Field(..., description="The URL to the source document containing this question/answer.")
    similarity_score: float = Field(..., description="Similarity score based on question and answer embeddings.")

class SimilarKeywordResponse(BaseModel):
    """Response model for a single similar keyword."""
    id: int = Field(..., description="The keyword ID.")
    keyword: str = Field(..., description="The keyword.")
    definition: str = Field(..., description="The keyword definition.")
    category: Optional[str] = Field(None, description="The keyword category.")
    similarity_score: float = Field(..., description="Similarity score based on keyword embedding.")

class V2SimilarResponse(BaseModel):
    """Response model for v2 similar endpoint that returns all types."""
    issues: List[SimilarIssueResponse] = Field(..., description="Similar GitHub issues.")
    discourse_posts: List[SimilarDiscourseResponse] = Field(..., description="Similar discourse posts.")
    metabase_docs: List[SimilarDocumentationResponse] = Field(..., description="Similar documentation pages.")
    questions: List[SimilarQuestionResponse] = Field(..., description="Similar questions and answers.")
    keywords: List[SimilarKeywordResponse] = Field(..., description="Similar keywords.")

class RerankRequest(BaseModel):
    """Request model for the rerank endpoint."""
    query: str = Field(..., description="The query to rerank against.")
    candidates: List[Dict[str, Any]] = Field(..., description="A list of candidate documents to rerank.")

class RerankResponse(BaseModel):
    """Response model for the rerank endpoint."""
    reranked_candidates: List[Dict[str, Any]] = Field(..., description="The reranked list of candidates.")

class EmbeddingRequest(BaseModel):
    """Request model for the embedding endpoint."""
    text: str = Field(..., description="The text to create an embedding for.")

class EmbeddingResponse(BaseModel):
    """Response model for the embedding endpoint."""
    embedding: List[float] = Field(..., description="The 768-dimensional embedding vector.")

class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    text: str = Field(..., description="The user's question or request.")
    chat_id: int = Field(..., description="The chat ID to identify the conversation session.")

class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    answer: str = Field(..., description="The AI's response to the user's question.")
    sources: List[str] = Field(..., description="List of source URLs used to generate the answer.")





# --- POST endpoint for embeddings ---
@app.post("/embedding", response_model=EmbeddingResponse)
# @limiter.limit("100/minute")
def create_embedding(
    request: Request,
    embedding_request: EmbeddingRequest,
    api_key: str = Security(get_api_key)
) -> EmbeddingResponse:
    """
    Create a 768-dimensional embedding vector for the given text.
    """
    logger.info(f"🌐 POST /embedding for text: '{embedding_request.text[:50]}...'")
    
    try:
        embedding = embedding_service.create_embedding(embedding_request.text)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to create embedding")
        logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")
        
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"❌ Error creating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating embedding: {str(e)}")

# --- POST endpoint for reranking ---
@app.post("/rerank", response_model=RerankResponse)
@limiter.limit("10/minute")
def rerank_results(
    request: Request,
    rerank_request: RerankRequest,
    api_key: str = Security(get_api_key)
) -> RerankResponse:
    """
    Rerank a list of candidates based on a query.
    """
    logger.info(f"🌐 POST /rerank for query: '{rerank_request.query[:50]}...' with {len(rerank_request.candidates)} candidates")

    if not reranker_service:
        raise HTTPException(status_code=503, detail="Reranker service is not available")

    reranked_candidates = reranker_service.rerank_results(
        query=rerank_request.query,
        candidates=rerank_request.candidates
    )

    return RerankResponse(reranked_candidates=reranked_candidates)

# --- POST endpoint at /v1/similar-github-issues ---
@app.post("/v1/similar-github-issues", response_model=List[SimilarIssueResponse])
@limiter.limit("10/minute")
def find_similar_github_issues_v1(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarIssueResponse]:
    """
    Find most similar GitHub issues based on a text query, with optional state filtering.
    The search is performed across issue titles, bodies, and summaries using vector embeddings.
    """
    logger.info(f"🌐 POST /v1/similar-github-issues for text: '{search_request.text[:50]}...' state: {search_request.state}")

    state = search_request.state
    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Use the new query builder
    columns = {
        'id': 'number',
        'select_cols': 'number, title, state',
        'title_embedding': 'title_embedding',
        'issue_embedding': 'issue_embedding', 
        'summary_embedding': 'summary_embedding',
        'group_by': 'number, title, state'
    }
    
    where_clause = "state = :state_param" if state else None
    where_params = {'state_param': state} if state else None
    
    logger.info("Executing similarity CTE query...")
    
    result = query_builder.execute_similarity_query(
        db, 'issues', embedding, columns, where_clause, where_params
    )
    
    issues = []
    for row in result:
        issues.append(SimilarIssueResponse(
            number=row.number,
            title=row.title,
            state=row.state,
            url=f"{settings.GITHUB_BASE_URL}/issues/{row.number}",
            similarity_score=float(row.similarity)
        ))
    return issues

# --- POST endpoint at /v1/similar-metabase-docs ---
@app.post("/v1/similar-metabase-docs", response_model=List[SimilarDocumentationResponse])
@limiter.limit("10/minute")
def find_similar_metabase_docs_v1(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarDocumentationResponse]:
    """
    Find most similar Metabase documentation pages based on a text query.
    The search is performed across markdown content and summaries using vector embeddings.
    """
    logger.info(f"POST /v1/similar-metabase-docs for text: '{search_request.text[:50]}...'")

    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Use centralized query builder
    logger.info("Executing metabase docs similarity query via builder...")
    columns = {
        'id': 'id',
        'select_cols': 'id, url',
        'content_embedding': 'markdown_embedding',
        'summary_embedding': 'summary_embedding',
        'group_by': 'id, url'
    }
    result = query_builder.execute_similarity_query(
        db, 'metabase_docs', embedding, columns, None, None, limit=10
    )
    docs = []
    for row in result:
        docs.append(SimilarDocumentationResponse(
            id=row.id,
            url=row.url,
            similarity_score=float(row.similarity)
        ))
    return docs

# --- POST endpoint at /v1/similar-discourse-posts ---
@app.post("/v1/similar-discourse-posts", response_model=List[SimilarDiscourseResponse])
@limiter.limit("10/minute")
def find_similar_discourse_posts_v1(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarDiscourseResponse]:
    """
    Find most similar discourse posts based on a text query.
    The search is performed across conversation content and summaries using vector embeddings.
    """
    logger.info(f"POST /v1/similar-discourse-posts for text: '{search_request.text[:50]}...'")

    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Use centralized query builder
    logger.info("Executing discourse similarity query via builder...")
    columns = {
        'id': 'id',
        'select_cols': 'id, topic_id, title, slug',
        'content_embedding': 'conversation_embedding',
        'summary_embedding': 'summary_embedding',
        'group_by': 'id, topic_id, title, slug'
    }
    result = query_builder.execute_similarity_query(
        db, 'discourse_posts', embedding, columns, None, None, limit=10
    )
    posts = []
    for row in result:
        posts.append(SimilarDiscourseResponse(
            id=row.id,
            title=row.title,
            url=f"{settings.DISCOURSE_BASE_URL}/t/{row.slug}/{row.topic_id}",
            similarity_score=float(row.similarity)
        ))
    return posts

# --- POST endpoint at /v1/similar-questions ---
@app.post("/v1/similar-questions", response_model=List[SimilarQuestionResponse])
@limiter.limit("10/minute")
def find_similar_questions_v1(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarQuestionResponse]:
    """
    Find most similar questions based on a text query.
    The search is performed across question and answer embeddings using vector embeddings.
    """
    logger.info(f"POST /v1/similar-questions for text: '{search_request.text[:50]}...'")

    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Use centralized query builder, then construct URLs in Python
    logger.info("Executing questions similarity query via builder...")
    columns = {
        'id': 'id',
        'select_cols': 'id, question, answer, source_type, source_id',
        'question_embedding': 'question_embedding',
        'answer_embedding': 'answer_embedding',
        'group_by': 'id, question, answer, source_type, source_id'
    }
    result = query_builder.execute_similarity_query(
        db, 'questions', embedding, columns, None, None, limit=20
    )

    # Collect IDs per source type for URL building
    from src.models import Issue, DiscoursePost, MetabaseDoc, SourceType
    issue_ids = []
    discourse_ids = []
    doc_ids = []
    rows = list(result)
    for row in rows:
        st = str(row.source_type)
        if st.endswith('ISSUE') or st.endswith('issue'):
            issue_ids.append(row.source_id)
        elif st.endswith('DISCOURSE_POST') or st.endswith('discourse_post'):
            discourse_ids.append(row.source_id)
        elif st.endswith('METABASE_DOC') or st.endswith('metabase_doc'):
            doc_ids.append(row.source_id)

    issues_map = {}
    discourse_map = {}
    docs_map = {}
    if issue_ids:
        for i in db.query(Issue).filter(Issue.id.in_(issue_ids)).all():
            issues_map[i.id] = i
    if discourse_ids:
        for d in db.query(DiscoursePost).filter(DiscoursePost.id.in_(discourse_ids)).all():
            discourse_map[d.id] = d
    if doc_ids:
        for m in db.query(MetabaseDoc).filter(MetabaseDoc.id.in_(doc_ids)).all():
            docs_map[m.id] = m

    responses = []
    for row in rows:
        url = None
        st = str(row.source_type)
        if st.endswith('ISSUE') or st.endswith('issue'):
            issue = issues_map.get(row.source_id)
            if issue and getattr(issue, 'number', None) is not None:
                url = f"{settings.GITHUB_BASE_URL}/issues/{issue.number}"
        elif st.endswith('DISCOURSE_POST') or st.endswith('discourse_post'):
            post = discourse_map.get(row.source_id)
            if post and getattr(post, 'slug', None) is not None and getattr(post, 'topic_id', None) is not None:
                url = f"{settings.DISCOURSE_BASE_URL}/t/{post.slug}/{post.topic_id}"
        elif st.endswith('METABASE_DOC') or st.endswith('metabase_doc'):
            doc = docs_map.get(row.source_id)
            if doc and getattr(doc, 'url', None):
                url = doc.url

        if url:
            responses.append(SimilarQuestionResponse(
                id=row.id,
                question=row.question,
                answer=row.answer,
                url=url,
                similarity_score=float(row.similarity)
            ))

    # Limit to top 10
    responses = sorted(responses, key=lambda r: r.similarity_score, reverse=True)[:10]
    return responses



# --- POST endpoint at /v2/similar-github-issues ---
@app.post("/v2/similar-github-issues", response_model=List[SimilarIssueResponse])
@limiter.limit("10/minute")
def find_similar_github_issues_v2(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarIssueResponse]:
    """
    Find most similar GitHub issues based on a text query with reranking.
    The search is performed across issue titles, bodies, and summaries using vector embeddings,
    then reranked using the reranker model for better relevance.
    """
    logger.info(f"🌐 POST /v2/similar-github-issues for text: '{search_request.text[:50]}...' state: {search_request.state}")

    if not reranker_service:
        logger.warning("Reranker not available, falling back to v1 endpoint")
        return find_similar_github_issues_v1(request, search_request, db, api_key)

    state = search_request.state
    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Build the SQL query with CTEs
    embedding_str = ','.join(str(v) for v in embedding)
    embedding_sql = f"'[{embedding_str}]'::vector"
    state_filter = ""
    if state:
        state_filter = "AND state = :state_param"
    sql = f"""
    WITH issue_sim AS (
        SELECT number, title, state, body, 1 - (issue_embedding <=> {embedding_sql}) AS similarity
        FROM issues
        WHERE issue_embedding IS NOT NULL {state_filter}
        AND 1 - (issue_embedding <=> {embedding_sql}) > 0.5
        ORDER BY issue_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    summary_sim AS (
        SELECT number, title, state, body, 1 - (summary_embedding <=> {embedding_sql}) AS similarity
        FROM issues
        WHERE summary_embedding IS NOT NULL {state_filter}
        AND 1 - (summary_embedding <=> {embedding_sql}) > 0.5
        ORDER BY summary_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    all_sim AS (
        SELECT * FROM issue_sim
        UNION ALL
        SELECT * FROM summary_sim
    )
    SELECT number, title, state, body, MAX(similarity) AS similarity
    FROM all_sim
    GROUP BY number, title, state, body
    ORDER BY similarity DESC
    LIMIT 50;
    """

    logger.info("Executing similarity query for reranking...")
    
    # Execute with parameters to prevent SQL injection
    params = {}
    if state:
        params['state_param'] = state
    
    result = db.execute(sql_text(sql), params)
    
    # Prepare candidates for reranking
    candidates = []
    for row in result:
        candidates.append({
            'id': row.number,
            'title': row.title,
            'state': row.state,
            'body': row.body or '',
            'source_type': 'issue',
            'url': f"{settings.GITHUB_BASE_URL}/issues/{row.number}",
            'similarity_score': float(row.similarity)
        })
    
    if not candidates:
        logger.info("No candidates found for reranking")
        return []
    
    # Rerank the candidates
    logger.info(f"Reranking {len(candidates)} candidates...")
    reranked_response = rerank_results(
        request,
        RerankRequest(query=search_request.text, candidates=candidates),
        api_key
    )
    reranked_candidates = reranked_response.reranked_candidates
    
    # Limit the number of results after reranking
    reranked_candidates = reranked_candidates[:settings.RERANKER_MAX_CANDIDATES]
    
    # Convert to response format - only return items with positive similarity scores
    issues = []
    for candidate in reranked_candidates:
        similarity_score = candidate.get('reranker_score', candidate['similarity_score'])
        if similarity_score > 0:
            issues.append(SimilarIssueResponse(
                number=candidate['id'],
                title=candidate['title'],
                state=candidate['state'],
                url=candidate['url'],
                similarity_score=similarity_score
            ))
    
    return issues

# --- POST endpoint at /v2/similar-metabase-docs ---
@app.post("/v2/similar-metabase-docs", response_model=List[SimilarDocumentationResponse])
@limiter.limit("10/minute")
def find_similar_metabase_docs_v2(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarDocumentationResponse]:
    """
    Find most similar Metabase documentation pages based on a text query with reranking.
    The search is performed across markdown content and summaries using vector embeddings,
    then reranked using the reranker model for better relevance.
    """
    logger.info(f"🌐 POST /v2/similar-metabase-docs for text: '{search_request.text[:50]}...'")

    if not reranker_service:
        logger.warning("Reranker not available, falling back to v1 endpoint")
        return find_similar_metabase_docs_v1(request, search_request, db, api_key)

    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Build the SQL query with CTEs
    embedding_str = ','.join(str(v) for v in embedding)
    embedding_sql = f"'[{embedding_str}]'::vector"
    sql = f"""
    WITH content_sim AS (
        SELECT id, url, markdown, 1 - (markdown_embedding <=> {embedding_sql}) AS similarity
        FROM metabase_docs
        WHERE markdown_embedding IS NOT NULL
        AND 1 - (markdown_embedding <=> {embedding_sql}) > 0.5
        ORDER BY markdown_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    summary_sim AS (
        SELECT id, url, markdown, 1 - (summary_embedding <=> {embedding_sql}) AS similarity
        FROM metabase_docs
        WHERE summary_embedding IS NOT NULL
        AND 1 - (summary_embedding <=> {embedding_sql}) > 0.5
        ORDER BY summary_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    all_sim AS (
        SELECT * FROM content_sim
        UNION ALL
        SELECT * FROM summary_sim
    )
    SELECT id, url, markdown, MAX(similarity) AS similarity
    FROM all_sim
    GROUP BY id, url, markdown
    ORDER BY similarity DESC
    LIMIT 50;
    """

    logger.info("Executing metabase docs similarity query for reranking...")
    
    result = db.execute(sql_text(sql))
    
    # Prepare candidates for reranking
    candidates = []
    for row in result:
        candidates.append({
            'id': row.id,
            'title': row.url,
            'markdown': row.markdown or '',
            'source_type': 'docs',
            'url': row.url,
            'similarity_score': float(row.similarity)
        })
    
    if not candidates:
        logger.info("No candidates found for reranking")
        return []
    
    # Rerank the candidates
    logger.info(f"Reranking {len(candidates)} candidates...")
    reranked_response = rerank_results(
        request,
        RerankRequest(query=search_request.text, candidates=candidates),
        api_key
    )
    reranked_candidates = reranked_response.reranked_candidates
    
    # Limit the number of results after reranking
    reranked_candidates = reranked_candidates[:settings.RERANKER_MAX_CANDIDATES]
    
    # Convert to response format - only return items with positive similarity scores
    docs = []
    for candidate in reranked_candidates:
        similarity_score = candidate.get('reranker_score', candidate['similarity_score'])
        if similarity_score > 0:
            docs.append(SimilarDocumentationResponse(
                id=candidate['id'],
                url=candidate['url'],
                similarity_score=similarity_score
            ))
    
    return docs

# --- POST endpoint at /v2/similar-discourse-posts ---
@app.post("/v2/similar-discourse-posts", response_model=List[SimilarDiscourseResponse])
@limiter.limit("10/minute")
def find_similar_discourse_posts_v2(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarDiscourseResponse]:
    """
    Find most similar discourse posts based on a text query with reranking.
    The search is performed across conversation content and summaries using vector embeddings,
    then reranked using the reranker model for better relevance.
    """
    logger.info(f"🌐 POST /v2/similar-discourse-posts for text: '{search_request.text[:50]}...'")

    if not reranker_service:
        logger.warning("Reranker not available, falling back to v1 endpoint")
        return find_similar_discourse_posts_v1(request, search_request, db, api_key)

    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Build the SQL query with CTEs
    embedding_str = ','.join(str(v) for v in embedding)
    embedding_sql = f"'[{embedding_str}]'::vector"
    sql = f"""
    WITH conversation_sim AS (
        SELECT id, topic_id, title, slug, conversation, 1 - (conversation_embedding <=> {embedding_sql}) AS similarity
        FROM discourse_posts
        WHERE conversation_embedding IS NOT NULL
        AND 1 - (conversation_embedding <=> {embedding_sql}) > 0.5
        ORDER BY conversation_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    summary_sim AS (
        SELECT id, topic_id, title, slug, conversation, 1 - (summary_embedding <=> {embedding_sql}) AS similarity
        FROM discourse_posts
        WHERE summary_embedding IS NOT NULL
        AND 1 - (summary_embedding <=> {embedding_sql}) > 0.5
        ORDER BY summary_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    all_sim AS (
        SELECT * FROM conversation_sim
        UNION ALL
        SELECT * FROM summary_sim
    )
    SELECT id, topic_id, title, slug, conversation, MAX(similarity) AS similarity
    FROM all_sim
    GROUP BY id, topic_id, title, slug, conversation
    ORDER BY similarity DESC
    LIMIT 50;
    """

    logger.info("Executing discourse similarity query for reranking...")
    
    result = db.execute(sql_text(sql))
    
    # Prepare candidates for reranking
    candidates = []
    for row in result:
        candidates.append({
            'id': row.id,
            'title': row.title,
            'conversation': row.conversation or '',
            'source_type': 'discourse',
            'url': f"{settings.DISCOURSE_BASE_URL}/t/{row.slug}/{row.topic_id}",
            'similarity_score': float(row.similarity)
        })
    
    if not candidates:
        logger.info("No candidates found for reranking")
        return []
    
    # Rerank the candidates
    logger.info(f"Reranking {len(candidates)} candidates...")
    reranked_response = rerank_results(
        request,
        RerankRequest(query=search_request.text, candidates=candidates),
        api_key
    )
    reranked_candidates = reranked_response.reranked_candidates
    
    # Limit the number of results after reranking
    reranked_candidates = reranked_candidates[:settings.RERANKER_MAX_CANDIDATES]
    
    # Convert to response format - only return items with positive similarity scores
    posts = []
    for candidate in reranked_candidates:
        similarity_score = candidate.get('reranker_score', candidate['similarity_score'])
        if similarity_score > 0:
            posts.append(SimilarDiscourseResponse(
                id=candidate['id'],
                title=candidate['title'],
                url=candidate['url'],
                similarity_score=similarity_score
            ))
    
    return posts

# --- POST endpoint at /v2/similar-questions ---
@app.post("/v2/similar-questions", response_model=List[SimilarQuestionResponse])
@limiter.limit("10/minute")
def find_similar_questions_v2(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> List[SimilarQuestionResponse]:
    """
    Find most similar questions based on a text query with reranking.
    The search is performed across question and answer embeddings using vector embeddings,
    then reranked using the reranker model for better relevance.
    """
    logger.info(f"🌐 POST /v2/similar-questions for text: '{search_request.text[:50]}...'")

    # Use reranker_service (the unified service); previous reference to reranker_client was incorrect
    if not reranker_service:
        logger.warning("Reranker not available, falling back to v1 endpoint")
        return find_similar_questions_v1(request, search_request, db, api_key)

    embedding = embedding_service.create_embedding(search_request.text)
    if embedding is None:
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")

    # Build the SQL query with CTEs
    embedding_str = ','.join(str(v) for v in embedding)
    embedding_sql = f"'[{embedding_str}]'::vector"
    sql = f"""
    WITH question_sim AS (
        SELECT q.id, q.question, q.answer, q.source_type, q.source_id, 1 - (q.question_embedding <=> {embedding_sql}) AS similarity
        FROM questions q
        WHERE q.question_embedding IS NOT NULL
        AND 1 - (q.question_embedding <=> {embedding_sql}) > 0.5
        ORDER BY q.question_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    answer_sim AS (
        SELECT q.id, q.question, q.answer, q.source_type, q.source_id, 1 - (q.answer_embedding <=> {embedding_sql}) AS similarity
        FROM questions q
        WHERE q.answer_embedding IS NOT NULL
        AND 1 - (q.answer_embedding <=> {embedding_sql}) > 0.5
        ORDER BY q.answer_embedding <=> {embedding_sql}
        LIMIT 50
    ),
    all_sim AS (
        SELECT * FROM question_sim
        UNION ALL
        SELECT * FROM answer_sim
    ),
    grouped_sim AS (
        SELECT id, question, answer, source_type, source_id, MAX(similarity) AS similarity
        FROM all_sim
        GROUP BY id, question, answer, source_type, source_id
    ),
    questions_with_urls AS (
        SELECT 
            g.id, 
            g.question, 
            g.answer, 
            g.similarity,
            CASE 
                WHEN g.source_type = 'ISSUE' THEN CONCAT('{settings.GITHUB_BASE_URL}/issues/', i.number)
                WHEN g.source_type = 'DISCOURSE_POST' THEN CONCAT('{settings.DISCOURSE_BASE_URL}/t/', d.slug, '/', d.topic_id)
                WHEN g.source_type = 'METABASE_DOC' THEN m.url
                ELSE NULL
            END AS url
        FROM grouped_sim g
        LEFT JOIN issues i ON g.source_type = 'ISSUE' AND g.source_id = i.id
        LEFT JOIN discourse_posts d ON g.source_type = 'DISCOURSE_POST' AND g.source_id = d.id
        LEFT JOIN metabase_docs m ON g.source_type = 'METABASE_DOC' AND g.source_id = m.id
    )
    SELECT id, question, answer, url, similarity
    FROM questions_with_urls
    WHERE url IS NOT NULL
    ORDER BY similarity DESC
    LIMIT 50;
    """

    logger.info("Executing questions similarity query for reranking...")
    
    result = db.execute(sql_text(sql))
    
    # Prepare candidates for reranking
    candidates = []
    for row in result:
        candidates.append({
            'id': row.id,
            'title': row.question,
            'content': row.answer,
            'source_type': 'question',
            'url': row.url,
            'similarity_score': float(row.similarity)
        })
    
    if not candidates:
        logger.info("No candidates found for reranking")
        return []
    
    # Rerank the candidates
    logger.info(f"Reranking {len(candidates)} candidates...")
    reranked_response = rerank_results(
        request,
        RerankRequest(query=search_request.text, candidates=candidates),
        api_key
    )
    reranked_candidates = reranked_response.reranked_candidates
    
    # Limit the number of results after reranking
    reranked_candidates = reranked_candidates[:settings.RERANKER_MAX_CANDIDATES]
    
    # Convert to response format - only return items with positive similarity scores
    questions = []
    for candidate in reranked_candidates:
        similarity_score = candidate.get('reranker_score', candidate['similarity_score'])
        if similarity_score > 0:
            questions.append(SimilarQuestionResponse(
                id=candidate['id'],
                question=candidate['title'],
                answer=candidate['content'],
                url=candidate['url'],
                similarity_score=similarity_score
            ))
    
    return questions



# --- POST endpoint at /v1/similar ---
@app.post("/v1/similar", response_model=V2SimilarResponse)
@limiter.limit("10/minute")
async def find_similar_v1(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> V2SimilarResponse:
    """
    Find similar content across all types (issues, discourse posts, documentation, questions).
    Returns issues, discourse posts, documentation pages, and questions that match the query.
    This endpoint calls all v1 endpoints in parallel for better performance.
    """
    logger.info(f"🌐 POST /v1/similar for text: '{search_request.text[:50]}...' state: {search_request.state}")

    # Call individual v1 endpoints in parallel using threads to avoid blocking the event loop
    async def call_issues():
        # Create a dedicated DB session for thread execution
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_github_issues_v1, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()
    
    async def call_discourse():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_discourse_posts_v1, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()
    
    async def call_docs():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_metabase_docs_v1, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()
    
    async def call_questions():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_questions_v1, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()

    # Execute all calls in parallel
    issues, discourse_posts, metabase_docs, questions = await asyncio.gather(
        call_issues(),
        call_discourse(),
        call_docs(),
        call_questions(),
        return_exceptions=True
    )
    
    # Handle any exceptions that occurred during parallel execution
    if isinstance(issues, Exception):
        logger.error(f"Error fetching similar issues: {issues}")
        issues = []
    if isinstance(discourse_posts, Exception):
        logger.error(f"Error fetching similar discourse posts: {discourse_posts}")
        discourse_posts = []
    if isinstance(metabase_docs, Exception):
        logger.error(f"Error fetching similar metabase docs: {metabase_docs}")
        metabase_docs = []
    if isinstance(questions, Exception):
        logger.error(f"Error fetching similar questions: {questions}")
        questions = []

    return V2SimilarResponse(
        issues=issues,
        discourse_posts=discourse_posts,
        metabase_docs=metabase_docs,
        questions=questions,
        keywords=[]  # Keywords not implemented in v1 endpoints yet
    )

# --- POST endpoint at /v2/similar ---
@app.post("/v2/similar", response_model=V2SimilarResponse)
@limiter.limit("10/minute")
async def find_similar_v2(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> V2SimilarResponse:
    """
    Find similar content across all types (issues, discourse posts, documentation, questions) with reranking.
    Returns issues, discourse posts, documentation pages, and questions that match the query.
    This endpoint calls all v2 endpoints in parallel with reranker functionality.
    """
    logger.info(f"🌐 POST /v2/similar for text: '{search_request.text[:50]}...' state: {search_request.state}")

    # Call individual v2 endpoints in parallel using threads to avoid blocking the event loop
    async def call_issues_v2():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_github_issues_v2, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()
    
    async def call_discourse_v2():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_discourse_posts_v2, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()
    
    async def call_docs_v2():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_metabase_docs_v2, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()
    
    async def call_questions_v2():
        thread_db = SessionLocal()
        try:
            return await asyncio.to_thread(find_similar_questions_v2, request, search_request, thread_db, api_key)
        finally:
            thread_db.close()

    # Execute all calls in parallel
    issues, discourse_posts, metabase_docs, questions = await asyncio.gather(
        call_issues_v2(),
        call_discourse_v2(),
        call_docs_v2(),
        call_questions_v2(),
        return_exceptions=True
    )
    
    # Handle any exceptions that occurred during parallel execution
    if isinstance(issues, Exception):
        logger.error(f"Error fetching similar issues v2: {issues}")
        issues = []
    if isinstance(discourse_posts, Exception):
        logger.error(f"Error fetching similar discourse posts v2: {discourse_posts}")
        discourse_posts = []
    if isinstance(metabase_docs, Exception):
        logger.error(f"Error fetching similar metabase docs v2: {metabase_docs}")
        metabase_docs = []
    if isinstance(questions, Exception):
        logger.error(f"Error fetching similar questions v2: {questions}")
        questions = []

    return V2SimilarResponse(
        issues=issues,
        discourse_posts=discourse_posts,
        metabase_docs=metabase_docs,
        questions=questions,
        keywords=[]  # Keywords not implemented in v2 endpoints yet
    )





# Custom rate limit error handler
from fastapi.responses import JSONResponse
from typing import Callable

def custom_rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    """Custom rate limit exceeded handler.""" 
    return JSONResponse(
        status_code=429,
        content={"detail": f"Rate limit exceeded: {exc}"}
    )

# Add rate limit error handler  
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)

# --- Keyword Management Endpoints ---

class KeywordRequest(BaseModel):
    """Request model for keyword operations."""
    keyword: str = Field(..., description="The keyword to define.")
    definition: str = Field(..., description="The definition of the keyword.")
    category: Optional[str] = Field(None, description="Optional category for organization.")

class KeywordResponse(BaseModel):
    """Response model for keyword operations."""
    success: bool = Field(..., description="Whether the operation was successful.")
    message: str = Field(..., description="Response message.")
    keyword: Optional[str] = Field(None, description="The keyword if operation was successful.")

class KeywordListResponse(BaseModel):
    """Response model for listing keywords."""
    keywords: List[Dict] = Field(..., description="List of keyword definitions.")

class KeywordToggleRequest(BaseModel):
    """Request model for toggling keyword status."""
    keyword: str = Field(..., description="The keyword to toggle.")

class SynonymRequest(BaseModel):
    """Request model for synonym operations."""
    word: str = Field(..., description="The synonym word.")
    synonym_of: str = Field(..., description="The keyword this word is a synonym for.")

class SynonymResponse(BaseModel):
    """Response model for synonym operations."""
    success: bool = Field(..., description="Whether the operation was successful.")
    message: str = Field(..., description="Response message.")
    word: Optional[str] = Field(None, description="The synonym word if operation was successful.")

from src.keyword_service import KeywordService

keyword_service = KeywordService()

@app.post("/keywords/add", response_model=KeywordResponse)
@limiter.limit("10/minute")
def add_keyword(
    request: Request,
    keyword_request: KeywordRequest,
    api_key: str = Security(get_api_key)
) -> KeywordResponse:
    """
    Add a new keyword definition to the database.
    """
    try:
        success = keyword_service.add_keyword_definition(
            keyword=keyword_request.keyword,
            definition=keyword_request.definition,
            category=keyword_request.category
        )
        
        if success:
            return KeywordResponse(
                success=True,
                message=f"Keyword '{keyword_request.keyword}' added successfully",
                keyword=keyword_request.keyword
            )
        else:
            return KeywordResponse(
                success=False,
                message=f"Keyword '{keyword_request.keyword}' already exists",
                keyword=None
            )
    except Exception as e:
        logger.error(f"Error adding keyword: {e}")
        return KeywordResponse(
            success=False,
            message=f"Error adding keyword: {str(e)}",
            keyword=None
        )

@app.put("/keywords/update", response_model=KeywordResponse)
@limiter.limit("10/minute")
def update_keyword(
    request: Request,
    keyword_request: KeywordRequest,
    api_key: str = Security(get_api_key)
) -> KeywordResponse:
    """
    Update an existing keyword definition.
    """
    try:
        success = keyword_service.update_keyword_definition(
            keyword=keyword_request.keyword,
            definition=keyword_request.definition,
            category=keyword_request.category
        )
        
        if success:
            return KeywordResponse(
                success=True,
                message=f"Keyword '{keyword_request.keyword}' updated successfully",
                keyword=keyword_request.keyword
            )
        else:
            return KeywordResponse(
                success=False,
                message=f"Keyword '{keyword_request.keyword}' not found",
                keyword=None
            )
    except Exception as e:
        logger.error(f"Error updating keyword: {e}")
        return KeywordResponse(
            success=False,
            message=f"Error updating keyword: {str(e)}",
            keyword=None
        )

@app.delete("/keywords/delete", response_model=KeywordResponse)
@limiter.limit("10/minute")
def delete_keyword(
    request: Request,
    keyword_request: KeywordRequest,
    api_key: str = Security(get_api_key)
) -> KeywordResponse:
    """
    Delete a keyword definition from the database.
    """
    try:
        success = keyword_service.delete_keyword_definition(
            keyword=keyword_request.keyword
        )
        
        if success:
            return KeywordResponse(
                success=True,
                message=f"Keyword '{keyword_request.keyword}' deleted successfully",
                keyword=keyword_request.keyword
            )
        else:
            return KeywordResponse(
                success=False,
                message=f"Keyword '{keyword_request.keyword}' not found",
                keyword=None
            )
    except Exception as e:
        logger.error(f"Error deleting keyword: {e}")
        return KeywordResponse(
            success=False,
            message=f"Error deleting keyword: {str(e)}",
            keyword=None
        )

@app.post("/keywords/toggle", response_model=KeywordResponse)
@limiter.limit("10/minute")
def toggle_keyword(
    request: Request,
    toggle_request: KeywordToggleRequest,
    api_key: str = Security(get_api_key)
) -> KeywordResponse:
    """
    Toggle the active status of a keyword definition.
    """
    try:
        success = keyword_service.toggle_keyword_status(
            keyword=toggle_request.keyword
        )
        
        if success:
            return KeywordResponse(
                success=True,
                message=f"Keyword '{toggle_request.keyword}' status toggled successfully",
                keyword=toggle_request.keyword
            )
        else:
            return KeywordResponse(
                success=False,
                message=f"Keyword '{toggle_request.keyword}' not found",
                keyword=None
            )
    except Exception as e:
        logger.error(f"Error toggling keyword: {e}")
        return KeywordResponse(
            success=False,
            message=f"Error toggling keyword: {str(e)}",
            keyword=None
        )

@app.get("/keywords/list", response_model=KeywordListResponse)
@limiter.limit("10/minute")
def list_keywords(
    request: Request,
    category: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> KeywordListResponse:
    """
    List all keyword definitions, optionally filtered by category.
    """
    try:
        keywords = keyword_service.list_keywords(db, category=category)
        return KeywordListResponse(keywords=keywords)
    except Exception as e:
        logger.error(f"Error listing keywords: {e}")
        return KeywordListResponse(keywords=[])

# --- Synonym Management Endpoints ---

@app.post("/synonyms/add", response_model=SynonymResponse)
@limiter.limit("10/minute")
def add_synonym(
    request: Request,
    synonym_request: SynonymRequest,
    api_key: str = Security(get_api_key)
) -> SynonymResponse:
    """
    Add a new synonym to the database.
    """
    db = None
    try:
        from src.models import Synonym
        db = SessionLocal()
        
        # Check if synonym already exists
        existing = db.query(Synonym).filter(
            Synonym.word == synonym_request.word,
            Synonym.synonym_of == synonym_request.synonym_of
        ).first()
        
        if existing:
            return SynonymResponse(
                success=False,
                message=f"Synonym '{synonym_request.word}' for '{synonym_request.synonym_of}' already exists",
                word=None
            )
        
        new_synonym = Synonym(
            word=synonym_request.word,
            synonym_of=synonym_request.synonym_of
        )
        
        db.add(new_synonym)
        db.commit()
        
        return SynonymResponse(
            success=True,
            message=f"Synonym '{synonym_request.word}' added successfully",
            word=synonym_request.word
        )
        
    except Exception as e:
        logger.error(f"Error adding synonym: {e}")
        return SynonymResponse(
            success=False,
            message=f"Error adding synonym: {str(e)}",
            word=None
        )
    finally:
        if db:
            db.close()

@app.delete("/synonyms/delete", response_model=SynonymResponse)
@limiter.limit("10/minute")
def delete_synonym(
    request: Request,
    synonym_request: SynonymRequest,
    api_key: str = Security(get_api_key)
) -> SynonymResponse:
    """
    Delete a synonym from the database.
    """
    db = None
    try:
        from src.models import Synonym
        db = SessionLocal()
        
        synonym = db.query(Synonym).filter(
            Synonym.word == synonym_request.word,
            Synonym.synonym_of == synonym_request.synonym_of
        ).first()
        
        if not synonym:
            return SynonymResponse(
                success=False,
                message=f"Synonym '{synonym_request.word}' for '{synonym_request.synonym_of}' not found",
                word=None
            )
        
        db.delete(synonym)
        db.commit()
        
        return SynonymResponse(
            success=True,
            message=f"Synonym '{synonym_request.word}' deleted successfully",
            word=synonym_request.word
        )
        
    except Exception as e:
        logger.error(f"Error deleting synonym: {e}")
        return SynonymResponse(
            success=False,
            message=f"Error deleting synonym: {str(e)}",
            word=None
        )
    finally:
        if db:
            db.close()

@app.get("/synonyms/list", response_model=Dict)
@limiter.limit("10/minute")
def list_synonyms(
    request: Request,
    keyword: Optional[str] = None,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> Dict:
    """
    List all synonyms, optionally filtered by keyword.
    """
    try:
        from src.models import Synonym
        
        query = db.query(Synonym)
        if keyword:
            query = query.filter(Synonym.synonym_of == keyword)
        
        synonyms = query.all()
        
        return {
            "synonyms": [
                {
                    "word": s.word,
                    "synonym_of": s.synonym_of,
                    "created_at": s.created_at.isoformat() if s.created_at is not None else None
                }
                for s in synonyms
            ],
            "count": len(synonyms)
        }
    except Exception as e:
        logger.error(f"Error listing synonyms: {e}")
        return {"synonyms": [], "count": 0, "error": str(e)}

def sanitize_user_input(user_input: str) -> str:
    """
    Enhanced sanitization based on OWASP recommendations.
    - Limit input length
    - Remove/escape dangerous patterns
    - HTML escape user content
    - Remove code injection attempts
    """
    # 1. Input Length Limiting (OWASP recommendation)
    max_length = 2000  # Reasonable limit for chat input
    if len(user_input) > max_length:
        logger.warning(f"Input truncated from {len(user_input)} to {max_length} characters")
        user_input = user_input[:max_length]

    # 2. HTML escape to prevent markup injection
    user_input = html.escape(user_input)

    # 3. Remove/replace suspicious prompt injection patterns
    suspicious_patterns = [
        # Direct instruction manipulation
        (r"(?i)ignore\s+(all\s+)?previous\s+(instructions?|prompts?)", "[FILTERED]"),
        (r"(?i)disregard\s+(all\s+)?(above|previous|prior)", "[FILTERED]"),
        (r"(?i)forget\s+(all\s+)?(previous|prior|above)", "[FILTERED]"),
        (r"(?i)reset\s+(your\s+)?instructions?", "[FILTERED]"),
        (r"(?i)change\s+(your\s+)?instructions?", "[FILTERED]"),
        
        # Role manipulation
        (r"(?i)(you\s+are\s+now|act\s+as|pretend\s+to\s+be)", "[FILTERED]"),
        (r"(?i)play\s+the\s+role\s+of", "[FILTERED]"),
        (r"(?i)simulate\s+(being|a)", "[FILTERED]"),
        
        # System/Assistant impersonation
        (r"(?i)(system|assistant|ai):\s*", "[FILTERED]: "),
        (r"(?i)\\n(system|assistant):", "\\n[FILTERED]:"),
        
        # Code injection attempts
        (r"```[\s\S]*?```", "[CODE_BLOCK_REMOVED]"),
        (r"<script[\s\S]*?</script>", "[SCRIPT_REMOVED]"),
        (r"javascript:", "[JS_REMOVED]"),
        
        # Prompt leakage attempts
        (r"(?i)(show|reveal|display|print)\s+(your\s+)?(prompt|instructions?|system\s+message)", "[FILTERED]"),
        (r"(?i)what\s+(are\s+)?your\s+(initial\s+)?instructions?", "[FILTERED]"),
        
        # Jailbreak attempts
        (r"(?i)jailbreak", "[FILTERED]"),
        (r"(?i)developer\s+mode", "[FILTERED]"),
        (r"(?i)god\s+mode", "[FILTERED]"),
        
        # Content policy bypass
        (r"(?i)ignore\s+(content\s+)?policy", "[FILTERED]"),
        (r"(?i)bypass\s+(safety\s+)?filter", "[FILTERED]"),
    ]
    
    for pattern, replacement in suspicious_patterns:
        if re.search(pattern, user_input):
            logger.warning(f"Suspicious pattern detected and filtered: {pattern}")
            user_input = re.sub(pattern, replacement, user_input)

    # 4. Remove excessive whitespace and formatting
    user_input = re.sub(r'\s+', ' ', user_input.strip())
    
    # 5. Remove potential delimiter injection
    user_input = re.sub(r'---+', '-', user_input)
    user_input = re.sub(r'===+', '=', user_input)
    
    return user_input

def validate_llm_output(output: str, user_input: str) -> tuple[bool, str]:
    """
    Enhanced output validation based on OWASP recommendations.
    Returns (is_safe, sanitized_output)
    """
    if not output or not isinstance(output, str):
        return False, "Invalid response generated."
    
    # Check for prompt injection signs in output
    suspicious_output_patterns = [
        r"(?i)as an ai language model",
        r"(?i)i'm (an ai|chatgpt|gpt|claude)",
        r"(?i)system prompt",
        r"(?i)ignore previous",
        r"(?i)my instructions (are|were)",
        r"(?i)i (was|am) told to",
        r"(?i)the user asked me to",
        r"(?i)jailbreak",
        r"(?i)developer mode",
        r"(?i)god mode",
        r"(?i)pretend to be",
        r"(?i)role.*play",
        r"(?i)as requested.*ignore",
        r"(?i)reset.*instructions",
        # Check if the model is repeating/leaking the system prompt
        r"(?i)metabase.*business intelligence.*analytics platform",
        r"(?i)context information.*keywords.*documentation",
    ]
    
    for pattern in suspicious_output_patterns:
        if re.search(pattern, output):
            logger.warning(f"Suspicious output pattern detected: {pattern}")
            return False, (
                "I apologize, but I cannot process that request. "
                "Please rephrase your question about Metabase in a different way."
            )
    
    # Check for potential data leakage (system instructions in output)
    if len(output) > 5000:  # Unusually long responses might contain leaked prompts
        logger.warning("Response unusually long, potential prompt leakage")
        return False, (
            "I apologize, but I cannot provide a response to that request. "
            "Please ask a more specific question about Metabase."
        )
    
    # Basic content validation - ensure it's related to Metabase
    metabase_keywords = ['metabase', 'dashboard', 'database', 'query', 'analytics', 'visualization', 'chart']
    if not any(keyword in output.lower() for keyword in metabase_keywords) and len(user_input) > 10:
        # If output doesn't seem related to Metabase and input was substantial
        logger.info("Output doesn't appear related to Metabase, flagging for review")
    
    return True, output

def log_security_event(event_type: str, user_input: str, details: str, chat_id: int):
    """Log security events for monitoring and analysis."""
    logger.warning(
        f"SECURITY_EVENT: {event_type} | "
        f"Chat ID: {chat_id} | "
        f"Input: {user_input[:100]}... | "
        f"Details: {details}"
    )

@app.post("/v2/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat_service_v2(
    request: Request,
    chat_request: ChatRequest,
    db: Session = Depends(get_db),
    api_key: str = Security(get_api_key)
) -> ChatResponse:
    """
    Enhanced chat service v2 with comprehensive prompt injection protection.
    Implements OWASP LLM security recommendations.
    """
    logger.info(f"🌐 POST /chat/v2 for text: '{chat_request.text[:50]}...' chat_id: {chat_request.chat_id}")
    
    # Create chat session record
    chat_session = ChatSession(
        chat_id=chat_request.chat_id,
        user_request=chat_request.text,
        sources=None,
        response=None
    )
    db.add(chat_session)
    db.commit()
    
    try:
        # Initialize prompt tracking variable
        full_prompt = f"Initial user request: {chat_request.text}"
        
        # Enhanced input validation and sanitization
        original_input = chat_request.text
        sanitized_input = sanitize_user_input(chat_request.text)
        
        # Log if significant sanitization occurred
        if len(original_input) != len(sanitized_input) or original_input != sanitized_input:
            log_security_event(
                "INPUT_SANITIZED", 
                original_input, 
                f"Sanitized from {len(original_input)} to {len(sanitized_input)} chars",
                chat_request.chat_id
            )
        
        # Input length validation
        if len(sanitized_input.strip()) < 3:
            raise HTTPException(
                status_code=400, 
                detail="Input too short. Please provide a meaningful question about Metabase."
            )
        
        # Step 1: Get relevant keywords for the context (direct service call)
        logger.info("🔍 Step 1: Getting relevant keywords...")
        try:
            relevant_keywords = keyword_service.get_relevant_keywords(sanitized_input, db)
        except Exception:
            logger.exception("Failed to fetch relevant keywords")
            relevant_keywords = []
        
        logger.info(f"📝 Found {len(relevant_keywords)} relevant keywords")
        
        # Step 2: Call /v2/similar endpoint to get documentation and questions/answers
        logger.info("🔍 Step 2: Getting similar content from v2/similar endpoint...")
        similar_response = await find_similar_v2(
            request, 
            SearchRequest(text=sanitized_input, state=None), 
            db, 
            api_key
        )
        
        logger.info(f"📊 Found {len(similar_response.metabase_docs)} docs, {len(similar_response.questions)} questions")
        
        # Step 3: Fetch detailed content from database (only documentation and questions/answers)
        detailed_data = []
        sources = []
        
        # Bulk fetch metabase doc details
        doc_ids = [doc.id for doc in similar_response.metabase_docs if doc and hasattr(doc, 'id') and doc.id is not None]
        if doc_ids:
            db_docs = db.query(MetabaseDoc).filter(MetabaseDoc.id.in_(doc_ids)).all()
            doc_dict = {doc.id: doc for doc in db_docs}
            
            for doc in similar_response.metabase_docs:
                if doc and hasattr(doc, 'id') and doc.id is not None and doc.id in doc_dict:
                    db_doc = doc_dict[doc.id]
                    detailed_data.append({
                        "type": "metabase_doc",
                        "url": db_doc.url,
                        "markdown": db_doc.markdown
                    })
                    sources.append(doc.url)
                    
                    # Track entity injection
                    entity = ChatSessionEntity(
                        chat_id=chat_session.id,
                        entity_type="metabase_doc",
                        entity_id=doc.id,
                        entity_url=doc.url,
                        similarity_score=doc.similarity_score
                    )
                    db.add(entity)
        
        # Bulk fetch question/answer details
        qa_ids = [qa.id for qa in similar_response.questions if qa and hasattr(qa, 'id') and qa.id is not None]
        if qa_ids:
            db_qas = db.query(Question).filter(Question.id.in_(qa_ids)).all()
            qa_dict = {qa.id: qa for qa in db_qas}
            
            for qa in similar_response.questions:
                if qa and hasattr(qa, 'id') and qa.id is not None and qa.id in qa_dict:
                    db_qa = qa_dict[qa.id]
                    detailed_data.append({
                        "type": "question_answer",
                        "question": db_qa.question,
                        "answer": db_qa.answer,
                        "url": qa.url
                    })
                    sources.append(qa.url)
                    
                    # Track entity injection
                    entity = ChatSessionEntity(
                        chat_id=chat_session.id,
                        entity_type="question_answer",
                        entity_id=qa.id,
                        entity_url=qa.url,
                        similarity_score=qa.similarity_score
                    )
                    db.add(entity)
        
        # Update chat session with sources (convert to JSON serializable format)
        import json
        serializable_sources = []
        for source in sources:
            if hasattr(source, '__dict__'):
                serializable_sources.append(source.__dict__)
            else:
                serializable_sources.append(str(source))
        
        # Update the sources column properly for SQLAlchemy
        db.execute(
            sql_text("UPDATE chat_sessions SET sources = :sources WHERE id = :id"),
            {"sources": json.dumps(serializable_sources) if serializable_sources else None, "id": chat_session.id}
        )
        db.commit()
        
        # Step 4: Build context with keywords and content
        context_parts = []
        
        # Add relevant keywords
        if relevant_keywords:
            keyword_info = "Relevant Keywords:\n"
            for keyword in relevant_keywords:
                keyword_info += f"- {keyword['keyword']}: {keyword['definition']}\n"
                # Track keyword entity injection
                from src.models import KeywordDefinition
                db_keyword = db.query(KeywordDefinition).filter(
                    KeywordDefinition.keyword == keyword['keyword']
                ).first()
                
                if db_keyword:
                    entity = ChatSessionEntity(
                        chat_id=chat_session.id,
                        entity_type="keyword",
                        entity_id=db_keyword.id,
                        entity_url=None,
                        similarity_score=None  # NULL for keywords since they don't have similarity scores
                    )
                    db.add(entity)
                    logger.info(f"Added keyword entity: {keyword['keyword']} with actual ID: {db_keyword.id}")
                else:
                    logger.warning(f"Keyword '{keyword['keyword']}' not found in database, skipping entity tracking")
            context_parts.append(keyword_info)
        
        # Add content from database (only documentation and questions/answers)
        for item in detailed_data:
            if item["type"] == "metabase_doc":
                context_parts.append(f"Documentation: {item['markdown']}\nURL: {item['url']}")
            elif item["type"] == "question_answer":
                context_parts.append(f"Q&A: {item['question']}\nAnswer: {item['answer']}\nURL: {item['url']}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Step 5: Enhanced LLM interaction with strict separation
        logger.info("🤖 Step 5: Generating final answer with enhanced security...")
        
        # Strict message construction - complete separation of user input and system instructions
        system_prompt = get_api_chat_system_prompt()
        context_prompt = get_api_context_prompt(context)
        
        # Construct messages with strict separation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": context_prompt},
            {"role": "user", "content": sanitized_input}
        ]

        # Capture the full prompt for storage
        full_prompt = "\n\n".join([
            f"[SYSTEM]: {system_prompt}",
            f"[CONTEXT]: {context_prompt}",
            f"[USER]: {sanitized_input}"
        ])

        # Call LLM with enhanced error handling and token tracking
        try:
            llm_response = llm_client.call_llm_with_usage(
                messages=messages,
                model="openai-slow",
                max_retries=3
            )
            
            # Print relevant response information for troubleshooting
            logger.info(f"🔍 LLM Response Type: {llm_response.get('response_type', 'N/A') if llm_response and isinstance(llm_response, dict) else 'N/A'}")
            logger.info(f"🔍 LLM Response ID: {llm_response.get('response_id', 'N/A') if llm_response and isinstance(llm_response, dict) else 'N/A'}")
            logger.info(f"🔍 LLM Response Model: {llm_response.get('response_model', 'N/A') if llm_response and isinstance(llm_response, dict) else 'N/A'}")
            logger.info(f"🔍 LLM Response Cache Hit: {llm_response.get('cache_hit', 'N/A') if llm_response and isinstance(llm_response, dict) else 'N/A'}")
            logger.info(f"🔍 LLM Response Tokens Sent: {llm_response.get('tokens_sent', 'N/A') if llm_response and isinstance(llm_response, dict) else 'N/A'}")
            logger.info(f"🔍 LLM Response Tokens Received: {llm_response.get('tokens_received', 'N/A') if llm_response and isinstance(llm_response, dict) else 'N/A'}")
            
        except Exception as llm_error:
            logger.error(f"LLM call failed: {llm_error}")
            log_security_event("LLM_ERROR", sanitized_input, str(llm_error), chat_request.chat_id)
            raise HTTPException(status_code=500, detail="Unable to generate response. Please try again.")

        if not llm_response or not llm_response.get('content'):
            raise HTTPException(status_code=500, detail="Failed to generate final answer")

        final_answer = llm_response['content']
        tokens_sent = llm_response.get('tokens_sent', 0)
        tokens_received = llm_response.get('tokens_received', 0)
        cache_hit = llm_response.get('cache_hit', False)
        
        # If cache hit, set tokens_received to 0 as no new tokens were generated
        if cache_hit:
            tokens_received = 0
            logger.debug(f"Cache hit detected - recording tokens_sent: {tokens_sent}, tokens_received: 0")

        # Enhanced output validation
        is_safe, validated_answer = validate_llm_output(final_answer, sanitized_input)
        
        if not is_safe:
            log_security_event("UNSAFE_OUTPUT", sanitized_input, final_answer[:200], chat_request.chat_id)
            final_answer = validated_answer

        # Update chat session with final response, prompt, token usage, and cache hit status
        db.execute(
            sql_text("UPDATE chat_sessions SET response = :response, prompt = :prompt, tokens_sent = :tokens_sent, tokens_received = :tokens_received, cache_hit = :cache_hit WHERE id = :id"),
            {
                "response": final_answer, 
                "prompt": full_prompt, 
                "tokens_sent": tokens_sent,
                "tokens_received": tokens_received,
                "cache_hit": cache_hit,
                "id": chat_session.id
            }
        )
        db.commit()
        
        logger.info(f"✅ Chat v2 response generated successfully")
        
        return ChatResponse(
            answer=final_answer,
            sources=sources
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Error in chat v2 endpoint: {e}")
        log_security_event("SYSTEM_ERROR", chat_request.text, str(e), chat_request.chat_id)
        
        # Rollback any pending changes and update chat session with error information
        db.rollback()
        # Use the full_prompt if it was created, otherwise use a fallback
        error_prompt = locals().get('full_prompt', f"Error occurred before prompt construction: {chat_request.text}")
        
        db.execute(
            sql_text("UPDATE chat_sessions SET response = :response, prompt = :prompt, tokens_sent = :tokens_sent, tokens_received = :tokens_received, cache_hit = :cache_hit WHERE id = :id"),
            {
                "response": f"Error: {str(e)}", 
                "prompt": error_prompt, 
                "tokens_sent": 0,
                "tokens_received": 0,
                "cache_hit": False,
                "id": chat_session.id
            }
        )
        db.commit()
        raise HTTPException(status_code=500, detail="An error occurred processing your request. Please try again.")

# Server startup code
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")