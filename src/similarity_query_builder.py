"""
Shared query builder for similarity searches to eliminate code duplication.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session
from .constants import DEFAULT_SIMILARITY_LIMIT, DEFAULT_CANDIDATE_LIMIT


class SimilarityQueryBuilder:
    """Centralized query builder for vector similarity searches."""
    
    def __init__(self):
        self.default_limit = DEFAULT_SIMILARITY_LIMIT
        self.default_candidates = DEFAULT_CANDIDATE_LIMIT
    
    def build_similarity_query(
        self, 
        table_name: str, 
        embedding_param_name: str, 
        columns: Dict[str, str],
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Build a CTE-based similarity query for any table with embeddings.
        
        Args:
            table_name: Name of the database table
            embedding_param_name: Parameter name for the embedding vector (e.g., 'embedding_vector')
            columns: Dict mapping column purposes to actual column names
                    e.g., {'id': 'id', 'content_embedding': 'markdown_embedding'}
            where_clause: Optional WHERE clause for filtering
            limit: Final result limit (defaults to self.default_limit)
            
        Returns:
            SQL query string with parameterized embedding
        """
        if limit is None:
            limit = self.default_limit
            
        # Build CTEs for different embedding types
        ctes = []
        union_parts = []
        
        # Get required columns
        id_col = columns.get('id', 'id')
        select_cols = columns.get('select_cols', '*')
        
        # Content embedding CTE
        if 'content_embedding' in columns:
            content_col = columns['content_embedding']
            cte_name = f"{table_name}_content_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({content_col} <=> :{embedding_param_name}) AS similarity
        FROM {table_name}
        WHERE {content_col} IS NOT NULL
        ORDER BY {content_col} <=> :{embedding_param_name}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Summary embedding CTE
        if 'summary_embedding' in columns:
            summary_col = columns['summary_embedding']
            cte_name = f"{table_name}_summary_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({summary_col} <=> :{embedding_param_name}) AS similarity
        FROM {table_name}
        WHERE {summary_col} IS NOT NULL
        ORDER BY {summary_col} <=> :{embedding_param_name}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Title embedding CTE (for issues)
        if 'title_embedding' in columns:
            title_col = columns['title_embedding']
            cte_name = f"{table_name}_title_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({title_col} <=> :{embedding_param_name}) AS similarity
        FROM {table_name}
        WHERE {title_col} IS NOT NULL
        ORDER BY {title_col} <=> :{embedding_param_name}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Issue embedding CTE (for issues)
        if 'issue_embedding' in columns:
            issue_col = columns['issue_embedding']
            cte_name = f"{table_name}_issue_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({issue_col} <=> :{embedding_param_name}) AS similarity
        FROM {table_name}
        WHERE {issue_col} IS NOT NULL
        ORDER BY {issue_col} <=> :{embedding_param_name}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Question/Answer embeddings CTE
        if 'question_embedding' in columns:
            question_col = columns['question_embedding']
            answer_col = columns.get('answer_embedding', question_col)
            cte_name = f"{table_name}_question_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({question_col} <=> :{embedding_param_name}) AS similarity
        FROM {table_name}
        WHERE {question_col} IS NOT NULL
        ORDER BY {question_col} <=> :{embedding_param_name}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
            
            if answer_col != question_col:
                answer_cte_name = f"{table_name}_answer_sim"
                ctes.append(f"""
    {answer_cte_name} AS (
        SELECT {select_cols}, 1 - ({answer_col} <=> :{embedding_param_name}) AS similarity
        FROM {table_name}
        WHERE {answer_col} IS NOT NULL
        ORDER BY {answer_col} <=> :{embedding_param_name}
        LIMIT {self.default_candidates}
    )""")
                union_parts.append(f"SELECT * FROM {answer_cte_name}")
        
        # Build the full query
        group_by_cols = columns.get('group_by', id_col)
        
        all_sim_cte = f"""
    all_sim AS (
        {' UNION ALL '.join(union_parts)}
    )"""
        
        # Add where clause if provided
        where_part = f"WHERE {where_clause}" if where_clause else ""
        
        query = f"""
    WITH{','.join(ctes)},
{all_sim_cte}
    SELECT {group_by_cols}, MAX(similarity) AS similarity
    FROM all_sim
    {where_part}
    GROUP BY {group_by_cols}
    ORDER BY similarity DESC
    LIMIT {limit};
    """
        
        return query
    
    def execute_similarity_query(
        self, 
        db: Session, 
        table_name: str, 
        embedding: List[float], 
        columns: Dict[str, str],
        where_clause: Optional[str] = None,
        where_params: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ):
        """
        Execute a similarity query and return results.
        
        Args:
            db: Database session
            table_name: Name of the database table
            embedding: The embedding vector to search against
            columns: Column configuration
            where_clause: Optional WHERE clause
            where_params: Parameters for WHERE clause
            limit: Final result limit
            
        Returns:
            Query result rows
        """
        # For now, build the embedding directly into the SQL until we can properly 
        # handle vector parameters in SQLAlchemy
        embedding_str = ','.join(str(v) for v in embedding)
        embedding_sql = f"'[{embedding_str}]'::vector"
        
        # Use a temporary approach - build query with direct embedding
        query = self._build_similarity_query_with_embedding(
            table_name, embedding_sql, columns, where_clause, limit
        )
        
        params = where_params or {}
        return db.execute(sql_text(query), params)
    
    def _build_similarity_query_with_embedding(
        self, 
        table_name: str, 
        embedding_sql: str, 
        columns: Dict[str, str],
        where_clause: Optional[str] = None,
        limit: Optional[int] = None
    ) -> str:
        """
        Build a CTE-based similarity query with direct embedding SQL.
        This is a temporary method until proper vector parameter support is added.
        """
        if limit is None:
            limit = self.default_limit
            
        # Build CTEs for different embedding types
        ctes = []
        union_parts = []
        
        # Get required columns
        id_col = columns.get('id', 'id')
        select_cols = columns.get('select_cols', '*')
        
        # Content embedding CTE
        if 'content_embedding' in columns:
            content_col = columns['content_embedding']
            cte_name = f"{table_name}_content_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({content_col} <=> {embedding_sql}) AS similarity
        FROM {table_name}
        WHERE {content_col} IS NOT NULL
        ORDER BY {content_col} <=> {embedding_sql}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Summary embedding CTE
        if 'summary_embedding' in columns:
            summary_col = columns['summary_embedding']
            cte_name = f"{table_name}_summary_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({summary_col} <=> {embedding_sql}) AS similarity
        FROM {table_name}
        WHERE {summary_col} IS NOT NULL
        ORDER BY {summary_col} <=> {embedding_sql}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Title embedding CTE (for issues)
        if 'title_embedding' in columns:
            title_col = columns['title_embedding']
            cte_name = f"{table_name}_title_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({title_col} <=> {embedding_sql}) AS similarity
        FROM {table_name}
        WHERE {title_col} IS NOT NULL
        ORDER BY {title_col} <=> {embedding_sql}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Issue embedding CTE (for issues)
        if 'issue_embedding' in columns:
            issue_col = columns['issue_embedding']
            cte_name = f"{table_name}_issue_sim"
            ctes.append(f"""
    {cte_name} AS (
        SELECT {select_cols}, 1 - ({issue_col} <=> {embedding_sql}) AS similarity
        FROM {table_name}
        WHERE {issue_col} IS NOT NULL
        ORDER BY {issue_col} <=> {embedding_sql}
        LIMIT {self.default_candidates}
    )""")
            union_parts.append(f"SELECT * FROM {cte_name}")
        
        # Build the full query
        group_by_cols = columns.get('group_by', id_col)
        
        all_sim_cte = f"""
    all_sim AS (
        {' UNION ALL '.join(union_parts)}
    )"""
        
        # Add where clause if provided
        where_part = f"WHERE {where_clause}" if where_clause else ""
        
        query = f"""
    WITH{','.join(ctes)},
{all_sim_cte}
    SELECT {group_by_cols}, MAX(similarity) AS similarity
    FROM all_sim
    {where_part}
    GROUP BY {group_by_cols}
    ORDER BY similarity DESC
    LIMIT {limit};
    """
        
        return query
