from fastapi import Security, HTTPException, Depends
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session
from starlette import status

from .db import get_db
from .models import ApiKey

api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(
    db: Session = Depends(get_db),
    api_key: str = Security(api_key_header)
) -> str:
    """
    Dependency to validate the API key.
    """
    db_api_key = db.query(ApiKey).filter(ApiKey.key == api_key).first()
    if not db_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key 