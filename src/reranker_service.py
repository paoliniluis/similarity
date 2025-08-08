import logging
from typing import List, Dict, Any, Optional, Protocol

import requests

from src import settings
from src.reranker_client import RerankerClient


logger = logging.getLogger(__name__)


class RerankerProvider(Protocol):
    def rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...


class LocalRerankerProvider:
    """Local reranker using a CrossEncoder model via `RerankerClient`."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.client = RerankerClient(
            model_name=model_name or settings.RERANKER_MODEL,
            device=device or settings.RERANKER_DEVICE,
        )
        logger.info("Initialized local reranker provider")

    def rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.client.rerank_results(query=query, candidates=candidates)


class APIRerankerProvider:
    """API-based reranker that forwards requests to an external rerank HTTP endpoint."""

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rerank_path: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        self.api_base_url = (api_base_url or settings.RERANKER_API_BASE).rstrip("/")
        path = rerank_path or settings.RERANKER_API_RERANK_PATH
        self.rerank_path = path if path.startswith("/") else f"/{path}"
        self.api_key = api_key or settings.RERANKER_API_KEY or settings.API_KEY
        self.timeout_seconds = timeout_seconds or getattr(settings, "RERANKER_API_TIMEOUT", 30)

        self.session = requests.Session()
        if self.api_key:
            self.session.headers["X-API-Key"] = str(self.api_key)
        self.session.headers["Content-Type"] = "application/json"

        logger.info(
            f"Initialized API reranker provider with base URL: {self.api_base_url}, path: {self.rerank_path}"
        )

    def rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            url = f"{self.api_base_url}{self.rerank_path}"
            payload = {"query": query, "candidates": candidates}
            response = self.session.post(url, json=payload, timeout=self.timeout_seconds)

            if response.status_code == 200:
                data = response.json()
                reranked = data.get("reranked_candidates")
                if isinstance(reranked, list):
                    return reranked
                logger.error("Rerank API returned invalid format: missing 'reranked_candidates'")
                return candidates
            else:
                logger.error(
                    f"Rerank API request failed with status {response.status_code}: {response.text}"
                )
                return candidates
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling external rerank API: {e}")
            return candidates


class RerankerService:
    """Facade over different reranker providers (local or API)."""

    def __init__(self, provider: RerankerProvider):
        self.provider = provider

    def rerank_results(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.provider.rerank_results(query, candidates)


_reranker_service: Optional[RerankerService] = None


def get_reranker_service() -> Optional[RerankerService]:
    global _reranker_service
    if not settings.RERANKER_ENABLED:
        logger.info("Reranker is disabled via settings")
        return None

    if _reranker_service is None:
        provider_name = str(getattr(settings, "RERANKER_PROVIDER", "local")).lower()
        if provider_name == "api":
            provider: RerankerProvider = APIRerankerProvider(
                api_base_url=getattr(settings, "RERANKER_API_BASE", "http://localhost:8000"),
                api_key=getattr(settings, "RERANKER_API_KEY", None),
                rerank_path=getattr(settings, "RERANKER_API_RERANK_PATH", "/rerank"),
                timeout_seconds=getattr(settings, "RERANKER_API_TIMEOUT", 30),
            )
        else:
            provider = LocalRerankerProvider(
                model_name=getattr(settings, "RERANKER_MODEL", None),
                device=getattr(settings, "RERANKER_DEVICE", None),
            )
        _reranker_service = RerankerService(provider)

    return _reranker_service


def set_reranker_service(service: Optional[RerankerService]):
    global _reranker_service
    _reranker_service = service


