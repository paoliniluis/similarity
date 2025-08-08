from typing import List, Dict, Any, Optional

def build_keyword_context(keywords: List[Dict[str, str]]) -> str:
    """Build a consistent keyword context section for prompts."""
    if not keywords:
        return ""

    keywords_by_category: Dict[str, List[Dict[str, str]]] = {}
    for kw in keywords:
        category = kw.get("category") or "General"
        keywords_by_category.setdefault(category, []).append(kw)

    lines: List[str] = []
    lines.append("\n\nRELEVANT SPECIALIZED TERMINOLOGY:")
    lines.append("The following terms are relevant to this content:\n")
    for category, category_keywords in keywords_by_category.items():
        for kw in category_keywords:
            lines.append(f"• {kw['keyword']}: {kw['definition']}")
        lines.append("")
    lines.append("Please consider these definitions when generating your response.")
    return "\n".join(lines)

def clean_llm_json_response(raw: str) -> str:
    """Strip common wrappers like code fences and labels from LLM JSON outputs."""
    cleaned = (raw or "").strip()
    prefixes = [
        "```json", "```", "Here is the JSON:", "Here's the JSON response:", "JSON:", "Response:",
    ]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned

"""
Utility functions for the application.
"""

import torch
from typing import Literal

DeviceType = Literal["cuda", "cpu"]

def get_device() -> DeviceType:
    """
    Get the best available device for model inference.
    
    Returns:
        DeviceType: "cuda" if available, otherwise "cpu"
    """
    return "cuda" if torch.cuda.is_available() else "cpu" 