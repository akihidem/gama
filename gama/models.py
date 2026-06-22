"""Core enums for gama (stdlib-only)."""
from __future__ import annotations

from enum import Enum


class ModelTier(str, Enum):
    """Coarse capability/cost tier a backend maps to a concrete model."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

    @property
    def rank(self) -> int:
        return {"small": 0, "medium": 1, "large": 2}[self.value]


class TaskType(str, Enum):
    """What kind of work a task is — used as routing-table keys / benchmark classes."""

    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    DOC_FORMATTING = "doc_formatting"
    SUMMARIZATION = "summarization"
    RESEARCH = "research"
    SPEC_DESIGN = "spec_design"
    ARCHITECTURE = "architecture"
    CODE_IMPLEMENTATION = "code_implementation"
    CODE_REVIEW = "code_review"
    TEST_AUTHORING = "test_authoring"
    QA = "qa"
    SECURITY_REVIEW = "security_review"
    RELEASE = "release"
    INCIDENT_RESPONSE = "incident_response"
    CONTENT = "content"
    INTEGRATION = "integration"
    GENERIC = "generic"
