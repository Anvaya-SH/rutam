"""Structured provenance schema and authorization tracking for memory validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rutam.config import nonempty_string
from rutam.exceptions import ValidationError
from rutam.graph.schema import timestamp, utc_now


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """Structured provenance schema recording origin metadata and verification evidence.

    Invariants: Models cannot fabricate authorized sources without explicit credentials.
    """

    source_id: str
    source_category: str = "neural_write_head"
    observation_timestamp: datetime = field(default_factory=utc_now)
    creation_timestamp: datetime = field(default_factory=utc_now)
    content_hash: str | None = None
    evidence: str | None = None
    authorization_category: str = "unverified_model_hypothesis"
    originating_proposal_id: str | None = None
    attestation: str | None = None

    def __post_init__(self) -> None:
        nonempty_string(self.source_id, "source_id")
        nonempty_string(self.source_category, "source_category")
        timestamp(self.observation_timestamp, "observation_timestamp")
        timestamp(self.creation_timestamp, "creation_timestamp")
        nonempty_string(self.authorization_category, "authorization_category")
        if self.content_hash is not None:
            nonempty_string(self.content_hash, "content_hash")
        if self.evidence is not None and not isinstance(self.evidence, str):
            raise ValidationError("evidence must be a string or None")
        if self.originating_proposal_id is not None:
            nonempty_string(self.originating_proposal_id, "originating_proposal_id")
        if self.attestation is not None:
            nonempty_string(self.attestation, "attestation")

    @classmethod
    def compute_content_hash(cls, content: str) -> str:
        """Calculate SHA-256 hash of normalized text content."""
        if not isinstance(content, str):
            raise ValidationError("Content for hashing must be a string")
        return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_category": self.source_category,
            "observation_timestamp": self.observation_timestamp.isoformat(),
            "creation_timestamp": self.creation_timestamp.isoformat(),
            "content_hash": self.content_hash,
            "evidence": self.evidence,
            "authorization_category": self.authorization_category,
            "originating_proposal_id": self.originating_proposal_id,
            "attestation": self.attestation,
        }

    def attestation_payload(self) -> bytes:
        """Canonical bytes a trusted issuer signs after verifying identity/evidence."""
        payload = self.to_dict()
        payload.pop("attestation")
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")

    def fingerprint(self) -> str:
        """Return an unkeyed integrity digest, never an authorization proof."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRecord:
        if not isinstance(data, dict):
            raise ValidationError("Provenance data must be a dictionary")
        try:
            obs = (
                datetime.fromisoformat(data["observation_timestamp"])
                if "observation_timestamp" in data
                else utc_now()
            )
            cre = (
                datetime.fromisoformat(data["creation_timestamp"])
                if "creation_timestamp" in data
                else utc_now()
            )
            return cls(
                source_id=data["source_id"],
                source_category=data.get("source_category", "neural_write_head"),
                observation_timestamp=obs,
                creation_timestamp=cre,
                content_hash=data.get("content_hash"),
                evidence=data.get("evidence"),
                authorization_category=data.get(
                    "authorization_category", "unverified_model_hypothesis"
                ),
                originating_proposal_id=data.get("originating_proposal_id"),
                attestation=data.get("attestation"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError(f"Invalid provenance dictionary: {exc}") from exc


@dataclass(frozen=True, slots=True)
class AuthorizationContext:
    """Policy data from the host's authenticated request context.

    This dataclass is not itself an authentication credential. Applications must
    build it from their authenticated principal, never from request/proposal data.
    """

    user_id: str | None = None
    role: str = "user"
    authorized_categories: tuple[str, ...] = (
        "user_input",
        "external_document",
        "user_authorized",
    )
    allow_sensitive_data: bool = False
    allow_unverified_hypotheses: bool = False
    permissions: frozenset[str] = frozenset({"read", "propose", "validate"})
    authorized_scopes: tuple[str, ...] = ("default",)

    def __post_init__(self) -> None:
        if self.user_id is not None:
            nonempty_string(self.user_id, "user_id")
        nonempty_string(self.role, "role")
        if not isinstance(self.authorized_categories, (tuple, list)):
            raise ValidationError("authorized_categories must be a sequence of strings")
        for cat in self.authorized_categories:
            nonempty_string(cat, "authorized_category")
        object.__setattr__(self, "authorized_categories", tuple(self.authorized_categories))
        if not isinstance(self.authorized_scopes, (tuple, list)):
            raise ValidationError("authorized_scopes must be a sequence of strings")
        for scope in self.authorized_scopes:
            nonempty_string(scope, "authorized_scope")
        object.__setattr__(self, "authorized_scopes", tuple(self.authorized_scopes))
        if not isinstance(self.permissions, (set, frozenset, tuple, list)):
            raise ValidationError("permissions must be a sequence or set of strings")
        for permission in self.permissions:
            nonempty_string(permission, "permission")
        object.__setattr__(self, "permissions", frozenset(self.permissions))


ProvenanceVerifier = Callable[[ProvenanceRecord, AuthorizationContext], bool]


def validate_provenance_record(
    prov: Any,
    auth_context: AuthorizationContext | None = None,
    *,
    require_provenance: bool = True,
    allow_unverified_hypotheses: bool = False,
    provenance_verifier: ProvenanceVerifier | None = None,
) -> tuple[bool, str | None, ProvenanceRecord | None]:
    """Validate a provenance record against requirements and authorization boundaries."""
    if prov is None or prov == "":
        if require_provenance:
            return False, "MISSING_PROVENANCE", None
        return True, None, None

    if isinstance(prov, str):
        record = ProvenanceRecord(source_id=prov)
    elif isinstance(prov, dict):
        try:
            record = ProvenanceRecord.from_dict(prov)
        except ValidationError:
            return False, "INVALID_PROVENANCE_SCHEMA", None
    elif isinstance(prov, ProvenanceRecord):
        record = prov
    else:
        return False, "UNSUPPORTED_PROVENANCE_TYPE", None

    # Check unverified hypotheses
    allow_unverified = allow_unverified_hypotheses or (
        auth_context.allow_unverified_hypotheses if auth_context else False
    )
    if record.authorization_category == "unverified_model_hypothesis" and not allow_unverified:
        return False, "UNVERIFIED_HYPOTHESIS_DISALLOWED", record

    # Authorization and evidence claims are caller-controlled proposal fields.
    # Policy values and unkeyed hashes are not proof; a verifier must check a
    # signed attestation using keys/data held outside the proposal and context.
    if auth_context is not None:
        if (
            record.authorization_category == "user_authorized"
            and (auth_context.user_id is None or record.source_id != auth_context.user_id)
        ):
            return False, "PROVENANCE_SOURCE_NOT_AUTHORIZED", record
        if (
            record.authorization_category == "external_document"
            and (record.evidence is None or not record.evidence.strip())
        ):
            return False, "MISSING_PROVENANCE_EVIDENCE", record
        if record.authorization_category in {"user_authorized", "external_document"}:
            if provenance_verifier is None:
                return False, "PROVENANCE_VERIFIER_UNAVAILABLE", record
            try:
                verified = provenance_verifier(record, auth_context)
            except Exception:
                return False, "PROVENANCE_VERIFICATION_FAILED", record
            if verified is not True:
                return False, "PROVENANCE_ATTESTATION_INVALID", record
        if (
            record.authorization_category != "unverified_model_hypothesis"
            and record.authorization_category not in auth_context.authorized_categories
            and auth_context.role != "admin"
        ):
            return False, "UNAUTHORIZED_SOURCE_CATEGORY", record

    else:
        if record.authorization_category in {"user_authorized", "external_document"}:
            return False, "PROVENANCE_AUTH_CONTEXT_REQUIRED", record
        # An absent context conveys no elevated categories. Keep ordinary user
        # input available to offline validation, but reject invented authority
        # labels such as "system" instead of silently treating them as trusted.
        if (
            record.authorization_category != "unverified_model_hypothesis"
            and record.authorization_category != "user_input"
        ):
            return False, "UNAUTHORIZED_SOURCE_CATEGORY", record

    return True, None, record
