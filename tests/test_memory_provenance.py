"""Tests for memory provenance schema, content hashing, and authorization validation."""

from __future__ import annotations

import hashlib
from datetime import datetime

import pytest

from rutam.exceptions import ValidationError
from rutam.memory.provenance import (
    AuthorizationContext,
    ProvenanceRecord,
    validate_provenance_record,
)


def test_provenance_record_creation_and_hash() -> None:
    content = "RUTAM native memory validation"
    expected_hash = hashlib.sha256(content.strip().encode("utf-8")).hexdigest()
    computed = ProvenanceRecord.compute_content_hash(content)
    assert computed == expected_hash

    rec = ProvenanceRecord(
        source_id="doc_42",
        source_category="external_document",
        content_hash=computed,
        evidence="paragraph 3",
        authorization_category="user_authorized",
    )
    assert rec.source_id == "doc_42"
    assert rec.content_hash == expected_hash
    assert rec.evidence == "paragraph 3"
    assert rec.authorization_category == "user_authorized"

    # Roundtrip from_dict / to_dict
    d = rec.to_dict()
    restored = ProvenanceRecord.from_dict(d)
    assert restored == rec

    with pytest.raises(ValidationError):
        ProvenanceRecord.compute_content_hash(123)  # type: ignore[arg-type]


def test_authorization_context() -> None:
    auth = AuthorizationContext(user_id="user_1", role="user")
    assert auth.user_id == "user_1"
    assert "user_input" in auth.authorized_categories
    assert "validate" in auth.permissions
    assert not auth.allow_sensitive_data
    assert not auth.allow_unverified_hypotheses

    with pytest.raises(ValidationError):
        AuthorizationContext(role="")


def test_validate_provenance_record_scenarios() -> None:
    # 1. Missing provenance when required
    valid, reason, _ = validate_provenance_record(None, require_provenance=True)
    assert not valid
    assert reason == "MISSING_PROVENANCE"

    # Missing provenance when not required
    valid, reason, _ = validate_provenance_record(None, require_provenance=False)
    assert valid
    assert reason is None

    # 2. String source_id coercion
    valid, reason, rec = validate_provenance_record("source_doc_99", require_provenance=True)
    assert not valid  # default category is unverified_model_hypothesis which is disallowed
    assert reason == "UNVERIFIED_HYPOTHESIS_DISALLOWED"
    assert rec is not None and rec.source_id == "source_doc_99"

    # 3. Allow unverified hypothesis when flag is True
    valid, reason, _ = validate_provenance_record(
        "source_doc_99",
        require_provenance=True,
        allow_unverified_hypotheses=True,
    )
    assert valid
    assert reason is None

    # 4. Unauthorized source category
    auth = AuthorizationContext(
        authorized_categories=("user_input", "external_document"),
        allow_unverified_hypotheses=True,
    )
    prov_unknown = ProvenanceRecord(
        source_id="src_1",
        authorization_category="confidential_internal",
    )
    valid, reason, _ = validate_provenance_record(prov_unknown, auth)
    assert not valid
    assert reason == "UNAUTHORIZED_SOURCE_CATEGORY"

    # Admin role can override category
    admin_auth = AuthorizationContext(role="admin")
    valid, reason, _ = validate_provenance_record(prov_unknown, admin_auth)
    assert valid
    assert reason is None


def test_invalid_provenance_dict_and_types() -> None:
    valid, reason, _ = validate_provenance_record(
        {"invalid": "no_source_id"}, require_provenance=True
    )
    assert not valid
    assert reason == "INVALID_PROVENANCE_SCHEMA"

    valid, reason, _ = validate_provenance_record(12345, require_provenance=True)
    assert not valid
    assert reason == "UNSUPPORTED_PROVENANCE_TYPE"

    with pytest.raises(ValidationError):
        ProvenanceRecord.from_dict("not_a_dict")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ProvenanceRecord(source_id="s", observation_timestamp=datetime.now())  # naive
