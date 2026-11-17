"""RUTAM native memory writing, validation, and control subsystem."""

from rutam.config import MemoryControllerConfig, MemoryValidationConfig, MemoryWriteConfig
from rutam.memory.audit import AuditLogger, AuditRecord
from rutam.memory.conflict_detection import detect_conflicts
from rutam.memory.controller import RUTAMMemoryController
from rutam.memory.controller_output import (
    CommitResult,
    ControllerStatus,
    MemoryReadOutput,
    MemoryWritePipelineOutput,
)
from rutam.memory.duplicate_detection import detect_duplicates
from rutam.memory.memory_projection import (
    MemoryProjectionOutput,
    NeuralMemoryProjection,
    TargetMemoryScorer,
)
from rutam.memory.memory_state import MemoryState
from rutam.memory.operation_predictor import (
    MemoryOperationPredictor,
    OperationPredictorOutput,
)
from rutam.memory.provenance import (
    AuthorizationContext,
    ProvenanceRecord,
    ProvenanceVerifier,
    validate_provenance_record,
)
from rutam.memory.read_pipeline import MemoryReadPipeline
from rutam.memory.transaction import GraphTransaction
from rutam.memory.validation_output import (
    MemoryValidationResult,
    ValidationBatchResult,
    ValidationDecision,
)
from rutam.memory.validation_rules import (
    check_sensitive_data,
    check_structural_invariants,
)
from rutam.memory.validator import RUTAMMemoryValidator
from rutam.memory.write_gate import (
    MemoryWriteGate,
    WriteGateOutput,
    availability_mask,
)
from rutam.memory.write_head import RUTAMMemoryWriteHead
from rutam.memory.write_losses import (
    MemoryWriteLossOutput,
    memory_write_loss,
)
from rutam.memory.write_pipeline import MemoryWritePipeline
from rutam.memory.write_proposal import (
    MemoryOperation,
    MemoryProposal,
    MemoryWriteOutput,
)

__all__ = [
    "AuditLogger",
    "AuditRecord",
    "AuthorizationContext",
    "CommitResult",
    "ControllerStatus",
    "GraphTransaction",
    "MemoryControllerConfig",
    "MemoryOperation",
    "MemoryOperationPredictor",
    "MemoryProjectionOutput",
    "MemoryProposal",
    "MemoryReadOutput",
    "MemoryReadPipeline",
    "MemoryState",
    "MemoryValidationConfig",
    "MemoryValidationResult",
    "MemoryWriteConfig",
    "MemoryWriteGate",
    "MemoryWriteLossOutput",
    "MemoryWriteOutput",
    "MemoryWritePipeline",
    "MemoryWritePipelineOutput",
    "NeuralMemoryProjection",
    "OperationPredictorOutput",
    "ProvenanceRecord",
    "ProvenanceVerifier",
    "RUTAMMemoryController",
    "RUTAMMemoryValidator",
    "RUTAMMemoryWriteHead",
    "TargetMemoryScorer",
    "ValidationBatchResult",
    "ValidationDecision",
    "WriteGateOutput",
    "availability_mask",
    "check_sensitive_data",
    "check_structural_invariants",
    "detect_conflicts",
    "detect_duplicates",
    "memory_write_loss",
    "validate_provenance_record",
]
