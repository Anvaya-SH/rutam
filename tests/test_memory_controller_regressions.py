"""Memory-controller lifecycle recovery tests."""
from unittest.mock import patch

import pytest
import torch

from rutam.memory.controller import RUTAMMemoryController
from rutam.memory.memory_state import MemoryState


def test_failed_proposal_restores_controller_state():
    controller = RUTAMMemoryController()
    before = controller.state
    with patch.object(controller.write_pipeline, 'forward', side_effect=RuntimeError('failed')):
        with pytest.raises(RuntimeError, match='failed'):
            controller.propose_memory_update(
                torch.zeros(1, 1, controller.config.memory_write.hidden_dim)
            )
    assert controller.state == before
    assert controller.state != MemoryState.WRITE_PENDING
