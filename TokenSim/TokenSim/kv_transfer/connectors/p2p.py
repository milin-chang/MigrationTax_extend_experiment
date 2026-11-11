from __future__ import annotations

from TokenSim.llm.llm_request import Request

from .base import BaseKVConnector, transfer_block_bytes
from .metadata import ConnectorTransferPlan


class P2PConnector(BaseKVConnector):
    name = "P2PConnector"

    def build_transfer_plan(
        self,
        *,
        requests: list[Request],
        source_worker_id: int,
        target_worker_id: int,
        kind: str,
        latency: float,
        request_block_counts: dict[int, int] | None = None,
    ) -> ConnectorTransferPlan:
        if request_block_counts is None:
            request_block_counts = {
                req.id: req.num_physical_token_blocks for req in requests
            }
        blocks = sum(request_block_counts.values())
        bytes_ = blocks * transfer_block_bytes(self.cache_config)
        return ConnectorTransferPlan(
            request_ids=[req.id for req in requests],
            request_block_counts=dict(request_block_counts),
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            blocks=blocks,
            bytes=bytes_,
            latency=latency,
            kind=kind,
        )
