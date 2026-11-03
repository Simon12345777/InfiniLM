# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright 2026 InfiniLM Contributors
# Adapted from https://github.com/vllm-project/vllm (KV-transfer interfaces/utilities).
# Modified for InfiniLM engine and connector integration; see licenses/THIRD_PARTY.txt.

import os
import uuid
from dataclasses import dataclass, field
from typing import Optional

KV_ROLE_CHOICES = frozenset({"kv_producer", "kv_consumer"})


@dataclass
class KVTransferConfig:
    """Configuration for KV cache transfer in prefill/decode (P/D) separation.

    Attributes:
        kv_connector: Name of the KV connector to use (e.g. 'MooncakeConnector').
                      None disables KV transfer.
        kv_role: Role of this node: 'kv_producer' (prefill) or 'kv_consumer' (decode).
        engine_id: Unique identifier for this engine instance used in KV transfers.
                   Auto-generated (UUID) if not provided.
        kv_connector_extra_config: Extra configuration dict passed to the connector backend.
    """

    kv_connector: Optional[str] = None
    kv_role: Optional[str] = None
    engine_id: Optional[str] = None
    kv_connector_extra_config: Optional[dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kv_connector is not None and self.kv_role is None:
            raise ValueError("Please specify kv_role when kv_connector is set.")

        if self.kv_role is not None and self.kv_role not in KV_ROLE_CHOICES:
            raise ValueError(
                f"Unsupported kv_role: {self.kv_role!r}. "
                f"Supported roles are {sorted(KV_ROLE_CHOICES)}"
            )

        if self.engine_id is None:
            self.engine_id = f"{self.kv_role}_" + str(uuid.uuid4())

        self.kv_connector_extra_config = dict(self.kv_connector_extra_config or {})
        self.kv_connector_extra_config.setdefault("mooncake_protocol", "rdma")

        overlap = self.kv_connector_extra_config.get("chunk_transfer_overlap", False)
        if type(overlap) is not bool:
            raise ValueError("chunk_transfer_overlap must be a boolean")
        if overlap and (self.kv_role != "kv_producer" or self.kv_connector != "MooncakeConnector"):
            raise ValueError("chunk_transfer_overlap requires a Mooncake producer")
        limit = self.kv_connector_extra_config.get("max_inflight_prefix_writes", 0)
        if type(limit) is not int or limit < 0:
            raise ValueError("max_inflight_prefix_writes must be a nonnegative integer")
        if limit and not overlap:
            raise ValueError("max_inflight_prefix_writes requires chunk_transfer_overlap")
        allowed_extra_config_keys = frozenset({
            "mooncake_protocol", "num_workers", "chunk_transfer_overlap", "max_inflight_prefix_writes"
        })
        unknown_keys = (
            set(self.kv_connector_extra_config.keys()) - allowed_extra_config_keys
        )
        if unknown_keys:
            raise ValueError(
                f"Unsupported kv_connector_extra_config keys: {sorted(unknown_keys)}. "
                f"Supported keys are {sorted(allowed_extra_config_keys)}"
            )

        mooncake_protocol = self.kv_connector_extra_config["mooncake_protocol"]
        if mooncake_protocol not in ["tcp", "rdma"]:
            raise ValueError(f"only support tcp or rdma, but got {mooncake_protocol}")

        if mooncake_protocol == "tcp":
            # NOTE: force use tcp for Mooncake
            os.environ["MC_FORCE_TCP"] = "true"
