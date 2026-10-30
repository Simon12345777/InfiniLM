from dataclasses import dataclass
from typing import Optional

from infinilm.config.kv_transfer import KVTransferConfig


@dataclass
class EngineConfig:
    """Configuration for LLM Engine.

    Attributes:
        model_path: Path to the model directory.
        draft_model_path: Optional Eagle/MTP draft model directory.
        num_draft_tokens: Number of Eagle draft tokens to verify per step.
        device: Device type string ('cpu', 'cuda', 'mlu', etc.).
        dtype: Data type string ('float16', 'bfloat16', 'float32').
        tensor_parallel_size: Number of devices for tensor parallelism.
        pipeline_parallel_size: Number of pipeline stages.
        pipeline_parallel_stage: Pipeline stage index for this engine.
        master_addr: Address used to bootstrap distributed communication.
        master_port: TCP port used to bootstrap distributed communication.
        moe_ep_backend: MoE expert-parallel backend.
        moe_ep_size: MoE expert-parallel size.
        cache_type: Cache type ('paged' or 'static').
        max_batch_size: Maximum batch size for inference (only for paged cache).
        static_cache_capacity: Physical static KV rows (1 or 2); prefills are separate and decode can batch both rows.
        max_tokens: Default maximum tokens to generate.
        num_blocks: Number of KV cache blocks (only for paged cache).
        block_size: Size of each KV cache block (only for paged cache).
        max_cache_len: Maximum sequence length (only for static cache).
        enable_prefix_caching: Whether to reuse KV cache across requests.
        enable_chunked_prefill: Split text prefill into bounded paged batches.
        prefill_chunk_size: Maximum tokens in one prefill chunk (default 1024).
        enable_chunk_prefill_graph: Record/replay chunk prefill graphs independently of enable_graph.
        max_num_batched_tokens: Per-step token budget; None uses the environment/default.
        temperature: Default sampling temperature.
        top_p: Default top-p sampling parameter.
        top_k: Default top-k sampling parameter.
        enable_graph: Whether to enable graph compiling.
        attn_backend: Attention backend to use ('default', 'flash-attn').
        use_mla: Whether to use DeepSeek V2 MLA attention when supported.
        weight_load_mode: Weight loading mode across tensor-parallel workers.
        skip_load: Whether to skip loading model weights (for testing).
    """

    model_path: str
    draft_model_path: Optional[str] = None
    num_draft_tokens: int = 4
    device: str = "cuda"
    dtype: str = "float16"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    pipeline_parallel_stage: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    moe_ep_backend: str = "disabled"
    moe_ep_size: int = 1
    cache_type: str = "paged"  # "paged" or "static"
    max_batch_size: int = 16
    max_tokens: int = 4096
    num_blocks: int = 512
    block_size: int = 256
    max_cache_len: int = 4096
    temperature: float = 1.0
    top_p: float = 0.8
    top_k: int = 1
    enable_graph: bool = False
    attn_backend: str = "default"
    use_mla: bool = False
    pre_transpose: bool = False
    weight_load_mode: str = "async"
    skip_load: bool = False
    kv_transfer_config: Optional[KVTransferConfig] = None
    enable_prefix_caching: bool = True
    static_cache_capacity: int = 1
    enable_chunked_prefill: bool = False
    max_num_batched_tokens: Optional[int] = None
    prefill_chunk_size: int = 1024
    enable_chunk_prefill_graph: bool = False

    def __post_init__(self) -> None:
        if self.enable_chunk_prefill_graph and not self.enable_chunked_prefill:
            raise ValueError(
                "enable_chunk_prefill_graph requires enable_chunked_prefill"
            )
        if type(self.prefill_chunk_size) is not int or self.prefill_chunk_size < 1:
            raise ValueError("prefill_chunk_size must be a positive integer")
        if self.max_num_batched_tokens is not None and (
            type(self.max_num_batched_tokens) is not int
            or self.max_num_batched_tokens < 1
        ):
            raise ValueError("max_num_batched_tokens must be a positive integer")
        if (
            self.enable_chunked_prefill
            and self.max_num_batched_tokens is not None
            and self.max_num_batched_tokens < self.prefill_chunk_size
        ):
            raise ValueError(
                "max_num_batched_tokens must be >= prefill_chunk_size when chunked prefill is enabled"
            )
        if self.enable_chunked_prefill and (
            self.cache_type != "paged"
            or self.draft_model_path is not None
            or (self.kv_transfer_config and self.kv_transfer_config.kv_connector)
        ):
            raise ValueError(
                "chunked prefill requires paged cache without speculative decoding or KV transfer"
            )
        if (
            type(self.static_cache_capacity) is not int
            or self.static_cache_capacity < 1
        ):
            raise ValueError("static_cache_capacity must be a positive integer")
        if self.num_draft_tokens < 1:
            raise ValueError("num_draft_tokens must be >= 1")
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        if not 0 <= self.pipeline_parallel_stage < self.pipeline_parallel_size:
            raise ValueError(
                "pipeline_parallel_stage must be in [0, pipeline_parallel_size)"
            )
        if not 1 <= self.master_port <= 65535:
            raise ValueError("master_port must be in [1, 65535]")

        if self.weight_load_mode not in {"async", "sync"}:
            raise ValueError("weight_load_mode must be either 'async' or 'sync'")

        if (
            self.kv_transfer_config is not None
            and self.kv_transfer_config.kv_connector
            and self.cache_type != "paged"
        ):
            raise ValueError("kv_transfer_config requires cache_type='paged'")
