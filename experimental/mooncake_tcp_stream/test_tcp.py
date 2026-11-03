"""Bounded real TCP WRITE/READ test; run separately with the switch OFF/ON."""
import json
import os
import torch
import mooncake.engine
from mooncake.engine import TransferEngine

print("engine:", mooncake.engine.__file__, flush=True)
engines = [TransferEngine(), TransferEngine()]
buffers = [torch.empty(2 * 1024 * 1024 + 17, dtype=torch.uint8, device=f"cuda:{i}") for i in range(2)]
registered = []
try:
    for engine, tensor in zip(engines, buffers):
        assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0
        assert engine.register_memory(tensor.data_ptr(), tensor.numel()) == 0
        registered.append((engine, tensor))
    session = f"127.0.0.1:{engines[1].get_rpc_port()}"
    for size in (1, 17, 65536, 65537, buffers[0].numel()):
        buffers[0].fill_(137)
        buffers[1].zero_()
        for device in (0, 1): torch.cuda.synchronize(device)
        assert engines[0].batch_transfer_sync_write(session, [buffers[0].data_ptr()], [buffers[1].data_ptr()], [size]) == 0
        assert bool((buffers[1][:size] == 137).all().item())
        assert bool((buffers[1][size:] == 0).all().item())
        buffers[1].fill_(49)
        buffers[0].zero_()
        for device in (0, 1): torch.cuda.synchronize(device)
        assert engines[0].batch_transfer_sync_read(session, [buffers[0].data_ptr()], [buffers[1].data_ptr()], [size]) == 0
        assert bool((buffers[0][:size] == 49).all().item())
        assert bool((buffers[0][size:] == 0).all().item())
    print(json.dumps(dict(success=True, cases=10, switch=os.getenv("MC_TCP_PRIVATE_CUDA_STREAM", "0"))))
finally:
    for engine, tensor in registered:
        engine.unregister_memory(tensor.data_ptr())
