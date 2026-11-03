# Mooncake TCP private CUDA stream experiment

This is an opt-in source patch for Mooncake commit
`e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c` (v0.3.14), not an InfiniLM native-library patch.
The installed wheel's exact source commit is unknown. Compare OFF and ON using
the **same rebuilt engine** before attributing a difference to this patch.

Run `python prepare_source.py /path/to/isolated/Mooncake` once on that checkout.
Build the `engine` CMake target with CUDA and TCP enabled; do not install over
the existing wheel. Load the resulting extension from a separate Python package
directory and check `mooncake.engine.__file__` before starting P and D.

`MC_TCP_PRIVATE_CUDA_STREAM=1` enables pinned session staging buffers and a
reusable `cudaStreamNonBlocking` per session/device. Unset or `0` keeps the
original CUDA copy path. Enable it on both P and D for the experiment.

## Ordering and lifetime

- The producer must finish writing the source **before** submission and retain
  immutable source pages until transfer completion. The private stream does
  not wait for unrelated model work on the default stream.
- The audited InfiniLM single-GPU P path reads sampled tokens through
  `to_numpy()` / `Runtime::memcpyD2H()`, which synchronizes the owner stream
  before `_update_requests()` publishes a computed chunk. Other producer
  paths, tensor/pipeline parallelism and speculative execution require a
  separate readiness audit before enabling this experiment.
- Every staging copy synchronizes only its private stream before advancing
  the TCP callback chain. Socket completion retains existing buffer ownership.
  No final KV-ready flag is published early by this patch.
- Failed pinned allocation closes/fails the session. Failure to prove CUDA
  completion is fatal to the process rather than allowing unsafe page reuse.
- This removes an implicit default-stream dependency, not the TCP copy itself.
  The I/O thread still waits for the actual copy duration.

## Validation

`test_copy.cpp` exercises D2H/H2D data, partial slices, resizing and device
switches. Compile with g++, the CUDA include/library paths, `-lcudart -lglog`.
Run in separate processes with the environment unset and set to `1` (pass an
extra argument for the enabled case). Full TCP and PD validation must also pass
before treating the rebuilt engine as ready for a benchmark.

The current artifacts are experimental; see the execution report for which
build and integration checks have actually completed.

## Bounded idle resource pool

`MC_TCP_CUDA_POOL_SLOTS=32` enables reuse of up to 32 idle buffer/stream pairs
per device; default `0` disables it. Requires the private-copy path to be enabled.
Accepted range is 0..256. Each retained buffer is at most 1 MiB, so the idle
pinned-memory bound is slots * 1 MiB per device (normal 64 KiB slices use less).
This bounds the **idle cache**, not active transfer memory. Misses allocate as
before; overflow and oversized resources are destroyed rather than cached.

Pool entries are exclusively leased, separated by device, and protected by a
mutex. No CUDA operation runs while holding the pool mutex. Only session
destruction returns a healthy resource: the Asio shared owners have then finished
using it and copy() has drained CUDA. Any observed CUDA error prevents reuse.
Pooled idle resources live until process exit, deliberately avoiding CUDA/static
destructor ordering problems; OS/CUDA process teardown reclaims them. There is
no mid-process trim API in this experiment.
