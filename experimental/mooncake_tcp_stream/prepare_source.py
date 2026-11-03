"""Apply the opt-in CUDA TCP patch to an isolated, pinned Mooncake checkout."""
import argparse
import re
import shutil
import subprocess
from pathlib import Path

COMMIT = "e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    root = args.source.resolve()
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise SystemExit(f"Expected {COMMIT}, found {actual}; review patch against this version first")
    folder = root / "mooncake-transfer-engine/src/transport/tcp_transport"
    impl = folder / "tcp_transport_session_impl.h"
    cpp = folder / "tcp_transport.cpp"
    text, cpp_text = impl.read_text(), cpp.read_text()
    if "TcpPrivateCudaCopy" in text:
        raise SystemExit("Patch already applied; no files changed")
    original = '    char* ensure(size_t size) {\n'
    assert text.count(original) == 1
    text = text.replace(original, '''    char* ensure(size_t size, int device) {
#ifdef USE_CUDA
        if (tcpPrivateCudaStreamEnabled()) return private_copy_.ensure(size, device);
#endif
''')
    original = '   private:\n    std::vector<char> buffer_;'
    assert text.count(original) == 1
    text = text.replace(original, '''#ifdef USE_CUDA
    cudaError_t copy(void* dst, const void* src, size_t size, int device) {
        if (tcpPrivateCudaStreamEnabled())
            return private_copy_.copy(dst, src, size, device);
        return cudaMemcpy(dst, src, size, cudaMemcpyDefault);
    }
#endif

   private:
#ifdef USE_CUDA
    TcpPrivateCudaCopy private_copy_;
#endif
    std::vector<char> buffer_;''')
    # Allocate pinned memory only for the opt-in path. Allocation failure must
    # close/fail the session, never pass a null staging pointer to Asio.
    marker = 'dram_buffer = staging_buffer_.ensure(buffer_size);'
    assert text.count(marker) == 4
    server, client = text.split('struct ClientSession', 1)
    server = server.replace(marker, '''dram_buffer = staging_buffer_.ensure(buffer_size, cuda_device_);
            if (!dram_buffer) return;  // Close the server session on allocation failure.''')
    client = client.replace(marker, '''dram_buffer = staging_buffer_.ensure(buffer_size, cuda_device_);
            if (!dram_buffer) {
                finalize(TransferStatusEnum::FAILED, false);
                return;
            }''')
    text = server + 'struct ClientSession' + client
    # Preserve MACA and every other backend's behavior; CUDA alone gets the
    # environment-controlled helper in all four READ/WRITE staging paths.
    pattern = r'(#ifdef USE_MACA\n.*?)(#else\n)(.*?cudaMemcpy\((.*?)cudaMemcpyDefault\);)(\n#endif)'
    def replace(match):
        args_text = match[4].rstrip().rstrip(',').rstrip()
        return (match[1] + '#elif defined(USE_CUDA)\n'
                + '            cudaError_t cuda_status = staging_buffer_.copy('
                + args_text + ', cuda_device_);\n' + match[2] + match[3] + match[5])
    text, count = re.subn(pattern, replace, text, flags=re.S)
    assert count == 4, count
    include = '#include "cuda_alike.h"'
    assert cpp_text.count(include) == 1
    cpp_text = cpp_text.replace(include, include + '\n#ifdef USE_CUDA\n#include "tcp_cuda_copy.h"\n#endif')
    shutil.copyfile(Path(__file__).with_name("tcp_cuda_copy.h"), folder / "tcp_cuda_copy.h")
    impl.write_text(text)
    cpp.write_text(cpp_text)
    print(f"Patched {root}; default CUDA path unchanged unless MC_TCP_PRIVATE_CUDA_STREAM=1")


if __name__ == "__main__":
    main()
