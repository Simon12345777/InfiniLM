#include "tcp_cuda_copy.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::cerr << cudaGetErrorString(status) << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    assert(mooncake::tcpPrivateCudaStreamEnabled() == (argc > 1));
    int count = 0;
    check(cudaGetDeviceCount(&count));
    assert(count >= 2);
    mooncake::TcpPrivateCudaCopy staging;
    unsigned cases = 0;
    // Reuse across sizes and device switches; include the final partial slice.
    for (int device : {0, 1, 0}) {
        check(cudaSetDevice(device));
        void* gpu = nullptr;
        check(cudaMalloc(&gpu, 131073));
        for (size_t size : {size_t(1), size_t(65536), size_t(131073), size_t(17)}) {
            std::vector<unsigned char> expected(size), result(size);
            for (size_t i = 0; i < size; ++i) expected[i] = (i * 31 + cases) % 251;
            // Explicit producer completion: this experiment must not rely on
            // implicit default-stream ordering for source readiness.
            check(cudaMemcpy(gpu, expected.data(), size, cudaMemcpyHostToDevice));
            check(cudaDeviceSynchronize());
            char* host = staging.ensure(size, device);
            assert(host);
            check(staging.copy(host, gpu, size, device));
            assert(std::equal(expected.begin(), expected.end(),
                              reinterpret_cast<unsigned char*>(host)));
            std::fill(host, host + size, char(137));
            check(staging.copy(gpu, host, size, device));
            check(cudaMemcpy(result.data(), gpu, size, cudaMemcpyDeviceToHost));
            assert(std::all_of(result.begin(), result.end(), [](auto x) { return x == 137; }));
            ++cases;
        }
        check(cudaFree(gpu));
    }
    std::cout << "PASS " << cases << " D2H/H2D round trips, resize and device switching\n";
}
