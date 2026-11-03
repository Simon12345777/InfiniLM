#include "tcp_cuda_copy.h"
#include <algorithm>
#include <cassert>
#include <memory>
#include <thread>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    assert(mooncake::tcpCudaPoolSlots() == 1);
    // Two live sessions must never share an idle lease. With one retained slot,
    // the second return overflows and follows normal cleanup.
    auto a = std::make_unique<mooncake::TcpPrivateCudaCopy>();
    auto b = std::make_unique<mooncake::TcpPrivateCudaCopy>();
    auto first = a->ensure(65536, 0);
    auto second = b->ensure(65536, 0);
    assert(first && second && first != second);
    a.reset();
    b.reset();
    {
        mooncake::TcpPrivateCudaCopy reused;
        assert(reused.ensure(65536, 0) == first);
    }
    // Stress exclusive leases and both devices; contents differ by worker.
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 4; ++worker) threads.emplace_back([worker] {
        int device = worker % 2;
        assert(cudaSetDevice(device) == cudaSuccess);
        void* gpu = nullptr;
        assert(cudaMalloc(&gpu, 65536) == cudaSuccess);
        for (int iteration = 0; iteration < 16; ++iteration) {
            mooncake::TcpPrivateCudaCopy copy;
            auto* host = copy.ensure(65536, device);
            assert(host);
            std::fill(host, host + 65536, char(worker + 31));
            assert(copy.copy(gpu, host, 65536, device) == cudaSuccess);
            std::fill(host, host + 65536, char(0));
            assert(copy.copy(host, gpu, 65536, device) == cudaSuccess);
            assert(std::all_of(host, host + 65536, [worker](char x) { return x == worker + 31; }));
        }
        assert(cudaFree(gpu) == cudaSuccess);
    });
    for (auto& thread : threads) thread.join();
    std::cout << "PASS: reuse, exclusive leases, overflow, two GPUs, 64 concurrent round trips\n";
}
