// Experimental NVIDIA TCP staging for Mooncake v0.3.14.
// Caller contract: source data is ready and immutable, and destination memory
// is not consumed or recycled until the transfer completes. This stream does
// NOT inherit ordering from a model's default stream.
#pragma once

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <cstdlib>
#include <array>
#include <mutex>
#include <map>

namespace mooncake {

// Bound retained idle resources, not in-flight transfers. A pool miss never
// waits for another session: it uses the existing allocation path.
inline size_t tcpCudaPoolSlots() {
    static const size_t slots = [] {
        const char* value = std::getenv("MC_TCP_CUDA_POOL_SLOTS");
        if (!value || !*value) return size_t(0);
        char* end = nullptr;
        const auto n = std::strtol(value, &end, 10);
        if (*end || n < 0 || n > 256) {
            LOG(FATAL) << "MC_TCP_CUDA_POOL_SLOTS must be between 0 and 256";
            std::abort();
        }
        return size_t(n);
    }();
    return slots;
}

struct TcpCudaResource {
    cudaStream_t stream = nullptr;
    char* buffer = nullptr;
    size_t capacity = 0;
};

class TcpCudaIdlePool {
   public:
    static TcpCudaIdlePool& instance() {
        // CUDA runtime/static destruction order is unspecified. Retain this
        // bounded cache for process lifetime; OS/CUDA teardown reclaims it.
        static auto* pool = new TcpCudaIdlePool;
        return *pool;
    }
    TcpCudaResource take(int device, size_t minimum) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto& entries = devices_[device];
        for (auto& entry : entries) {
            if (entry.stream && entry.capacity >= minimum) {
                auto result = entry;
                entry = {};
                return result;
            }
        }
        return {};
    }
    bool put(int device, TcpCudaResource resource) {
        if (!resource.stream || !resource.buffer || resource.capacity > 1024 * 1024)
            return false;
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = devices_.find(device);
        if (found == devices_.end()) return false;
        for (size_t i = 0; i < tcpCudaPoolSlots(); ++i) {
            auto& entry = found->second[i];
            if (!entry.stream) { entry = resource; return true; }
        }
        return false;
    }
   private:
    std::mutex mutex_;
    std::map<int, std::array<TcpCudaResource, 256>> devices_;
};

inline bool tcpPrivateCudaStreamEnabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("MC_TCP_PRIVATE_CUDA_STREAM");
        const bool on = value && value[0] == '1' && value[1] == '\0';
        if (on) LOG(WARNING) << "Experimental MC_TCP_PRIVATE_CUDA_STREAM=1: "
                               "caller must provide ready, immutable source data";
        return on;
    }();
    return enabled;
}

// One instance per session. Asio's session callback chain serializes its use.
// Each copy is drained before returning, so socket callbacks may safely retain
// the buffer and reuse it for the next slice without a CUDA event queue.
class TcpPrivateCudaCopy {
   public:
    TcpPrivateCudaCopy() = default;
    TcpPrivateCudaCopy(const TcpPrivateCudaCopy&) = delete;
    TcpPrivateCudaCopy& operator=(const TcpPrivateCudaCopy&) = delete;

    ~TcpPrivateCudaCopy() {
        if (!stream_ && !buffer_) return;
        // Session destruction happens only after the Asio callbacks release
        // their shared_ptr owners. Every CUDA copy was drained in copy().
        // Thus no GPU or socket operation can still access this lease.
        if (returnToPool()) return;
        int previous = -1;
        checkStatus(cudaGetDevice(&previous));
        releaseStream();
        if (buffer_) checkStatus(cudaFreeHost(buffer_));
        if (previous >= 0) checkStatus(cudaSetDevice(previous));
    }

    char* ensure(size_t size, int device) {
        auto status = selectDevice(device, size);
        if (status != cudaSuccess) return nullptr;
        if (size > capacity_) {
            void* replacement = nullptr;
            status = checkStatus(cudaHostAlloc(&replacement, size, cudaHostAllocPortable));
            if (status != cudaSuccess) {
                LOG(ERROR) << "TCP pinned staging allocation failed: "
                           << cudaGetErrorString(status);
                return nullptr;
            }
            // No copy is outstanding, and ensure() runs only after the prior
            // socket operation has finished using this session's buffer.
            if (buffer_) checkStatus(cudaFreeHost(buffer_));
            buffer_ = static_cast<char*>(replacement);
            capacity_ = size;
        }
        return buffer_;
    }

    cudaError_t copy(void* dst, const void* src, size_t size, int device) {
        auto status = selectDevice(device);
        if (status != cudaSuccess) return status;
        status = checkStatus(cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream_));
        // Also drain after a failed submission: CUDA calls may report an
        // earlier asynchronous error. Never release KV pages on uncertain
        // completion. A drain failure is fatal, not a recoverable TCP retry.
        const auto drained = checkStatus(cudaStreamSynchronize(stream_));
        if (drained != cudaSuccess) {
            LOG(FATAL) << "TCP CUDA copy completion could not be proven: "
                       << cudaGetErrorString(drained);
            std::abort();
        }
        return status;
    }

   private:
    // A session that observed a CUDA error must never return its lease to the pool.
    cudaError_t checkStatus(cudaError_t status) {
        if (status != cudaSuccess) reusable_ = false;
        return status;
    }

    bool returnToPool() {
        if (!tcpCudaPoolSlots() || !reusable_ || !stream_ || !buffer_) return false;
        const bool returned = TcpCudaIdlePool::instance().put(device_, {stream_, buffer_, capacity_});
        if (returned) {
            stream_ = nullptr;
            buffer_ = nullptr;
            capacity_ = 0;
            device_ = -1;
        }
        return returned;
    }

    cudaError_t selectDevice(int device, size_t minimum = 0) {
        // Device switching is uncommon, but never return the old buffer here:
        // copy() may still have received that host pointer from the caller.
        if (device_ != device) releaseStream();
        auto status = checkStatus(cudaSetDevice(device));
        if (status != cudaSuccess) return status;
        if (!stream_) {
            if (!buffer_ && tcpCudaPoolSlots()) {
                auto entry = TcpCudaIdlePool::instance().take(device, minimum);
                if (entry.stream) {
                    stream_ = entry.stream;
                    buffer_ = entry.buffer;
                    capacity_ = entry.capacity;
                    device_ = device;
                    return cudaSuccess;
                }
            }
            status = checkStatus(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            if (status != cudaSuccess) return status;
            device_ = device;
        }
        return cudaSuccess;
    }

    void releaseStream() {
        if (stream_) {
            checkStatus(cudaSetDevice(device_));
            // copy() has already synchronized; destroying this stream never
            // substitutes for proof of completion.
            checkStatus(cudaStreamDestroy(stream_));
            stream_ = nullptr;
        }
        device_ = -1;
    }

    cudaStream_t stream_ = nullptr;
    int device_ = -1;
    char* buffer_ = nullptr;
    size_t capacity_ = 0;
    bool reusable_ = true;
};

}  // namespace mooncake
