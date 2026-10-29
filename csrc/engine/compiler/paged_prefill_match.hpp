#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace infinilm::engine {
// Shape/boundary admission only. Native request validation still owns page IDs
// and writable-slot validity. All arrays are validated CPU int32 tensors.
inline bool matches_paged_prefill_chunk(
    size_t chunk, size_t batch, size_t tokens, size_t capacity,
    const int32_t *offsets, const int32_t *cu,
    const int32_t *past, const int32_t *total) {
    if (chunk <= 1 || batch == 0 || !offsets || !cu || !past || !total
        || chunk > static_cast<size_t>(std::numeric_limits<int32_t>::max()) / batch
        || tokens != batch * chunk || offsets[0] != 0 || cu[0] != 0) {
        return false;
    }
    for (size_t row = 0; row < batch; ++row) {
        if (int64_t(offsets[row + 1]) - offsets[row] != int64_t(chunk)
            || past[row] < 0 || int64_t(total[row]) - past[row] != int64_t(chunk)
            || int64_t(cu[row + 1]) - cu[row] != total[row]
            || total[row] <= 0 || static_cast<size_t>(total[row]) > capacity) {
            return false;
        }
    }
    return true;
}
} // namespace infinilm::engine
