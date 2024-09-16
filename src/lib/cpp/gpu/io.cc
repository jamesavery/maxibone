/**
 * @file io.cc
 * GPU implementation of the I/O functions.
 */
#include <iostream>
#include <fstream>

#include "../cpu_seq/io.cc"

namespace gpu {

    template <typename T>
    void load_contiguous_slice(T *data,
            const std::string filename,
            const uint64_t offset,
            const uint64_t size) {
        cpu_seq::load_contiguous_slice(data, filename, offset, size);
    }

    template <typename T>
    void write_contiguous_slice(const T *data,
            const std::string filename,
            const uint64_t offset,
            const uint64_t size) {
        cpu_seq::write_contiguous_slice(data, filename, offset, size);
    }

}