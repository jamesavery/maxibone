#include <iostream>
#include <fstream>

#include "io.hh"
#include "../cpu_seq/io.cc"

using namespace std;
namespace cpu {

template <typename T>
void load_contiguous_slice(T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    cpu_seq::load_contiguous_slice(data, filename, offset, size);
}

template <typename T>
void write_contiguous_slice(const T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    cpu_seq::write_contiguous_slice(data, filename, offset, size);
}

}
