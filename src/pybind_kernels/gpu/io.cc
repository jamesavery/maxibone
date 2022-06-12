#include <iostream>
#include <fstream>

#include "io.hh"

using namespace std;

template <typename T>
void load_contiguous_slice(T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    throw runtime_error(string("Library doesn't have a gpu implementation of ") + __FUNCTION__);
}

template <typename T>
void write_contiguous_slice(const T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    throw runtime_error(string("Library doesn't have a gpu implementation of ") + __FUNCTION__);
}
