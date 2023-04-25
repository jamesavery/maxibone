#ifndef io_h
#define io_h

#include "datatypes.hh"

namespace NS {

template <typename T>
void load_contiguous_slice(const T *data, const string filename, const uint64_t offset, const uint64_t size);
template <typename T>
void write_contiguous_slice(const T *np_data, const string filename, const uint64_t offset, const uint64_t size);

}

#endif