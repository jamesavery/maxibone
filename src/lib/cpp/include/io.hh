#ifndef io_h
#define io_h

template <typename T>
void load_contiguous_slice(T *data, const string filename, const uint64_t offset, const uint64_t size);
template <typename T>
void write_contiguous_slice(T *np_data, const string filename, const uint64_t offset, const uint64_t size);

#endif