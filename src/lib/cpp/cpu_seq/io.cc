#include <iostream>
#include <fstream>

#include "io.hh"

using namespace std;

template <typename T>
void load_contiguous_slice(T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    ifstream file;
    file.open(filename.c_str(), ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "load_slice: Error opening %s for reading.\n", filename.c_str());
        exit(-1);
    }
    file.seekg(offset * sizeof(T), ios::beg);
    file.read((char*) data, size * sizeof(T));
    file.close();
}

template <typename T>
void write_contiguous_slice(const T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    ofstream file;
    file.open(filename.c_str(), ios::binary | ios::in);
    if (!file.is_open()) {
        file.clear();
        file.open(filename.c_str(), ios::binary);
    }
    file.seekp(offset * sizeof(T), ios::beg);
    file.write((char*) data, size * sizeof(T));
    file.close();
}

// TODO non-contiguous
