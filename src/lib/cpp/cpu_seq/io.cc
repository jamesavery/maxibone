#include <iostream>
#include <fstream>

#include "io.hh"

using namespace std;
namespace cpu_seq {

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
    uint64_t n = file.tellg() - offset * sizeof(T);
    assert (n == size * sizeof(T) && "Error reading the correct amount of bytes.");
    file.close();
}

template <typename T>
void write_contiguous_slice(const T *data,
        const string filename,
        const uint64_t offset,
        const uint64_t size) {
    ofstream file;
    file.open(filename.c_str(), ios::binary | ios::in | ios::out);
    if (!file.is_open()) {
        file.clear();
        file.open(filename.c_str(), ios::binary | ios::out);
    }
    // Error handling
    if (file.seekp(offset * sizeof(T), ios::beg).fail()) {
        fprintf(stderr, "write_slice: Error seeking to %lu in %s.\n", offset, filename.c_str());
        file.close();
        exit(-1);
    }
    //file.seekp(offset * sizeof(T), ios::beg);
    uint64_t
        byte_size = size * sizeof(T),
        block_size = 4096 * 1024,
        blocks = byte_size / block_size;
    blocks = byte_size % block_size == 0 ? blocks : blocks + 1;
    assert(blocks * block_size >= byte_size && "Error calculating the amount of blocks to write.");
    for (uint64_t block = 0; block < blocks; block++) {
        uint64_t this_block_size = std::min(block_size, byte_size - block * block_size);
        if (file.write((char*) data + block * block_size, this_block_size).fail()) {
            fprintf(stderr, "write_slice: Error writing block %ld to %s.\n", block, filename.c_str());
            // Print reason for failure
            if (file.bad()) {
                fprintf(stderr, "write_slice: Bad bit set.\n");
            }
            if (file.fail()) {
                fprintf(stderr, "write_slice: Fail bit set.\n");
            }
            if (file.eof()) {
                fprintf(stderr, "write_slice: EOF bit set.\n");
            }
            if (file.good()) {
                fprintf(stderr, "write_slice: Good bit set.\n");
            }
            file.close();
            exit(-1);
        }
        // Check if the correct number of bytes were written
        if ((uint64_t) file.tellp() != (offset * sizeof(T) + block * block_size + this_block_size)) {
            fprintf(stderr, "write_slice: Error writing block %lu to %s.\n", block, filename.c_str());
            fprintf(stderr, "write_slice: Wrote %ld bytes, expected %lu.\n", (int64_t) file.tellp(), offset * sizeof(T) + block * block_size + this_block_size);
            file.close();
            exit(-1);
        }

    }
    assert ((uint64_t) file.tellp() == offset * sizeof(T) + size * sizeof(T) && "Error writing the correct amount of bytes.");
    //if (file.write((char*) data, size * sizeof(T)).fail()) {
    //    fprintf(stderr, "write_slice: Error writing to %s.\n", filename.c_str());
    //    file.close();
    //    exit(-1);
    //}
    //file.write((char*) data, size * sizeof(T));
    if (file.flush().fail()) {
        fprintf(stderr, "write_slice: Error flushing %s.\n", filename.c_str());
        file.close();
        exit(-1);
    }
    //file.flush(); // Should have flushed, but just in case
    file.close();
}

// TODO non-contiguous
}
