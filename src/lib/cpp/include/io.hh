#ifndef io_h
#define io_h

#include "datatypes.hh"
#include <fcntl.h>
#include <fstream>

// TODO these should be in a proper spot.
// TODO Emphasize names!
// e_ = element sizes / indices
// b_ = byte sizes / indices
constexpr int64_t b_disk_block_size = 4096; // TODO get from filesystem.

//
// External functions (expected to be called by Python)
//
namespace NS {

    /**
     * Load a contiguous array from a file.
     *
     * @param data The memory pointer to load the data into.
     * @param filename The path to the file.
     * @param offset The offset in number of elements into the file to start reading from.
     * @param size The number of elements to read.
     * @tparam T The type of the data.
     */
    template <typename T>
    void load_contiguous_slice(const T *data, const std::string filename, const uint64_t offset, const uint64_t size);

    /**
     * Write a contiguous array to a file.
     *
     * @param data The memory pointer to write the data from.
     * @param filename The path to the file.
     * @param offset The offset in number of elements into the file to start writing to.
     * @param size The number of elements to write.
     * @tparam T The type of the data.
     */
    template <typename T>
    void write_contiguous_slice(const T *data, const std::string filename, const uint64_t offset, const uint64_t size);

    // TODO non-contiguous

}

//
// Internal Functions (expected to be called by C++)
//

/**
 * Open a file for reading. The file is expected to be in binary format.
 * This version uses the standard `fopen` function, which lets the OS handle caching.
 *
 * @param path The path to the file.
 * @return A pointer to the file.
 */
FILE* open_file_read(const std::string &path);

/**
 * Open a file for reading. The file is expected to be in binary format.
 * This version uses the `O_DIRECT` flag, which bypasses the OS cache. This allows for faster reading, but comes with the following restrictions:
 *
 * 1. The file must be aligned to the block size of the disk.
 *
 * 2. The file must be read in multiples of the block size of the disk.
 *
 * 3. The pointer to memory must be aligned to the block size of the disk.
 *
 * @param path The path to the file.
 * @return A pointer to the file.
 * @see open_file_read
 */
FILE* open_file_read_direct(const std::string &path);

/**
 * Open a file for writing. The file is expected to be in binary format.
 * This version uses the standard `fopen` function, which lets the OS handle caching.
 *
 * @param path The path to the file.
 * @return A pointer to the file.
 */
FILE* open_file_write(const std::string &path);

/**
 * Open a file for writing. The file is expected to be in binary format.
 * This version uses the `O_DIRECT` flag, which bypasses the OS cache. This allows for faster writing, but comes with the following restrictions:
 *
 * 1. The file must be aligned to the block size of the disk.
 *
 * 2. The file must be written in multiples of the block size of the disk.
 *
 * 3. The pointer to memory must be aligned to the block size of the disk.
 *
 * @param path The path to the file.
 * @return A pointer to the file.
 */
FILE* open_file_write_direct(const std::string &path);

/**
 * Load a contiguous array from a file.
 *
 * @param dst The memory pointer to load the data into.
 * @param fp The file pointer.
 * @param e_offset The offset in number of elements into the file to start reading from.
 * @param e_n_elements The number of elements to read.
 * @tparam T The type of the data.
 * @return The number of elements read.
 */
template <typename T>
int64_t load_flat(T *__restrict__ dst, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Load a contiguous array from a file. This version assumes that the memory pointer and number of bytes to read is aligned to the block size of the disk.
 *
 * @param dst The memory pointer to load the data into.
 * @param fp The file pointer.
 * @param e_offset The offset in number of elements into the file to start reading from.
 * @param e_n_elements The number of elements to read.
 * @tparam T The type of the data.
 * @return The number of elements read.
 */
template <typename T>
int64_t load_flat_aligned(T *__restrict__ dst, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Reads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into `dst`.
 * `disk_shape` is the shape of the file on disk, and `shape` is the shape of the allocated memory.
 * This version exists to avoid allocating a vector for each call, and reads directly from disk.
 * The last stride is always assumed to be 1, for both src and dst.
 * It is up to the caller to ensure that 1) `range` doesn't exceed `shape`, 2) `dst` is large enough to hold the data, 3) `dst` is set to 0 in case of a partial read and 0s are desired and 4) `dst` is an aligned allocation (e.g. using `aligned_alloc()`) to maximize performance.
 *
 * @param dst The memory pointer to load the data into.
 * @param path The path to the file.
 * @param e_shape_total The total shape of the data in memory, in number of elements.
 * @param e_shape_global The shape of the data on disk, in number of elements.
 * @param e_range The range of the data to read, in number of elements.
 * @param e_offset_global The global offset of the data in memory, in number of elements.
 * @tparam T The type of the data.
 * @return The number of elements read.
 */
template <typename T>
int64_t load_strided(T *__restrict__ dst, const std::string &path, const idx3d &e_shape_total, const idx3d &e_shape_global, const idx3drange &e_range, const idx3d &e_offset_global);


// TODO Functions cannot be overloaded purely on return type. Hence the following function is commented out.
/**
 * Load a flat array from a file. This version allocates memory for the data.
 *
 * @param path The path to the file.
 * @param e_offset The offset in number of elements into the file to start reading from.
 * @param e_n_elements The number of elements to read.
 * @tparam T The type of the data.
 * @return A pointer to the data.
 */
//template <typename T>
//T* load_file_flat(const std::string &path, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Load a flat array from a file. This version allocates a vector for containing the data.
 *
 * @param path The path to the file.
 * @param e_offset The offset in number of elements into the file to start reading from.
 * @param e_n_elements The number of elements to read.
 * @tparam T The type of the data.
 * @return A vector with the data.
 */
template <typename T>
std::vector<T> load_file_flat(const std::string &path, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Load a flat array from a file.
 *
 * @param dst The memory pointer to load the data into.
 * @param path The path to the file.
 * @param e_offset The offset in number of elements into the file to start reading from.
 * @param e_n_elements The number of elements to read.
 * @tparam T The type of the data.
 */
template <typename T>
void load_file_flat(T *__restrict__ dst, const std::string &path, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Load a strided array from a file. This version allocates a vector for the data.
 *
 * @param path The path to the file.
 * @param e_disk_shape The shape of the data on disk, in number of elements.
 * @param e_shape The shape of the data in memory, in number of elements.
 * @param e_range The range of the data to read, in number of elements.
 * @param e_offset_global The global offset of the data in memory, in number of elements.
 * @tparam T The type of the data.
 * @return A vector with the data.
 */
template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global);

/**
 * Store a contiguous array to a file.
 *
 * @param src The memory pointer to write the data from.
 * @param fp The file pointer.
 * @param e_offset The offset in number of elements into the file to start writing to.
 * @param e_n_elements The number of elements to write.
 * @tparam T The type of the data.
 * @return The number of elements written.
 */
template <typename T>
int64_t store_flat(const T *__restrict__ src, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Store a contiguous array to a file. This version assumes that the memory pointer and number of bytes to write is aligned to the block size of the disk.
 *
 * @param src The memory pointer to write the data from.
 * @param fp The file pointer.
 * @param e_offset The offset in number of elements into the file to start writing to.
 * @param e_n_elements The number of elements to write.
 * @tparam T The type of the data.
 * @return The number of elements written.
 */
template <typename T>
int64_t store_flat_aligned(const T *__restrict__ src, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Store a strided array to a file.
 *
 * @param src The memory pointer to write the data from.
 * @param path The path to the file.
 * @param e_shape_total The total shape of the data in memory, in number of elements.
 * @param e_shape_global The shape of the data on disk, in number of elements.
 * @param e_range The range of the data to write, in number of elements.
 * @param e_offset_global The global offset of the data in memory, in number of elements.
 * @tparam T The type of the data.
 * @return The number of elements written.
 */
template <typename T>
int64_t store_strided(const T *__restrict__ src, const std::string &path, const idx3d &e_shape_total, const idx3d &e_shape_global, const idx3drange &e_range, const idx3d &e_offset_global);

/**
 * Store a flat array to a file.
 *
 * @param src The memory pointer to write the data from.
 * @param path The path to the file.
 * @param e_offset The offset in number of elements into the file to start writing to.
 * @param e_n_elements The number of elements to write.
 * @tparam T The type of the data.
 */
template <typename T>
void store_file_flat(const T *__restrict__ data, const std::string &path, const int64_t e_offset, const int64_t e_n_elements);

/**
 * Store a flat array to a file.
 *
 * @param src The vector to write the data from.
 * @param path The path to the file.
 * @param e_offset The offset in number of elements into the file to start writing to.
 * @param e_n_elements The number of elements to write.
 * @tparam T The type of the data.
 */
template <typename T>
void store_file_flat(const std::vector<T> &data, const std::string &path, const int64_t e_offset);

/**
 * Store a strided array to a file.
 *
 * @param src The memory pointer to write the data from.
 * @param path The path to the file.
 * @param e_disk_shape The shape of the data on disk, in number of elements.
 * @param e_shape The shape of the data in memory, in number of elements.
 * @param e_range The range of the data to write, in number of elements.
 * @param e_offset_global The global offset of the data in memory, in number of elements.
 * @tparam T The type of the data.
 */
template <typename T>
void store_file_strided(const T *__restrict__ data, const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global);

/**
 * Store a strided array to a file.
 *
 * @param src The vector to write the data from.
 * @param path The path to the file.
 * @param e_disk_shape The shape of the data on disk, in number of elements.
 * @param e_shape The shape of the data in memory, in number of elements.
 * @param e_range The range of the data to write, in number of elements.
 * @param e_offset_global The global offset of the data in memory, in number of elements.
 * @tparam T The type of the data.
 */
template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global);

//
// File open functions
//

FILE* open_file_read(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY);
    return fdopen(fd, "rb");
}

FILE* open_file_read_direct(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    return fdopen(fd, "rb");
}

FILE* open_file_write(const std::string &path) {
    int fd = open(path.c_str(), O_CREAT | O_RDWR, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    return fdopen(fd, "r+b");
}

FILE* open_file_write_direct(const std::string &path) {
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    return fdopen(fd, "r+b");
}

//
// File load functions
//

template <typename T>
int64_t load_flat_aligned(T *__restrict__ dst, FILE *fp, const int64_t e_offset, const int64_t e_n_elements) {
    fseek(fp, e_offset*sizeof(T), SEEK_SET);
    int64_t e_n = fread((char *) dst, sizeof(T), e_n_elements, fp);
    return e_n;
}

template <typename T>
int64_t load_flat(T *__restrict__ dst, FILE *fp, const int64_t e_offset, const int64_t e_n_elements) {
    int64_t
        e_disk_block_size = b_disk_block_size / sizeof(T),
        b_offset = e_offset * sizeof(T),
        b_n_elements = e_n_elements * sizeof(T),
        e_buffer_start = e_offset % e_disk_block_size,
        e_buffer_end = e_buffer_start + e_n_elements,
        e_front_elements = (e_disk_block_size - e_buffer_start) % e_disk_block_size,
        e_back_elements = e_buffer_end % e_disk_block_size,
        e_in_between = (e_n_elements - e_front_elements) - e_back_elements,
        e_aligned_start = (e_offset / e_disk_block_size) * e_disk_block_size,
        e_aligned_end = e_offset + e_n_elements + ((e_disk_block_size - e_back_elements) % e_disk_block_size),
        e_aligned_n_elements = e_aligned_end - e_aligned_start,
        b_aligned_start = e_aligned_start * sizeof(T),
        b_aligned_end = e_aligned_end * sizeof(T),
        b_aligned_n_elements = e_aligned_n_elements * sizeof(T);

    if (b_offset % b_disk_block_size == 0 && b_n_elements % b_disk_block_size == 0 && (int64_t) dst % b_disk_block_size == 0) {
        int64_t e_n = load_flat_aligned(dst, fp, e_offset, e_n_elements);
        assert (e_n == e_n_elements && "Failed to read all elements");
        return e_n;
    }

    if (DEBUG) {
        std::cout << "--------------- sizes are" << std::endl;
        std::cout << "e_offset " << e_offset << std::endl;
        std::cout << "e_n_elements " << e_n_elements << std::endl;
        std::cout << "e_disk_block_size " << e_disk_block_size << std::endl;
        std::cout << "b_offset " << b_offset << std::endl;
        std::cout << "b_n_elements " << b_n_elements << std::endl;
        std::cout << "e_buffer_start " << e_buffer_start << std::endl;
        std::cout << "e_buffer_end " << e_buffer_end << std::endl;
        std::cout << "e_front_elements " << e_front_elements << std::endl;
        std::cout << "e_back_elements " << e_back_elements << std::endl;
        std::cout << "e_in_between " << e_in_between << std::endl;
        std::cout << "e_aligned_start " << e_aligned_start << std::endl;
        std::cout << "e_aligned_end " << e_aligned_end << std::endl;
        std::cout << "e_aligned_n_elements " << e_aligned_n_elements << std::endl;
        std::cout << "b_aligned_start " << b_aligned_start << std::endl;
        std::cout << "b_aligned_end " << b_aligned_end << std::endl;
        std::cout << "b_aligned_n_elements " << b_aligned_n_elements << std::endl;
        std::cout << "---------------" << std::endl;
    }

    T *buffer = (T *) aligned_alloc(b_disk_block_size, b_aligned_n_elements);
    int64_t e_n = load_flat_aligned(buffer, fp, e_aligned_start, e_aligned_n_elements);
    int64_t b_pos = ftell(fp);

    if (DEBUG) std::cout << "e_n " << e_n << std::endl;

    // Correct read, or read the rest of the file:
    assert ((e_n == e_aligned_n_elements || b_pos == b_offset + b_n_elements) && "Failed to read all elements");
    memcpy((char *) dst, (char *) (buffer + e_buffer_start), b_n_elements);
    free(buffer);

    return e_n_elements;
}

template <typename T>
int64_t load_strided(T *__restrict__ dst, const std::string &path, const idx3d &e_shape_total, const idx3d &e_shape_global, const idx3drange &e_range, const idx3d &e_offset_global) {
    // Calculate the strides and sizes
    const idx3d
        e_strides_total = {e_shape_total.y*e_shape_total.x, e_shape_total.x, 1},
        e_strides_global = {e_shape_global.y*e_shape_global.x, e_shape_global.x, 1},
        e_sizes = {e_range.z_end - e_range.z_start, e_range.y_end - e_range.y_start, e_range.x_end - e_range.x_start};

    // If the shape on disk is the same as the shape in memory, just load the entire file
    if (e_shape_global.y == e_shape_total.y && e_shape_global.x == e_shape_total.x && e_offset_global.y == 0 && e_offset_global.x == 0 && e_range.y_start == 0 && e_range.x_start == 0 && e_range.y_end == e_shape_total.y && e_range.x_end == e_shape_total.x) {
        return load_file(dst + (e_offset_global.z*e_strides_global.z), path, e_range.z_start*e_strides_total.z, e_sizes.z*e_strides_total.z);
    }
    assert (false && "Not implemented yet :) - After the deadline!");

    // Open the file
    int64_t e_total_n = 0;
    FILE *fp = open_file_read(path);
    fseek(fp, (e_range.z_start*e_strides_total.z + e_range.y_start*e_strides_total.y + e_range.x_start*e_strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t e_z = 0; e_z < e_sizes.z; e_z++) {
        for (int64_t e_y = 0; e_y < e_sizes.y; e_y++) {
            int64_t e_n = fread((char *) &dst[(e_z+e_offset_global.z)*e_strides_global.z + (e_y+e_offset_global.y)*e_strides_global.y + e_offset_global.x*e_strides_global.x], sizeof(T), e_sizes.x, fp);
            assert(e_n == e_sizes.x && "Failed to read all elements");
            e_total_n += e_n;
            fseek(fp, (e_strides_total.y - e_sizes.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (e_strides_total.z - e_sizes.y*e_strides_total.y)*sizeof(T), SEEK_CUR);
    }
    fclose(fp);

    return e_total_n;
}

// TODO Functions cannot be overloaded purely on return type. Hence the following function is commented out.
//template <typename T>
//T* load_file_flat(const std::string &path, const int64_t e_offset, const int64_t e_n_elements) {
//    T *data = (T *) malloc(e_n_elements * sizeof(T));
//    FILE *fp = open_file_read(path);
//    int64_t e_n = load_flat(data, fp, e_offset, e_n_elements);
//    assert (e_n == e_n_elements && "Failed to read all elements");
//    fclose(fp);
//    return data;
//}

template <typename T>
std::vector<T> load_file_flat(const std::string &path, const int64_t e_offset, const int64_t e_n_elements) {
    std::vector<T> data(e_n_elements);
    FILE *fp = open_file_read_direct(path);
    int64_t e_n = load_flat(data.data(), fp, e_offset, e_n_elements);
    assert (e_n == e_n_elements && "Failed to read all elements");
    fclose(fp);
    return data;
}

template <typename T>
void load_file_flat(T *__restrict__ dst, const std::string &path, const int64_t e_offset, const int64_t e_n_elements) {
    FILE *fp = open_file_read_direct(path);
    int64_t e_n = load_flat(dst, fp, e_offset, e_n_elements);
    assert (e_n == e_n_elements && "Failed to read all elements");
    fclose(fp);
}

template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global) {
    std::vector<T> data(e_shape.z*e_shape.y*e_shape.x);
    load_strided(data.data(), path, e_disk_shape, e_shape, e_range, e_offset_global);
    return data;
}

//
// File store functions
//
// Assumes that `src` and `src + offset` is aligned to `disk_block_size` and that `n_elements` is a multiple of `disk_block_size`.
template <typename T>
int64_t store_flat_aligned(const T *__restrict__ src, FILE *fp, const int64_t e_offset, const int64_t e_n_elements) {
    fseek(fp, e_offset*sizeof(T), SEEK_SET);
    int64_t e_n = fwrite((char *) src, sizeof(T), e_n_elements, fp);
    return e_n;
}

template <typename T>
int64_t store_flat(const T *__restrict__ src, FILE *fp, const int64_t e_offset, const int64_t e_n_elements) {
    int64_t
        e_disk_block_size = b_disk_block_size / sizeof(T),
        b_offset = e_offset * sizeof(T),
        b_n_elements = e_n_elements * sizeof(T),
        e_buffer_start = e_offset % e_disk_block_size,
        e_buffer_end = e_buffer_start + e_n_elements,
        e_front_elements = (e_disk_block_size - e_buffer_start) % e_disk_block_size,
        e_back_elements = e_buffer_end % e_disk_block_size,
        e_in_between = (e_n_elements - e_front_elements) - e_back_elements,
        e_aligned_start = (e_offset / e_disk_block_size) * e_disk_block_size,
        e_aligned_end = e_offset + e_n_elements + ((e_disk_block_size - e_back_elements) % e_disk_block_size),
        e_aligned_n_elements = e_aligned_end - e_aligned_start,
        b_aligned_start = e_aligned_start * sizeof(T),
        b_aligned_end = e_aligned_end * sizeof(T),
        b_aligned_n_elements = e_aligned_n_elements * sizeof(T);

    if (b_offset % b_disk_block_size == 0 && b_n_elements % b_disk_block_size == 0 && (int64_t) src % b_disk_block_size == 0) {
        int64_t e_n = store_flat_aligned(src, fp, e_offset, e_n_elements);
        return e_n;
    }

    assert(e_front_elements + e_in_between + e_back_elements == e_n_elements && "Front, in-between and back elements don't add up to n_elements");
    assert(e_aligned_n_elements > e_n_elements && "Aligned n_elements is smaller than n_elements");
    assert(b_aligned_n_elements % b_disk_block_size == 0 && "Aligned n_elements is not a multiple of disk_block_size");

    if (DEBUG) std::cout << "sizes are " << e_front_elements << " " << e_in_between << " " << e_back_elements << std::endl;

    // Get the current_file_size
    fseek(fp, 0, SEEK_END);
    int64_t b_current_file_size = ftell(fp);

    T *buffer = (T *) aligned_alloc(b_disk_block_size, b_aligned_n_elements);

    // Mask the buffer, primarily for debugging, as this is easy to spot in hexdump
    memset(buffer, 0xff, b_aligned_n_elements);

    if (DEBUG) std::cout << b_current_file_size << " > " << b_aligned_start << "(" << (b_current_file_size > b_aligned_start) << ") && " << e_front_elements << " != 0 (" << (e_front_elements != 0) << ")" << std::endl;
    if (b_current_file_size > b_aligned_start && e_front_elements != 0) {
        int64_t
            b_this_n = std::min(b_current_file_size - b_aligned_start, b_disk_block_size),
            e_this_n = b_this_n / sizeof(T),
            e_n = load_flat_aligned(buffer, fp, e_aligned_start, e_this_n);

        if (DEBUG) {
            std::cout << "e_disk_block_size " << e_disk_block_size << std::endl;
            std::cout << "b_offset " << b_offset << std::endl;
            std::cout << "b_n_elements " << b_n_elements << std::endl;
            std::cout << "e_buffer_start " << e_buffer_start << std::endl;
            std::cout << "e_buffer_end " << e_buffer_end << std::endl;
            std::cout << "e_front_elements " << e_front_elements << std::endl;
            std::cout << "e_back_elements " << e_back_elements << std::endl;
            std::cout << "e_in_between " << e_in_between << std::endl;
            std::cout << "e_aligned_start " << e_aligned_start << std::endl;
            std::cout << "e_aligned_end " << e_aligned_end << std::endl;
            std::cout << "e_aligned_n_elements " << e_aligned_n_elements << std::endl;
            std::cout << "b_aligned_start " << b_aligned_start << std::endl;
            std::cout << "b_aligned_end " << b_aligned_end << std::endl;
            std::cout << "b_aligned_n_elements " << b_aligned_n_elements << std::endl;
            std::cout << "Front elements: " << e_front_elements << " | e_n: " << e_n << std::endl;
        }

        assert(e_n == e_this_n && "Failed to read all elements");
    }

    if (b_current_file_size > b_aligned_end - b_disk_block_size && e_back_elements != 0) {
        int64_t
            b_this_n = std::min(b_current_file_size - (b_aligned_end - b_disk_block_size), b_disk_block_size),
            e_this_n = b_this_n / sizeof(T),
            e_n = load_flat_aligned(buffer + e_buffer_start + e_front_elements + e_in_between, fp, e_aligned_end - e_disk_block_size, e_this_n);

        if (DEBUG) {
            std::cout << "e_disk_block_size " << e_disk_block_size << std::endl;
            std::cout << "b_offset " << b_offset << std::endl;
            std::cout << "b_n_elements " << b_n_elements << std::endl;
            std::cout << "e_buffer_start " << e_buffer_start << std::endl;
            std::cout << "e_buffer_end " << e_buffer_end << std::endl;
            std::cout << "e_front_elements " << e_front_elements << std::endl;
            std::cout << "e_back_elements " << e_back_elements << std::endl;
            std::cout << "e_in_between " << e_in_between << std::endl;
            std::cout << "e_aligned_start " << e_aligned_start << std::endl;
            std::cout << "e_aligned_end " << e_aligned_end << std::endl;
            std::cout << "e_aligned_n_elements " << e_aligned_n_elements << std::endl;
            std::cout << "b_aligned_start " << b_aligned_start << std::endl;
            std::cout << "b_aligned_end " << b_aligned_end << std::endl;
            std::cout << "b_aligned_n_elements " << b_aligned_n_elements << std::endl;
            std::cout
                << "Back elements: " << e_back_elements
                << " | e_n: " << e_n
                << " | e_this_n: " << e_this_n
                << " | current_file_size: " << b_current_file_size
                << std::endl;
        }

        assert(e_n == e_this_n && "Failed to read all elements");
    }

    memcpy((char *) (buffer + e_buffer_start), (char *) src, b_n_elements);

    store_flat_aligned(buffer, fp, e_aligned_start, e_aligned_n_elements);

    free(buffer);

    return e_n_elements;
}

template <typename T>
int64_t store_strided(const T *__restrict__ src, const std::string &path, const idx3d &e_shape_total, const idx3d &e_shape_global, const idx3drange &e_range, const idx3d &e_offset_global) {
    // Calculate the strides and sizes
    const idx3d
        e_strides_total = {e_shape_total.y*e_shape_total.x, e_shape_total.x, 1},
        e_strides_global = {e_shape_global.y*e_shape_global.x, e_shape_global.x, 1},
        e_sizes = {e_range.z_end - e_range.z_start, e_range.y_end - e_range.y_start, e_range.x_end - e_range.x_start};

    FILE *fp = open_file_write(path);

    // If the shape on disk is the same as the shape in memory, just store the entire file
    if (e_shape_global.y == e_shape_total.y && e_shape_global.x == e_shape_total.x && e_offset_global.y == 0 && e_offset_global.x == 0 && e_range.y_start == 0 && e_range.x_start == 0 && e_range.y_end == e_shape_total.y && e_range.x_end == e_shape_total.x) {
        int64_t e_n = store_flat(src + e_offset_global.z*e_strides_global.z, fp, e_range.z_start*e_strides_total.z, e_sizes.z*e_strides_total.z);
        fclose(fp);
        return e_n;
    }

    assert (false && "Not implemented yet :) - After the deadline!");

    // Open the file
    fseek(fp, (e_range.z_start*e_strides_total.z + e_range.y_start*e_strides_total.y + e_range.x_start*e_strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < e_sizes.z; z++) {
        for (int64_t y = 0; y < e_sizes.y; y++) {
            int64_t n = fwrite((char *) &src[(z+e_offset_global.z)*e_strides_global.z + (y+e_offset_global.y)*e_strides_global.y + (0+e_offset_global.x)*e_strides_global.x], sizeof(T), e_sizes.x, fp);
            assert (n == e_sizes.x && "Failed to write all elements");
            fseek(fp, (e_strides_total.y - e_sizes.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (e_strides_total.z - e_sizes.y*e_strides_total.y) * sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

template <typename T>
void store_file_flat(const T *__restrict__ data, const std::string &path, const int64_t e_offset, const int64_t e_n_elements) {
    FILE *fp = open_file_write(path);
    int64_t e_n = store_flat(data, fp, e_offset, e_n_elements);
    assert (e_n == e_n_elements && "Failed to write all elements");
    fclose(fp);
}

template <typename T>
void store_file_flat(const std::vector<T> &data, const std::string &path, const int64_t e_offset) {
    store_file_flat(data.data(), path, e_offset, data.size());
}

template <typename T>
void store_file_strided(const T *__restrict__ data, const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global) {
    store_strided(data, path, e_disk_shape, e_shape, e_range, e_offset_global);
}

template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global) {
    store_file_strided(data.data(), path, e_disk_shape, e_shape, e_range, e_offset_global);
}

#endif