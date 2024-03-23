
#include "connected_components.hh"

#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <unordered_set>

// Debug functions
constexpr bool DEBUG = false;

void print_idx3d(const idx3d &idx) {
    std::cout << idx.z << " " << idx.y << " " << idx.x << std::endl;
}

void print_vector(const std::vector<int64_t> &vec) {
    std::cout << "[ ";
    for (int64_t i = 0; i < (int64_t) vec.size(); i++) {
        if (i > 0)
            std::cout << ", ";
        std::cout << vec[i];
    }
    std::cout << " ]" << std::endl;
}

// TODO these should be in a proper spot.
// TODO Emphasize names!
// e_ = element sizes / indices
// b_ = byte sizes / indices
int64_t b_disk_block_size = 4096; // TODO get from filesystem.

FILE* open_file_read(const std::string &path);
FILE* open_file_read_direct(const std::string &path);
FILE* open_file_write(const std::string &path);
FILE* open_file_write_direct(const std::string &path);

template <typename T> int64_t load_flat_aligned(T *__restrict__ dst, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> int64_t load_flat(T *__restrict__ dst, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> int64_t load_strided(T *__restrict__ dst, const std::string &path, const idx3d &e_shape_total, const idx3d &e_shape_global, const idx3drange &e_range, const idx3d &e_offset_global);
//template <typename T> T* load_file_flat(const std::string &path, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> std::vector<T> load_file_flat(const std::string &path, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> void load_file_flat(T *__restrict__ dst, const std::string &path, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> std::vector<T> load_file_strided(const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global);

template <typename T> int64_t store_flat_aligned(const T *__restrict__ src, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> int64_t store_flat(const T *__restrict__ src, FILE *fp, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> int64_t store_strided(const T *__restrict__ src, const std::string &path, const idx3d &e_shape_total, const idx3d &e_shape_global, const idx3drange &e_range, const idx3d &e_offset_global);
template <typename T> void store_file_flat(const T *__restrict__ data, const std::string &path, const int64_t e_offset, const int64_t e_n_elements);
template <typename T> void store_file_flat(const std::vector<T> &data, const std::string &path, const int64_t e_offset);
template <typename T> void store_file_strided(const T *__restrict__ data, const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global);
template <typename T> void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &e_disk_shape, const idx3d &e_shape, const idx3drange &e_range, const idx3d &e_offset_global);

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

// Assumes that `dst` and `dst + offset` is aligned to `disk_block_size` and that `n_elements` is a multiple of `disk_block_size`.
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

// Reads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into `dst`.
// `disk_shape` is the shape of the file on disk, and `shape` is the shape of the allocated memory.
// This version exists to avoid allocating a vector for each call, and reads directly from disk.
// The last stride is always assumed to be 1, for both src and dst.
// It is up to the caller to ensure that 1) `range` doesn't exceed `shape`, 2) `dst` is large enough to hold the data, 3) `dst` is set to 0 in case of a partial read and 0s are desired and 4) `dst` is an aligned allocation (e.g. using `aligned_alloc()`) to maximize performance.
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
    FILE *fp = open_file_read(path);
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

namespace cpu_par {

void apply_renaming(std::vector<int64_t> &img, std::vector<int64_t> &to_rename) {
    NS::apply_renaming(img.data(), img.size(), to_rename);
}

void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        assert (img[i] < (int64_t) to_rename.size() && "Label out of bounds");
        if (img[i] < (int64_t) to_rename.size()) {
            img[i] = to_rename[img[i]];
        }
    }
}

int64_t apply_renamings(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &e_total_shape, const idx3d &e_global_shape, const std::vector<std::vector<int64_t>> &renames, const bool verbose) {
    auto cc_app_start = std::chrono::high_resolution_clock::now();

    // Apply the renaming to a new global file
    int64_t chunks = n_labels.size();
    // Generate the paths to the different chunks
    std::vector<std::string> paths(chunks);
    for (int64_t i = 0; i < chunks; i++) {
        paths[i] = base_path + std::to_string(i) + ".int64";
    }
    std::string all_path = base_path + "all.int64";
    int64_t
        //largest_chunk = std::max(e_global_shape.z, (e_total_shape.z - (e_total_shape.z / e_global_shape.z) * e_global_shape.z) + e_global_shape.z),
        //e_disk_block_size = b_disk_block_size / sizeof(int64_t),
        e_largest_chunk = e_total_shape.z - ((chunks-1) * e_global_shape.z),
        e_chunk_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
        //b_chunk_size = e_chunk_size * sizeof(int64_t),
        e_largest_chunk_size = e_largest_chunk * e_global_shape.y * e_global_shape.x,
        b_largest_chunk_size = e_largest_chunk_size * sizeof(int64_t),
        b_aligned_chunk_size = ((b_largest_chunk_size + b_disk_block_size-1) / b_disk_block_size) * b_disk_block_size;

    assert (e_largest_chunk >= e_global_shape.z && "The largest chunk is smaller than the global shape");
    assert ((chunks-1) * e_global_shape.z + e_largest_chunk == e_total_shape.z && "The chunks don't add up to the total shape");

    if (verbose) {
        std::cout << "Largest chunk: " << e_largest_chunk << std::endl;
        std::cout << "Total shape: " << e_total_shape.z << " " << e_total_shape.y << " " << e_total_shape.x << std::endl;
        std::cout << "Global shape: " << e_global_shape.z << " " << e_global_shape.y << " " << e_global_shape.x << std::endl;
        std::cout << "Chunks: " << chunks << std::endl;
        std::cout << "Chunk size: " << e_chunk_size << std::endl;
        std::cout << "Largest chunk size: " << e_largest_chunk_size << std::endl;
    }

    FILE *all_file = open_file_write(all_path);

    int64_t
        *debug_image,
        e_img_size = e_total_shape.z * e_total_shape.y,
        b_img_size = e_img_size * sizeof(int64_t);
    if (DEBUG) {
        debug_image = (int64_t *) malloc(b_img_size);
        memset(debug_image, 0, b_img_size);
    }

    int64_t *chunk = (int64_t *) aligned_alloc(b_disk_block_size, b_aligned_chunk_size * sizeof(int64_t));
    for (int64_t i = 0; i < chunks; i++) {
        int64_t e_this_chunk_size = (i == chunks-1) ? e_largest_chunk_size : e_chunk_size;

        auto load_start = std::chrono::high_resolution_clock::now();
        load_file_flat(chunk, paths[i], 0, e_this_chunk_size);
        auto load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = load_end - load_start;
        if (verbose) {
            std::cout << "load_file: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
        }

        auto apply_start = std::chrono::high_resolution_clock::now();
        apply_renaming(chunk, e_this_chunk_size, renames[i]);
        auto apply_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
        if (verbose) {
            std::cout << "apply_renaming: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_apply.count() / 1e9 << " GB/s" << std::endl;
        }

        if (DEBUG) {
            for (int64_t ii = i*e_global_shape.z; ii < (i*e_global_shape.z) + (e_this_chunk_size / (e_global_shape.y * e_global_shape.x)); ii++) {
                for (int64_t jj = 0; jj < e_total_shape.y; jj++) {
                    debug_image[ii*e_total_shape.y + jj] = chunk[(ii-i*e_global_shape.z)*e_global_shape.y*e_global_shape.x + jj*e_global_shape.x + (e_global_shape.x/2)];
                }
            }
        }

        auto store_start = std::chrono::high_resolution_clock::now();
        store_flat(chunk, all_file, i*e_chunk_size, e_this_chunk_size);
        auto store_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_store = store_end - store_start;
        if (verbose) {
            std::cout << "store_partial: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_store.count() / 1e9 << " GB/s" << std::endl;
        }
    }
    free(chunk);
    fclose(all_file);

    if (DEBUG) {
        FILE *debug_file = open_file_write(base_path + "debug.int64");
        fseek(debug_file, 0, SEEK_SET);
        fwrite((char *) debug_image, sizeof(int64_t), e_img_size, debug_file);
        fclose(debug_file);
        free(debug_image);
    }

    auto cc_app_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cc_app = cc_app_end - cc_app_start;
    if (verbose) {
        std::cout << "connected_components lut application: " << elapsed_cc_app.count() << " s" << std::endl;
    }

    return n_labels[0];
}

void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape, const bool verbose) {
    std::vector<bool> found(n_labels+1, false);
    const idx3d strides = { global_shape.y * global_shape.x, global_shape.x, 1 };
    int64_t n_chunks = total_shape.z / global_shape.z; // Assuming that they are divisible
    FILE *file = open_file_read(path);
    int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
    int64_t *img = (int64_t *) aligned_alloc(b_disk_block_size, chunk_size * sizeof(int64_t));
    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        if (verbose) {
            std::cout << "Chunk " << chunk << " / " << n_chunks << std::endl;
        }

        auto start = std::chrono::high_resolution_clock::now();
        load_flat(img, file, chunk*chunk_size, chunk_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        if (verbose) {
            std::cout << "load_partial: " << (double) (chunk_size*sizeof(int64_t)) / elapsed.count() / 1e9 << " GB/s" << std::endl;
        }

        for (int64_t i = 0; i < chunk_size; i++) {
            int64_t label = img[i];
            if (label > n_labels || label < 0) {
                if (verbose) {
                    std::cout << "Label " << label << " in chunk " << chunk << " at index " << i <<" is outside of bounds 0:" << n_labels << std::endl;
                }
                continue;
            }
            if (!found[label]) {
                int64_t
                    z = (i / strides.z) + (chunk * global_shape.z),
                    y = (i % strides.z) / strides.y,
                    x = (i % strides.y) / strides.x;
                found[label] = true;
                out[(4*label)+0] = z;
                out[(4*label)+1] = y;
                out[(4*label)+2] = x;
            }
            out[(4*label)+3] += 1;
        }
    }
    free(img);
    fclose(file);
}

std::vector<std::vector<int64_t>> connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &global_shape, const bool verbose) {
    auto cc_start = std::chrono::high_resolution_clock::now();
    // Check if the call is well-formed
    int64_t chunks = n_labels.size();
    assert ((chunks & (chunks - 1)) == 0 && "Chunks must be a power of 2");

    // Constants
    const idx3d
        global_strides = { global_shape.y * global_shape.x, global_shape.x, 1 };

    // Generate the paths to the different chunks
    std::vector<std::string> paths(chunks);
    for (int64_t i = 0; i < chunks; i++) {
        paths[i] = base_path + std::to_string(i) + ".int64";
    }

    // Generate the adjacency tree
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> index_tree = NS::generate_adjacency_tree(chunks);


    std::vector<std::vector<std::vector<int64_t>>> renames(index_tree.size(), std::vector<std::vector<int64_t>>(chunks));
    // Rename LUTs, one for each chunk
    for (int64_t i = 0; i < (int64_t) index_tree.size(); i++) {
        #pragma omp parallel for
        for (int64_t j = 0; j < (int64_t) index_tree[i].size(); j++) {
            auto [l, r] = index_tree[i][j];
            // This doesn't handle the different chunk sizes, but it should be fine as the last chunk is the only one that differs and we only read the first layer from that one
            int64_t last_layer = (global_shape.z-1) * global_strides.z;
            std::vector<int64_t> a = load_file_flat<int64_t>(paths[l], last_layer, global_strides.z);
            std::vector<int64_t> b = load_file_flat<int64_t>(paths[r], 0, global_strides.z);

            if (DEBUG) {
                store_file_flat(a, base_path + "a_" + std::to_string(l) + ".int64", 0);
                store_file_flat(b, base_path + "b_" + std::to_string(r) + ".int64", 0);
            }

            for (int64_t k = 0; k < i; k++) {
                // Apply the renamings obtained from the previous layer
                apply_renaming(a, renames[k][l]);
                apply_renaming(b, renames[k][r]);
            }

            auto [rename_l, rename_r, n_new_labels] = NS::relabel(a, n_labels[l], b, n_labels[r], global_shape, false);
            n_labels[l] = n_new_labels;
            n_labels[r] = n_new_labels;

            // Store the renamings
            renames[i][l] = rename_l;
            renames[i][r] = rename_r;
            if (i > 0) {
                int64_t subtrees = (int64_t) std::pow(2, i);

                // Run through the left subtree
                for (int64_t k = j*2*subtrees; k < (j*2*subtrees)+subtrees; k++) {
                    renames[i][k] = rename_l;
                    n_labels[k] = n_new_labels;
                }

                // Run through the right subtree
                for (int64_t k = (j*2*subtrees)+subtrees; k < (j*2*subtrees)+(2*subtrees); k++) {
                    renames[i][k] = rename_r;
                    n_labels[k] = n_new_labels;
                }
            }
        }
    }

    std::vector<std::vector<int64_t>> renames_final(chunks);
    for (int64_t i = 0; i < chunks; i++) {
        renames_final[i] = renames[0][i];
        for (int64_t j = 1; j < (int64_t) renames.size(); j++) {
            for (int64_t k = 0; k < (int64_t) renames_final[i].size(); k++) {
                renames_final[i][k] = renames[j][i][renames_final[i][k]];
            }
        }
    }

    auto cc_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cc = cc_end - cc_start;
    if (verbose) {
        std::cout << "connected_components lut building: " << elapsed_cc.count() << " s" << std::endl;
    }

    return renames_final;
}

void count_sizes(int64_t *__restrict__ img, std::vector<int64_t> &sizes, const int64_t n_labels, const int64_t size) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < size; i++) {
        assert (img[i] <= n_labels && "Label out of bounds");
        sizes[img[i]]++;
    }
}

void filter_largest(const std::string &base_path, bool *__restrict__ mask, const std::vector<std::vector<int64_t>> &renames, const int64_t largest, const idx3d &e_total_shape, const idx3d &e_global_shape, const bool verbose) {
    // Apply the renaming to a new global file
    int64_t
        e_global_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
        chunks = renames.size(),
        e_largest_chunk = e_total_shape.z - ((chunks-1) * e_global_shape.z),
        e_chunk_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
        e_largest_chunk_size = e_largest_chunk * e_global_shape.y * e_global_shape.x,
        b_largest_chunk_size = e_largest_chunk_size * sizeof(int64_t),
        b_aligned_chunk_size = ((b_largest_chunk_size + b_disk_block_size-1) / b_disk_block_size) * b_disk_block_size;

    // Generate the paths to the different chunks
    std::vector<std::string> paths(chunks);
    for (int64_t i = 0; i < chunks; i++) {
        paths[i] = base_path + std::to_string(i) + ".int64";
    }

    int64_t *chunk = (int64_t *) aligned_alloc(b_disk_block_size, b_aligned_chunk_size);

    int64_t
        b_img_size = e_total_shape.z * e_total_shape.y * sizeof(int64_t),
        b_aligned_img_size = ((b_img_size + (b_disk_block_size-1)) / b_disk_block_size) * b_disk_block_size,
        e_aligned_img_size = b_aligned_img_size / sizeof(int64_t),
        *debug_image;
    if (DEBUG) {
        debug_image = (int64_t *) aligned_alloc(b_disk_block_size, b_aligned_img_size);
        memset(debug_image, 0, b_aligned_img_size);
    }

    for (int64_t i = 0; i < chunks; i++) {
        int64_t e_this_chunk_size = (i == chunks-1) ? e_largest_chunk_size : e_chunk_size;

        auto load_start = std::chrono::high_resolution_clock::now();
        load_file_flat(chunk, paths[i], 0, e_this_chunk_size);
        auto load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = load_end - load_start;
        if (verbose) {
            std::cout << "load_file: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
        }

        auto apply_start = std::chrono::high_resolution_clock::now();
        apply_renaming(chunk, e_this_chunk_size, renames[i]);
        auto apply_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
        if (verbose) {
            std::cout << "apply_renaming: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_apply.count() / 1e9 << " GB/s" << std::endl;
        }

        if (DEBUG) {
            for (int64_t ii = i*e_global_shape.z; ii < (i*e_global_shape.z) + (e_this_chunk_size / (e_global_shape.y * e_global_shape.x)); ii++) {
                for (int64_t jj = 0; jj < e_total_shape.y; jj++) {
                    debug_image[ii*e_total_shape.y + jj] = chunk[(ii-i*e_global_shape.z)*e_global_shape.y*e_global_shape.x + jj*e_global_shape.x + (e_global_shape.x/2)];
                }
            }
        }

        auto filter_start = std::chrono::high_resolution_clock::now();
        for (int64_t j = 0; j < e_this_chunk_size; j++) {
            assert (i*e_global_size + j < (e_total_shape.z * e_total_shape.y * e_total_shape.x) && "Index out of bounds");
            mask[i*e_global_size + j] = chunk[j] == largest;
        }
        auto filter_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_filter = filter_end - filter_start;
        if (verbose) {
            std::cout << "filter_largest: " << (double) (e_this_chunk_size*sizeof(bool)) / elapsed_filter.count() / 1e9 << " GB/s" << std::endl;
        }
    }

    if (DEBUG) {
        FILE *debug_file = open_file_write(base_path + "debug_filter_largest.int64");
        store_flat(debug_image, debug_file, 0, e_aligned_img_size);
        free(debug_image);
        fclose(debug_file);
    }
}

std::tuple<mapping_t, mapping_t> get_mappings(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape) {
    std::vector<mapping_t> mappings_a;
    std::vector<mapping_t> mappings_b;

    mapping_t mapping_a(n_labels_a+1);
    mapping_t mapping_b(n_labels_b+1);

    #pragma omp parallel num_threads(8)
    {
        int64_t n_threads = omp_get_num_threads();

        #pragma omp single
        {
            mappings_a.resize(n_threads, mapping_t(n_labels_a+1));
            mappings_b.resize(n_threads, mapping_t(n_labels_b+1));
        }

        #pragma omp for schedule(static) collapse(2)
        for (int64_t y = 0; y < global_shape.y; y++) {
            for (int64_t x = 0; x < global_shape.x; x++) {
                int64_t i = (y * global_shape.x) + x;
                if (a[i] != 0 && b[i] != 0) {
                    mappings_a[omp_get_thread_num()][a[i]].insert(b[i]);
                    mappings_b[omp_get_thread_num()][b[i]].insert(a[i]);
                }
            }
        }

        for (int64_t i = 0; i < n_threads; i++) {
            #pragma omp for schedule(static)
            for (int64_t j = 1; j < n_labels_a+1; j++) {
                mapping_a[j].insert(mappings_a[i][j].begin(), mappings_a[i][j].end());
            }
            #pragma omp for schedule(static)
            for (int64_t j = 1; j < n_labels_b+1; j++) {
                mapping_b[j].insert(mappings_b[i][j].begin(), mappings_b[i][j].end());
            }
        }
    }

    return { mapping_a, mapping_b };
}

std::vector<int64_t> get_sizes(std::vector<int64_t> &img, int64_t n_labels) {
    std::vector<int64_t> sizes(n_labels, 0);
    for (int64_t i = 0; i < (int64_t) img.size(); i++) {
        sizes[img[i]]++;
    }

    return sizes;
}

std::vector<std::vector<std::tuple<int64_t, int64_t>>> generate_adjacency_tree(const int64_t chunks) {
    int64_t log_chunks = (int64_t) std::ceil(std::log2(chunks));
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> tree(log_chunks);
    for (int64_t layer = 0; layer < log_chunks; layer++) {
        int64_t n_elements = chunks >> (layer+1); // chunks / 2^layer
        int64_t i = 1 << layer; // 1 * 2^layer
        std::vector<std::tuple<int64_t, int64_t>> indices;
        for (int64_t j = i-1; j < i*n_elements*2; j += i*2) {
            indices.push_back({j, j+1});

        }
        tree[layer] = indices;
    }
    return tree;
}

int64_t largest_component(const std::string &base_path, const std::vector<std::vector<int64_t>> &renames, const int64_t n_labels, const idx3d &e_total_shape, const idx3d &e_global_shape, const bool verbose) {

    // Apply the renaming to a new global file
    int64_t
        chunks = renames.size(),
        e_largest_chunk = std::max(e_global_shape.z, (e_total_shape.z - (e_total_shape.z / e_global_shape.z) * e_global_shape.z) + e_global_shape.z),
        e_chunk_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
        e_largest_chunk_size = e_largest_chunk * e_global_shape.y * e_global_shape.x,
        b_largest_chunk_size = e_largest_chunk_size * sizeof(int64_t),
        b_aligned_chunk_size = ((b_largest_chunk_size + b_disk_block_size-1) / b_disk_block_size) * b_disk_block_size,
        e_aligned_chunk_size = b_aligned_chunk_size / sizeof(int64_t);

    // Generate the paths to the different chunks
    std::vector<std::string> paths(chunks);
    for (int64_t i = 0; i < chunks; i++) {
        paths[i] = base_path + std::to_string(i) + ".int64";
    }

    std::vector<int64_t> sizes(n_labels+1, 0);

    int64_t *chunk = (int64_t *) aligned_alloc(b_disk_block_size, b_aligned_chunk_size);

    for (int64_t i = 0; i < chunks; i++) {
        int64_t e_this_chunk_size = (i == chunks-1) ? e_largest_chunk_size : e_chunk_size;
        assert (e_this_chunk_size <= e_aligned_chunk_size && "Chunk size is larger than aligned chunk size");

        auto load_start = std::chrono::high_resolution_clock::now();
        load_file_flat(chunk, paths[i], 0, e_this_chunk_size);
        auto load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = load_end - load_start;
        if (verbose) {
            std::cout << "load_file: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
        }

        auto apply_start = std::chrono::high_resolution_clock::now();
        apply_renaming(chunk, e_this_chunk_size, renames[i]);
        auto apply_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
        if (verbose) {
            std::cout << "apply_renaming: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_apply.count() / 1e9 << " GB/s" << std::endl;
        }

        auto sizes_start = std::chrono::high_resolution_clock::now();
        count_sizes(chunk, sizes, n_labels, e_this_chunk_size);
        auto sizes_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_sizes = sizes_end - sizes_start;
        if (verbose) {
            std::cout << "count_sizes: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_sizes.count() / 1e9 << " GB/s" << std::endl;
        }
    }

    auto largest_start = std::chrono::high_resolution_clock::now();
    int64_t largest = 1;
    for (int64_t i = 2; i < n_labels+1; i++) {
        if (sizes[i] > sizes[largest]) {
            largest = i;
        }
    }
    auto largest_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_largest = largest_end - largest_start;
    if (verbose) {
        std::cout << "max_element: " << (double) (n_labels*sizeof(int64_t)) / elapsed_largest.count() / 1e9 << " GB/s" << std::endl;
        std::cout << "Largest element is: " << largest << std::endl;
        std::cout << "It occurs " << sizes[largest] << " times" << std::endl;
    }

    return largest;
}

std::vector<idx3d> merge_canonical_names(std::vector<idx3d> &names_a, std::vector<idx3d> &names_b) {
    std::vector<idx3d> names(names_a.size());
    for (int64_t i = 1; i < (int64_t) names_a.size(); i++) {
        if (names_a[i].z == -1) {
            names[i] = names_b[i];
        } else {
            names[i] = names_a[i];
        }
    }

    return names;
}

std::vector<int64_t> merge_labels(mapping_t &mapping_a, const mapping_t &mapping_b, const std::vector<int64_t> &to_rename_b) {
    std::list<int64_t> to_check;
    std::vector<int64_t> to_rename_a(mapping_a.size());
    to_rename_a[0] = 0;
    for (int64_t i = 1; i < (int64_t) mapping_a.size(); i++) {
        to_check.push_back(i);
        to_rename_a[i] = i;
    }
    bool updated;
    while (!to_check.empty()) {
        updated = false;
        int64_t label_a = to_check.front();
        std::unordered_set<int64_t> others_a = mapping_a[label_a];
        for (int64_t label_b : others_a) {
            if (label_b < (int64_t) to_rename_b.size()) { // Initially, the mapping_t will be empty
                label_b = to_rename_b[label_b];
            }
            std::unordered_set<int64_t> others_b = mapping_b[label_b];
            for (int64_t label_a2 : others_b) {
                label_a2 = to_rename_a[label_a2]; // Renames to self in the beginning
                if (label_a != label_a2) {
                    updated = true;
                    mapping_a[label_a].insert(mapping_a[label_a2].begin(), mapping_a[label_a2].end());
                    mapping_a[label_a2].clear();
                    mapping_a[label_a2].insert(-1);
                    to_rename_a[label_a2] = label_a;
                    to_check.remove(label_a2);
                }
            }
        }
        if (!updated) {
            to_check.pop_front();
        }
    }

    return to_rename_a;
}

void print_canonical_names(const std::vector<idx3d> &names_a) {
    std::cout << "Canonical names:" << std::endl;
    for (int64_t i = 1; i < (int64_t) names_a.size(); i++) {
        std::cout << i << ": " << names_a[i].z << " " << names_a[i].y << " " << names_a[i].x << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

void print_mapping(const mapping_t &mapping_) {
    std::cout << "Mapping:" << std::endl;
    for (int64_t i = 1; i < (int64_t) mapping_.size(); i++) {
        std::cout << i << ": { ";
        for (int64_t entry : mapping_[i]) {
            std::cout << entry << " ";
        }
        std::cout << "}" << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

void print_rename(const std::vector<int64_t> &to_rename) {
    std::cout << "Rename:" << std::endl;
    for (int64_t i = 1; i < (int64_t) to_rename.size(); i++) {
        std::cout << i << ": " << to_rename[i] << std::endl;
    }
    std::cout << "----------------" << std::endl;
}

// Ensures that the labels in the renaming LUTs are consecutive
int64_t recount_labels(const mapping_t &mapping_a, mapping_t &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b) {
    // We assume that mapping_t includes 0
    std::vector<int64_t> mapped_a, unmapped_a, unmapped_b;
    int64_t popped_a = 0, popped_b = 0;
    for (int64_t i = 1; i < (int64_t) mapping_a.size(); i++) {
        if (mapping_a[i].size() == 0) {
            unmapped_a.push_back(i);
        } else if (!mapping_a[i].contains(-1)) {
            mapped_a.push_back(i);
        } else {
            popped_a++;
        }
    }
    for (int64_t i = 1; i < (int64_t) mapping_b.size(); i++) {
        if (mapping_b[i].size() == 0) {
            unmapped_b.push_back(i);
        } else if (mapping_b[i].contains(-1)) {
            popped_b++;
        }
    }
    // Sanity check
    assert (mapped_a.size() + unmapped_a.size() == mapping_a.size()-popped_a-1);
    assert (mapped_a.size() + unmapped_b.size() == mapping_b.size()-popped_b-1);

    // Assign the first mapped_a labels to start from 1
    std::vector<int64_t> new_rename_a(mapping_a.size());
    for (int64_t i = 0; i < (int64_t) mapped_a.size(); i++) {
        new_rename_a[mapped_a[i]] = i+1;
    }
    // Assign the unmapped_a labels to start from mapped_a.size()+1
    for (int64_t i = 0; i < (int64_t) unmapped_a.size(); i++) {
        new_rename_a[unmapped_a[i]] = i+1+mapped_a.size();
    }

    // Apply the new renaming to the renaming LUT
    for (int64_t i = 0; i < (int64_t) to_rename_a.size(); i++) {
        to_rename_a[i] = new_rename_a[to_rename_a[i]];
    }

    // Update mapping_t b to use the new a labels
    for (int64_t i = 1; i < (int64_t) mapping_b.size(); i++) {
        auto entries = mapping_b[i];
        std::unordered_set<int64_t> new_entries;
        for (int64_t entry : entries) {
            if (entry != -1) {
                new_entries.insert(new_rename_a[entry]);
            }
        }
        mapping_b[i] = new_entries;
    }

    // Assign the first mapped_b labels to match the mapped_a labels
    std::vector<int64_t> new_rename_b(mapping_b.size());
    for (int64_t i = 0; i < (int64_t) mapped_a.size(); i++) {
        auto label = mapped_a[i];
        auto new_label = to_rename_a[label];
        auto entries = mapping_a[label];
        for (int64_t entry : entries) {
            if (entry != -1) {
                new_rename_b[entry] = new_label;
            }
        }
    }
    // Assign the unmapped_b labels to start from 1+mapped_a.size()+unmapped_a.size()
    for (int64_t i = 0; i < (int64_t) unmapped_b.size(); i++) {
        new_rename_b[unmapped_b[i]] = i+1+mapped_a.size()+unmapped_a.size();
    }
    // Apply the new renaming to the renaming LUT
    for (int64_t i = 0; i < (int64_t) to_rename_b.size(); i++) {
        to_rename_b[i] = new_rename_b[to_rename_b[i]];
    }

    return mapped_a.size() + unmapped_a.size() + unmapped_b.size();
}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape, const bool verbose) {
    auto start = std::chrono::high_resolution_clock::now();
    auto [mapping_a, mapping_b] = get_mappings(a, n_labels_a, b, n_labels_b, global_shape);
    auto mappings_end = std::chrono::high_resolution_clock::now();
    std::vector<int64_t> empty_vec;
    auto to_rename_a = merge_labels(mapping_a, mapping_b, empty_vec);
    auto merge_a_end = std::chrono::high_resolution_clock::now();
    auto to_rename_b = merge_labels(mapping_b, mapping_a, to_rename_a);
    auto merge_b_end = std::chrono::high_resolution_clock::now();
    NS::rename_mapping(mapping_a, to_rename_b);
    auto rename_a_end = std::chrono::high_resolution_clock::now();
    NS::rename_mapping(mapping_b, to_rename_a);
    auto rename_b_end = std::chrono::high_resolution_clock::now();
    int64_t n_new_labels = recount_labels(mapping_a, mapping_b, to_rename_a, to_rename_b);
    auto recount_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double>
        elapsed_get_mappings = mappings_end - start,
        elapsed_merge_a = merge_a_end - mappings_end,
        elapsed_merge_b = merge_b_end - merge_a_end,
        elapsed_rename_a = rename_a_end - merge_b_end,
        elapsed_rename_b = rename_b_end - rename_a_end,
        elapsed_recount = recount_end - rename_b_end;

    if (verbose) {
        std::cout << "get_mappings: " << elapsed_get_mappings.count() << " s" << std::endl;
        std::cout << "merge_a: " << elapsed_merge_a.count() << " s" << std::endl;
        std::cout << "merge_b: " << elapsed_merge_b.count() << " s" << std::endl;
        std::cout << "rename_a: " << elapsed_rename_a.count() << " s" << std::endl;
        std::cout << "rename_b: " << elapsed_rename_b.count() << " s" << std::endl;
        std::cout << "recount: " << elapsed_recount.count() << " s" << std::endl;
    }

    return { to_rename_a, to_rename_b, n_new_labels };
}

void rename_mapping(mapping_t &mapping_a, const std::vector<int64_t> &to_rename_other) {
    for (int64_t i = 1; i < (int64_t) mapping_a.size(); i++) {
        auto entries = mapping_a[i];
        std::unordered_set<int64_t> new_entries;
        for (int64_t entry : entries) {
            if (entry != -1) {
                new_entries.insert(to_rename_other[entry]);
            } else {
                new_entries.insert(-1);
            }
        }
        mapping_a[i] = new_entries;
    }
}

} // namespace cpu