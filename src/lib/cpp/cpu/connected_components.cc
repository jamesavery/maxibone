
#include "connected_components.hh"

#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <unordered_set>

// TODO these should be in a proper spot.
int64_t disk_block_size = 4096; // TODO get from filesystem.

FILE* open_file_read(const std::string &path);
FILE* open_file_write(const std::string &path);

template <typename T>
void load_file_no_alloc(T *dst, FILE *fp, const int64_t offset, const int64_t n_elements) {
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fread((char *) dst, sizeof(T), n_elements, fp);
    assert(n == n_elements && "Failed to read all elements");
}

// Loads `n_elements` of a file located at `path` on disk at `offset` elements from the beginning of the file, into a vector of type `T`.
template <typename T>
void load_file(T *dst, const std::string &path, const int64_t total_offset, const int64_t n_elements) {
    // Open the file
    FILE *fp = open_file_write(path);

    // Calculate the aligned start and end positions
    int64_t
        disk_block_size_elements = disk_block_size / sizeof(T),
        start_pos = total_offset*sizeof(T),
        end_pos = (total_offset+n_elements)*sizeof(T),
        aligned_start = (start_pos / disk_block_size) * disk_block_size,
        aligned_end = ((end_pos + disk_block_size - 1) / disk_block_size) * disk_block_size,
        aligned_size = aligned_end - aligned_start,
        aligned_n_elements = aligned_size / sizeof(T),
        aligned_offset = aligned_start / sizeof(T);

    if (start_pos % disk_block_size == 0 && end_pos % disk_block_size == 0 && n_elements % disk_block_size_elements == 0 && (int64_t) dst % disk_block_size == 0 && total_offset % disk_block_size_elements == 0) {
        load_file_no_alloc(dst, fp, total_offset, n_elements);
    } else {
        // Allocate a buffer for the write
        T *buffer = (T *) aligned_alloc(disk_block_size, aligned_size);

        // Read the buffer from disk
        load_file_no_alloc(buffer, fp, aligned_offset, aligned_n_elements);

        // Copy the data to the destination
        memcpy((char *) dst, (char *) buffer + start_pos - aligned_start, n_elements*sizeof(T));

        // Free the buffer and close the file
        free(buffer);
    }
    fclose(fp);
}

template <typename T>
std::vector<T> load_file(const std::string &path, const int64_t offset, const int64_t n_elements) {
    std::vector<T> data(n_elements);
    load_file(data.data(), path, offset, n_elements);
    return data;
}

// Reads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into `dst`.
// `disk_shape` is the shape of the file on disk, and `shape` is the shape of the allocated memory.
// This version exists to avoid allocating a vector for each call, and reads directly from disk.
// The last stride is always assumed to be 1, for both src and dst.
// It is up to the caller to ensure that 1) `range` doesn't exceed `shape`, 2) `dst` is large enough to hold the data, 3) `dst` is set to 0 in case of a partial read and 0s are desired and 4) `dst` is an aligned allocation (e.g. using `aligned_alloc()`) to maximize performance.
template <typename T>
void load_file_strided(T *dst, const std::string &path, const idx3d &shape_total, const idx3d &shape_global, const idx3drange &range, const idx3d &offset_global) {
    // Calculate the strides and sizes
    const idx3d
        strides_total = {shape_total.y*shape_total.x, shape_total.x, 1},
        strides_global = {shape_global.y*shape_global.x, shape_global.x, 1},
        sizes = {range.z_end - range.z_start, range.y_end - range.y_start, range.x_end - range.x_start};

    // If the shape on disk is the same as the shape in memory, just load the entire file
    if (shape_global.y == shape_total.y && shape_global.x == shape_total.x && offset_global.y == 0 && offset_global.x == 0 && range.y_start == 0 && range.x_start == 0 && range.y_end == shape_total.y && range.x_end == shape_total.x) {
        load_file(dst + (offset_global.z*strides_global.z), path, range.z_start*strides_total.z, sizes.z*strides_total.z);
        return;
    }
    assert (false && "Not implemented yet :) - After the deadline!");

    // Open the file
    FILE *fp = open_file_read(path);
    fseek(fp, (range.z_start*strides_total.z + range.y_start*strides_total.y + range.x_start*strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < sizes.z; z++) {
        for (int64_t y = 0; y < sizes.y; y++) {
            int64_t n = fread((char *) &dst[(z+offset_global.z)*strides_global.z + (y+offset_global.y)*strides_global.y + offset_global.x*strides_global.x], sizeof(T), sizes.x, fp);
            assert(n == sizes.x && "Failed to read all elements");
            fseek(fp, (strides_total.y - sizes.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_total.z - sizes.y*strides_total.y)*sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

// Loads the specified index `range` of a file located at `path` on disk which is of the given `shape`, into a vector of type `T`.
template <typename T>
std::vector<T> load_file_strided(const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global) {
    std::vector<T> data(shape.z*shape.y*shape.x);
    load_file_strided(data.data(), path, disk_shape, shape, range, offset_global);
    return data;
}

template <typename T>
void load_partial(T *__restrict__ dst, FILE *fp, const int64_t offset, const int64_t n_elements) {
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fread((char *) dst, sizeof(T), n_elements, fp);
    assert(n == n_elements && "Failed to read all elements");
}

FILE* open_file_read(const std::string &path) {
    int fd = open(path.c_str(), O_RDONLY | O_DIRECT);
    return fdopen(fd, "rb");
}

FILE* open_file_write(const std::string &path) {
    int fd = open(path.c_str(), O_CREAT | O_RDWR | O_DIRECT, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
    return fdopen(fd, "r+b");
}

// Stores `data.size()` elements of `data` into a file located at `path` on disk at `offset` elements from the beginning of the file.
template <typename T>
void store_file(const std::vector<T> &data, const std::string &path, const int64_t offset) {
    std::ofstream file;
    file.open(path, std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        file.clear();
        file.open(path, std::ios::binary | std::ios::out);
    }
    file.seekp(offset*sizeof(T), std::ios::beg);
    file.write(reinterpret_cast<const char*>(data.data()), data.size()*sizeof(T));
    file.flush();
    file.close();
}

template <typename T>
void store_file(const T *data, const std::string &path, const int64_t offset, const int64_t n_elements) {
    // Open the file
    FILE *fp = open_file_write(path);

    // Calculate the aligned start and end positions
    int64_t
        start_pos = offset*sizeof(T),
        end_pos = (offset+n_elements)*sizeof(T),
        aligned_start = (start_pos / disk_block_size) * disk_block_size,
        aligned_end = ((end_pos + disk_block_size - 1) / disk_block_size) * disk_block_size,
        aligned_size = aligned_end - aligned_start,
        aligned_n_elements = aligned_size / sizeof(T);

    // Allocate a buffer for the write
    T *buffer = (T *) aligned_alloc(disk_block_size, aligned_size);

    // If the start is not aligned, read the first block
    if (start_pos != aligned_start) {
        // Read the first block
        fseek(fp, aligned_start, SEEK_SET);
        int64_t n = fread((char *) buffer, sizeof(T), disk_block_size, fp);
        assert (n == disk_block_size && "Failed to read all elements");
    }

    // If the end is not aligned, read the last block
    if (end_pos != aligned_end) {
        // Read the last block
        fseek(fp, aligned_end - disk_block_size, SEEK_SET);
        int64_t n = fread((char *) buffer + aligned_size - disk_block_size, sizeof(T), disk_block_size, fp);
        assert (n == disk_block_size && "Failed to read all elements");
    }

    // Copy the data to the buffer
    memcpy((char *) buffer + start_pos - aligned_start, (char *) data, n_elements*sizeof(T));

    // Write the buffer to disk
    fseek(fp, aligned_start, SEEK_SET);
    int64_t n = fwrite((char *) buffer, sizeof(T), aligned_n_elements, fp);
    assert (n == aligned_n_elements && "Failed to write all elements");

    // Free the buffer and close the file
    free(buffer);
    fclose(fp);
}

template <typename T>
void store_file_strided(const T *data, const std::string &path, const idx3d &shape_total, const idx3d &shape_global, const idx3drange &range, const idx3d &offset_global) {
    // Calculate the strides and sizes
    const idx3d
        strides_total = {shape_total.y*shape_total.x, shape_total.x, 1},
        strides_global = {shape_global.y*shape_global.x, shape_global.x, 1},
        sizes = {range.z_end - range.z_start, range.y_end - range.y_start, range.x_end - range.x_start};

    // If the shape on disk is the same as the shape in memory, just store the entire file
    if (shape_global.y == shape_total.y && shape_global.x == shape_total.x && offset_global.y == 0 && offset_global.x == 0 && range.y_start == 0 && range.x_start == 0 && range.y_end == shape_total.y && range.x_end == shape_total.x) {
        store_file(data + offset_global.z*strides_global.z, path, range.z_start*strides_total.z, sizes.z*strides_total.z);
        return;
    }

    assert (false && "Not implemented yet :) - After the deadline!");

    // Open the file
    FILE *fp = open_file_write(path);
    fseek(fp, (range.z_start*strides_total.z + range.y_start*strides_total.y + range.x_start*strides_total.x)*sizeof(T), SEEK_SET);
    for (int64_t z = 0; z < sizes.z; z++) {
        for (int64_t y = 0; y < sizes.y; y++) {
            int64_t n = fwrite((char *) &data[(z+offset_global.z)*strides_global.z + (y+offset_global.y)*strides_global.y + (0+offset_global.x)*strides_global.x], sizeof(T), sizes.x, fp);
            assert (n == sizes.x && "Failed to write all elements");
            fseek(fp, (strides_total.y - sizes.x)*sizeof(T), SEEK_CUR);
        }
        fseek(fp, (strides_total.z - sizes.y*strides_total.y) * sizeof(T), SEEK_CUR);
    }
    fclose(fp);
}

template <typename T>
void store_file_strided(const std::vector<T> &data, const std::string &path, const idx3d &disk_shape, const idx3d &shape, const idx3drange &range, const idx3d &offset_global) {
    store_file_strided(data.data(), path, disk_shape, shape, range, offset_global);
}

template <typename T>
void store_partial(const T *__restrict__ src, FILE *fp, const int64_t offset, const int64_t n_elements) {
    fseek(fp, offset*sizeof(T), SEEK_SET);
    int64_t n = fwrite((char *) src, sizeof(T), n_elements, fp);
    assert(n == n_elements && "Failed to write all elements");
}

namespace cpu_par {

void apply_renaming(std::vector<int64_t> &img, std::vector<int64_t> &to_rename) {
    NS::apply_renaming(img.data(), img.size(), to_rename);
}

void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename) {
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        if (img[i] < (int64_t) to_rename.size()) {
            img[i] = to_rename[img[i]];
        }
    }
}

void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape) {
    std::vector<bool> found(n_labels+1, false);
    const idx3d strides = { global_shape.y * global_shape.x, global_shape.x, 1 };
    int64_t n_chunks = total_shape.z / global_shape.z; // Assuming that they are divisible
    FILE *file = open_file_read(path);
    int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
    int64_t *img = (int64_t *) aligned_alloc(disk_block_size, chunk_size * sizeof(int64_t));
    for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
        std::cout << "Chunk " << chunk << " / " << n_chunks << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        load_partial(img, file, chunk*chunk_size, chunk_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "load_partial: " << (chunk_size*sizeof(int64_t)) / elapsed.count() / 1e9 << " GB/s" << std::endl;
        for (int64_t i = 0; i < chunk_size; i++) {
            int64_t label = img[i];
            if (label > n_labels || label < 0) {
                std::cout << "Label " << label << " in chunk " << chunk << " at index " << i <<" is outside of bounds 0:" << n_labels << std::endl;
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

int64_t connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &global_shape, const bool verbose) {
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

    std::vector<std::vector<int64_t>> renames(chunks); // Rename LUTs, one for each chunk
    for (int64_t i = 0; i < (int64_t) index_tree.size(); i++) {
        //#pragma omp parallel for
        for (int64_t j = 0; j < (int64_t) index_tree[i].size(); j++) {
            auto [l, r] = index_tree[i][j];
            // TODO Handle when all chunks doesn't have the same shape.
            int64_t last_layer = (global_shape.z-1) * global_strides.z;
            std::vector<int64_t> a = load_file<int64_t>(paths[l], last_layer, global_strides.z);
            std::vector<int64_t> b = load_file<int64_t>(paths[r], 0, global_strides.z);

            if (i > 0) {
                // Apply the renamings obtained from the previous layer
                apply_renaming(a, renames[l]);
                apply_renaming(b, renames[r]);
            }
            auto [rename_l, rename_r, n_new_labels] = NS::relabel(a, n_labels[l], b, n_labels[r], global_shape, verbose);
            n_labels[l] = n_new_labels;
            n_labels[r] = n_new_labels;

            if (i > 0) {
                // Run through the left subtree
                int64_t subtrees = i << 1;
                for (int64_t k = j*2*subtrees; k < (j*2*subtrees)+subtrees; k++) {
                    apply_renaming(renames[k], rename_l);
                    n_labels[k] = n_new_labels;
                }

                // Run through the right subtree
                for (int64_t k = (j*2*subtrees)+subtrees; k < (j*2*subtrees)+(2*subtrees); k++) {
                    apply_renaming(renames[k], rename_r);
                    n_labels[k] = n_new_labels;
                }
            } else {
                renames[l] = rename_l;
                renames[r] = rename_r;
            }
        }
    }

    auto cc_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cc = cc_end - cc_start;
    if (verbose) {
        std::cout << "connected_components lut building: " << elapsed_cc.count() << " s" << std::endl;
    }

    auto cc_app_start = std::chrono::high_resolution_clock::now();

    // Apply the renaming to a new global file
    std::string all_path = base_path + "all.int64";
    int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
    FILE *all_file = open_file_write(all_path);
    // TODO handle chunks % disk_block_size != 0
    int64_t *chunk = (int64_t *) aligned_alloc(disk_block_size, chunk_size * sizeof(int64_t));
    for (int64_t i = 0; i < chunks; i++) {
        auto load_start = std::chrono::high_resolution_clock::now();
        load_file(chunk, paths[i], 0, chunk_size);
        auto load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = load_end - load_start;
        if (verbose) {
            std::cout << "load_file: " << (chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
        }

        auto apply_start = std::chrono::high_resolution_clock::now();
        apply_renaming(chunk, chunk_size, renames[i]);
        auto apply_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
        if (verbose) {
            std::cout << "apply_renaming: " << (chunk_size*sizeof(int64_t)) / elapsed_apply.count() / 1e9 << " GB/s" << std::endl;
        }

        auto store_start = std::chrono::high_resolution_clock::now();
        store_partial(chunk, all_file, i*chunk_size, chunk_size);
        auto store_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_store = store_end - store_start;
        if (verbose) {
            std::cout << "store_partial: " << (chunk_size*sizeof(int64_t)) / elapsed_store.count() / 1e9 << " GB/s" << std::endl;
        }
    }
    free(chunk);
    fclose(all_file);

    auto cc_app_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cc_app = cc_app_end - cc_app_start;
    if (verbose) {
        std::cout << "connected_components lut application: " << elapsed_cc_app.count() << " s" << std::endl;
    }

    return n_labels[0];
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
    int64_t log_chunks = std::ceil(std::log2(chunks));
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

    // TODO is this actually necessary? We'll see.
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