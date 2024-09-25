
/**
 * @file connected_components.cc
 * Parallel CPU implementation of the connected components merging functions.
 */
#include "connected_components.hh"

#include <chrono>
#include "io.hh"
#include <iostream>
#include <omp.h>
#include <stack>
#include <unordered_set>

namespace cpu_par {

    void apply_renaming(std::vector<int64_t> &img, std::vector<int64_t> &to_rename) {
        NS::apply_renaming(img.data(), img.size(), to_rename);
    }

    void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < n; i++) {
            // TODO Make into a debug macro
            //assert (img[i] < (int64_t) to_rename.size() && "Label out of bounds");
            img[i] = to_rename[img[i]];
        }
    }

    int64_t apply_renamings(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &e_total_shape, const idx3d &e_global_shape, const std::vector<std::vector<int64_t>> &renames, const int verbose) {
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

        if (verbose >= 2) {
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
            if (verbose >= 2) {
                std::cout << "load_file: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
            }

            auto apply_start = std::chrono::high_resolution_clock::now();
            apply_renaming(chunk, e_this_chunk_size, renames[i]);
            auto apply_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
            if (verbose >= 2) {
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
            if (verbose >= 2) {
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
        if (verbose >= 2) {
            std::cout << "connected_components lut application: " << elapsed_cc_app.count() << " s" << std::endl;
        }

        return n_labels[0];
    }

    void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape, const int verbose) {
        std::vector<bool> found(n_labels+1, false);
        const idx3d strides = { global_shape.y * global_shape.x, global_shape.x, 1 };
        int64_t n_chunks = total_shape.z / global_shape.z; // Assuming that they are divisible
        FILE *file = open_file_read(path);
        int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
        int64_t *img = (int64_t *) aligned_alloc(b_disk_block_size, chunk_size * sizeof(int64_t));
        for (int64_t chunk = 0; chunk < n_chunks; chunk++) {
            if (verbose >= 2) {
                std::cout << "Chunk " << chunk << " / " << n_chunks << std::endl;
            }

            auto start = std::chrono::high_resolution_clock::now();
            load_flat(img, file, chunk*chunk_size, chunk_size);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            if (verbose >= 2) {
                std::cout << "load_partial: " << (double) (chunk_size*sizeof(int64_t)) / elapsed.count() / 1e9 << " GB/s" << std::endl;
            }

            for (int64_t i = 0; i < chunk_size; i++) {
                int64_t label = img[i];
                if (label > n_labels || label < 0) {
                    if (verbose >= 2) {
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

    std::vector<std::vector<int64_t>> connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &global_shape, const int verbose) {
        auto cc_start = std::chrono::high_resolution_clock::now();
        // Check if the call is well-formed
        int64_t n_chunks = n_labels.size();
        assert ((n_chunks & (n_chunks - 1)) == 0 && "Chunks must be a power of 2");

        // Constants
        const idx3d
            global_strides = { global_shape.y * global_shape.x, global_shape.x, 1 };

        // Generate the paths to the different chunks
        std::vector<std::string> paths(n_chunks);
        for (int64_t i = 0; i < n_chunks; i++) {
            paths[i] = base_path + std::to_string(i) + ".int64";
        }

        // Generate the adjacency tree
        std::vector<std::vector<std::tuple<int64_t, int64_t>>> index_tree = NS::generate_adjacency_tree(n_chunks, verbose);

        std::vector<std::vector<int64_t>> renames(n_chunks, std::vector<int64_t>());
        for (int64_t i = 0; i < n_chunks; i++) {
            renames[i].resize(n_labels[i]+1);
            for (int64_t j = 0; j < n_labels[i]+1; j++) {
                renames[i][j] = j;
            }
        }

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

                // Apply the renamings obtained from the previous layers
                apply_renaming(a, renames[l]);
                apply_renaming(b, renames[r]);

                for (size_t k = 0; k < a.size(); k++) {
                    assert (a[k] <= n_labels[l] && "Label out of bounds");
                }
                for (size_t k = 0; k < b.size(); k++) {
                    assert (b[k] <= n_labels[r] && "Label out of bounds");
                }

                auto [rename_l, rename_r, n_new_labels] = NS::relabel(a, n_labels[l], b, n_labels[r], global_shape, verbose);
                n_labels[l] = n_new_labels;
                n_labels[r] = n_new_labels;

                for (size_t k = 0; k < rename_l.size(); k++) {
                    assert (rename_l[k] >= 0 && rename_l[k] <= n_new_labels && "Label out of bounds");
                }
                for (size_t k = 0; k < rename_r.size(); k++) {
                    assert (rename_r[k] >= 0 && rename_r[k] <= n_new_labels && "Label out of bounds");
                }

                // Store the renamings
                int64_t subtrees = (int64_t) std::pow(2, i);

                // Run through the left subtree
                for (int64_t k = j*2*subtrees; k < (j*2*subtrees)+subtrees; k++) {
                    for (int64_t sub_l = 0; sub_l < (int64_t)renames[k].size(); sub_l++) {
                        renames[k][sub_l] = rename_l[renames[k][sub_l]];
                    }
                    n_labels[k] = n_new_labels;
                }

                // Run through the right subtree
                for (int64_t k = (j*2*subtrees)+subtrees; k < (j*2*subtrees)+(2*subtrees); k++) {
                    for (int64_t sub_r = 0; sub_r < (int64_t)renames[k].size(); sub_r++) {
                        renames[k][sub_r] = rename_r[renames[k][sub_r]];
                    }
                    n_labels[k] = n_new_labels;
                }
            }
        }

        auto cc_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_cc = cc_end - cc_start;
        if (verbose >= 2) {
            std::cout << "connected_components lut building: " << elapsed_cc.count() << " s" << std::endl;
        }

        return renames;
    }

    void count_sizes(int64_t *__restrict__ img, std::vector<int64_t> &sizes, const int64_t n_labels, const int64_t size) {
        std::vector<std::vector<int64_t>> sizes_thread(omp_get_max_threads(), std::vector<int64_t>(n_labels+1, 0));
        #pragma omp parallel
        {
            int64_t thread_id = omp_get_thread_num();
            std::vector<int64_t> local_sizes(n_labels+1, 0);

            #pragma omp for
            for (int64_t i = 0; i < size; i++) {
                local_sizes[img[i]]++;
            }
            std::copy(local_sizes.begin(), local_sizes.end(), sizes_thread[thread_id].begin());

            #pragma omp for
            for (int64_t i = 0; i < n_labels+1; i++) {
                for (int64_t j = 0; j < omp_get_num_threads(); j++) {
                    sizes[i] += sizes_thread[j][i];
                }
            }
        }
    }

    void filter_largest(const std::string &base_path, bool *__restrict__ mask, const std::vector<std::vector<int64_t>> &renames, const int64_t largest, const idx3d &e_total_shape, const idx3d &e_global_shape, const int verbose) {
        // Apply the renaming to a new global file
        int64_t
            e_global_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
            chunks = renames.size(),
            e_largest_chunk = e_total_shape.z - ((chunks-1) * e_global_shape.z),
            e_chunk_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
            e_largest_chunk_size = e_largest_chunk * e_global_shape.y * e_global_shape.x,
            b_largest_chunk_size = e_largest_chunk_size * sizeof(int64_t),
            b_aligned_chunk_size = ((b_largest_chunk_size + b_disk_block_size-1) / b_disk_block_size) * b_disk_block_size;

        if (verbose >= 2) {
            std::cout << "Largest chunk: " << e_largest_chunk << std::endl;
            std::cout << "Total shape: " << e_total_shape.z << " " << e_total_shape.y << " " << e_total_shape.x << std::endl;
            std::cout << "Global shape: " << e_global_shape.z << " " << e_global_shape.y << " " << e_global_shape.x << std::endl;
            std::cout << "Chunks: " << chunks << std::endl;
            std::cout << "Chunk size: " << e_chunk_size << std::endl;
            std::cout << "Largest chunk size: " << e_largest_chunk_size << std::endl;
        }

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

        if (verbose >= 2) {
            std::cout << "Image size: " << b_img_size << std::endl;
            std::cout << "Aligned image size: " << b_aligned_img_size << std::endl;
        }

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
            if (verbose >= 2) {
                std::cout << "load_file: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
            }

            auto apply_start = std::chrono::high_resolution_clock::now();
            apply_renaming(chunk, e_this_chunk_size, renames[i]);
            auto apply_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
            if (verbose >= 2) {
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
            #pragma omp parallel for schedule(static)
            for (int64_t j = 0; j < e_this_chunk_size; j++) {
                //assert (i*e_global_size + j < (e_total_shape.z * e_total_shape.y * e_total_shape.x) && "Index out of bounds");
                mask[i*e_global_size + j] = chunk[j] == largest;
            }
            auto filter_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_filter = filter_end - filter_start;
            if (verbose >= 2) {
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
        mapping_t mapping_a(n_labels_a+1);
        mapping_t mapping_b(n_labels_b+1);

        const int64_t plane_size = global_shape.y * global_shape.x;
        for (int64_t i = 0; i < plane_size; i++) {
            if (a[i] != 0 && b[i] != 0) {
                mapping_a[a[i]].insert(b[i]);
                mapping_b[b[i]].insert(a[i]);
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

    std::vector<std::vector<std::tuple<int64_t, int64_t>>> generate_adjacency_tree(const int64_t chunks, const int verbose) {
        int64_t log_chunks = (int64_t) std::ceil(std::log2(chunks));
        std::vector<std::vector<std::tuple<int64_t, int64_t>>> tree(log_chunks);

        if (verbose >= 2) {
            std::cout << "Generating adjacency tree with " << log_chunks << " layers." << std::endl;
        }

        for (int64_t layer = 0; layer < log_chunks; layer++) {
            int64_t n_elements = chunks >> (layer+1); // chunks / 2^layer
            int64_t i = 1 << layer; // 1 * 2^layer
            std::vector<std::tuple<int64_t, int64_t>> indices;
            for (int64_t j = i-1; j < i*n_elements*2; j += i*2) {
                indices.push_back({j, j+1});

            }
            tree[layer] = indices;
        }

        if (verbose >= 2) {
            std::cout << "First layer has " << tree[0].size() << " pairs." << std::endl;
        }

        return tree;
    }

    int64_t largest_component(const std::string &base_path, const std::vector<std::vector<int64_t>> &renames, const int64_t n_labels, const idx3d &e_total_shape, const idx3d &e_global_shape, const int verbose) {
        // Apply the renaming to a new global file
        int64_t
            chunks = renames.size(),
            e_largest_chunk = std::max(e_global_shape.z, (e_total_shape.z - (e_total_shape.z / e_global_shape.z) * e_global_shape.z) + e_global_shape.z),
            e_chunk_size = e_global_shape.z * e_global_shape.y * e_global_shape.x,
            e_largest_chunk_size = e_largest_chunk * e_global_shape.y * e_global_shape.x,
            b_largest_chunk_size = e_largest_chunk_size * sizeof(int64_t),
            b_aligned_chunk_size = ((b_largest_chunk_size + b_disk_block_size-1) / b_disk_block_size) * b_disk_block_size,
            e_aligned_chunk_size = b_aligned_chunk_size / sizeof(int64_t);

        if (verbose >= 2) {
            std::cout << "Largest chunk: " << e_largest_chunk << std::endl;
            std::cout << "Total shape: " << e_total_shape.z << " " << e_total_shape.y << " " << e_total_shape.x << std::endl;
            std::cout << "Global shape: " << e_global_shape.z << " " << e_global_shape.y << " " << e_global_shape.x << std::endl;
            std::cout << "Chunks: " << chunks << std::endl;
            std::cout << "Chunk size: " << e_chunk_size << std::endl;
            std::cout << "Largest chunk size: " << e_largest_chunk_size << std::endl;
        }

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
            if (verbose >= 2) {
                std::cout << "load_file: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_load.count() / 1e9 << " GB/s" << std::endl;
            }

            auto apply_start = std::chrono::high_resolution_clock::now();
            apply_renaming(chunk, e_this_chunk_size, renames[i]);
            auto apply_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_apply = apply_end - apply_start;
            if (verbose >= 2) {
                std::cout << "apply_renaming: " << (double) (e_this_chunk_size*sizeof(int64_t)) / elapsed_apply.count() / 1e9 << " GB/s" << std::endl;
            }

            auto sizes_start = std::chrono::high_resolution_clock::now();
            count_sizes(chunk, sizes, n_labels, e_this_chunk_size);
            auto sizes_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_sizes = sizes_end - sizes_start;
            if (verbose >= 2) {
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
        if (verbose >= 2) {
            std::cout << "max_element: " << (double) (n_labels*sizeof(int64_t)) / elapsed_largest.count() / 1e9 << " GB/s" << std::endl;
            std::cout << "Largest element is: " << largest << std::endl;
            std::cout << "It occurs " << sizes[largest] << " times" << std::endl;
        }

        return largest;
    }

    int64_t merge_labeled_chunks(int64_t *chunks, const int64_t n_chunks, int64_t *n_labels, const idx3d &global_shape, const int64_t total_z, const int verbose) {
        // Generate the adjacency tree
        auto adj_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<std::tuple<int64_t, int64_t>>> index_tree = NS::generate_adjacency_tree(n_chunks, verbose);
        auto adj_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_adj = adj_end - adj_start;

        if (verbose >= 2) {
            std::cout << "generate_adjacency_tree: " << elapsed_adj.count() << " s" << std::endl;
        }

        const int64_t chunk_size = global_shape.z * global_shape.y * global_shape.x;
        const idx3d global_strides = { global_shape.y * global_shape.x, global_shape.x, 1 };

        std::vector<std::vector<int64_t>> renames(n_chunks, std::vector<int64_t>());
        for (int64_t i = 0; i < n_chunks; i++) {
            renames[i].resize(n_labels[i]+1);
            for (int64_t j = 0; j < n_labels[i]+1; j++) {
                renames[i][j] = j;
            }
        }

        // Rename LUTs, one for each chunk
        for (int64_t i = 0; i < (int64_t) index_tree.size(); i++) {
            #pragma omp parallel for
            for (int64_t j = 0; j < (int64_t) index_tree[i].size(); j++) {
                auto chunk_start = std::chrono::high_resolution_clock::now();
                auto [l, r] = index_tree[i][j];
                // This doesn't handle the different chunk sizes, but it should be fine as the last chunk is the only one that differs and we only read the first layer from that one
                int64_t last_layer = (global_shape.z-1) * global_strides.z;
                std::vector<int64_t> a(chunks + (l*chunk_size) + last_layer, chunks + (l*chunk_size) + last_layer + global_strides.z);
                std::vector<int64_t> b(chunks + (r*chunk_size), chunks + (r*chunk_size) + global_strides.z);
                auto chunk_init = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_chunk_init = chunk_init - chunk_start;
                if (verbose >= 2) {
                    std::cout << "chunk_init: " << elapsed_chunk_init.count() << " s" << std::endl;
                }

                // Apply the renamings obtained from the previous layers
                apply_renaming(a, renames[l]);
                apply_renaming(b, renames[r]);

                // TODO Make into a debug macro
                for (size_t k = 0; k < a.size(); k++) {
                    assert (a[k] >= 0 && a[k] <= n_labels[l] && "Label out of bounds");
                }
                for (size_t k = 0; k < b.size(); k++) {
                    assert (b[k] >= 0 && b[k] <= n_labels[r] && "Label out of bounds");
                }

                auto chunk_apply = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_chunk_apply = chunk_apply - chunk_init;
                if (verbose >= 2) {
                    std::cout << "chunk_apply: " << elapsed_chunk_apply.count() << " s" << std::endl;
                }

                auto [rename_l, rename_r, n_new_labels] = NS::relabel(a, n_labels[l], b, n_labels[r], global_shape, verbose);
                auto chunk_relabel = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_chunk_relabel = chunk_relabel - chunk_apply;
                if (verbose >= 2) {
                    std::cout << "chunk_relabel: " << elapsed_chunk_relabel.count() << " s" << std::endl;
                }
                n_labels[l] = n_new_labels;
                n_labels[r] = n_new_labels;
                if (verbose >= 2) {
                    std::cout << "Found " << n_new_labels << " new labels." << std::endl;
                }

                // Store the renamings
                int64_t subtrees = (int64_t) std::pow(2, i);

                // Run through the left subtree
                for (int64_t k = j*2*subtrees; k < (j*2*subtrees)+subtrees; k++) {
                    for (int64_t sub_l = 0; sub_l < (int64_t)renames[k].size(); sub_l++) {
                        renames[k][sub_l] = rename_l[renames[k][sub_l]];
                    }
                    n_labels[k] = n_new_labels;
                }

                // Run through the right subtree
                for (int64_t k = (j*2*subtrees)+subtrees; k < (j*2*subtrees)+(2*subtrees); k++) {
                    for (int64_t sub_r = 0; sub_r < (int64_t)renames[k].size(); sub_r++) {
                        renames[k][sub_r] = rename_r[renames[k][sub_r]];
                    }
                    n_labels[k] = n_new_labels;
                }

                auto chunk_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_store = chunk_end - chunk_relabel;
                if (verbose >= 2) {
                    std::cout << "chunk_store: " << elapsed_store.count() << " s" << std::endl;
                }
                std::chrono::duration<double> elapsed_chunk = chunk_end - chunk_start;
                if (verbose >= 2) {
                    std::cout << "chunk_total: " << elapsed_chunk.count() << " s" << std::endl;
                }
            }
        }

        for (int64_t i = 0; i < n_chunks; i++) {
            int64_t this_chunk_size = std::min(chunk_size, (total_z - (i * global_shape.z)) * global_shape.y * global_shape.x);
            apply_renaming(chunks + (i*chunk_size), this_chunk_size, renames[i]);
        }

        return n_labels[std::get<0>(index_tree[index_tree.size()-1][0])];
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

    int64_t recount_labels(std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b, int64_t max_label) {
        // Find the labels that are actually used
        std::unordered_set<int64_t> mapped_a;
        for (int64_t i = 1; i < (int64_t) to_rename_a.size(); i++) {
            if (to_rename_a[i] != 0) {
                mapped_a.insert(to_rename_a[i]);
            }
        }

        // Assign the first mapped_a labels to start from 1. These labels should be the same in both to_rename_a and to_rename_b
        std::vector<int64_t> new_rename(max_label+1, -1); // -1 to capture faulty values
        int64_t next = 1;
        for (int64_t label : mapped_a) {
            new_rename[label] = next++;
        }

        // Apply the new renaming to the renaming LUTs
        for (int64_t i = 1; i < (int64_t) to_rename_a.size(); i++) {
            to_rename_a[i] = new_rename[to_rename_a[i]];
        }
        for (int64_t i = 1; i < (int64_t) to_rename_b.size(); i++) {
            to_rename_b[i] = new_rename[to_rename_b[i]];
        }

        // Assign the rest of the unmapped labels in both LUTs
        to_rename_a[0] = 0;
        for (int64_t i = 1; i < (int64_t) to_rename_a.size(); i++) {
            if (to_rename_a[i] == -1) {
                to_rename_a[i] = next++;
            }
        }
        to_rename_b[0] = 0;
        for (int64_t i = 1; i < (int64_t) to_rename_b.size(); i++) {
            if (to_rename_b[i] == -1) {
                to_rename_b[i] = next++;
            }
        }

        return next-1;
    }

    std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape, const int verbose) {
        mapping_t mapping_a(n_labels_a+1);
        mapping_t mapping_b(n_labels_b+1);
        std::vector<int64_t> rename_a(n_labels_a+1, 0);
        std::vector<int64_t> rename_b(n_labels_b+1, 0);

        // TODO parallel where applicable? Do something with scan to get new global labels?

        if (verbose >= 2) {
            std::cout << "Relabeling" << std::endl;
        }

        int64_t next = 0;
        int64_t last_a = 0, last_b = 0;
        for (int64_t y = 0; y < global_shape.y; y++) {
            for (int64_t x = 0; x < global_shape.x; x++) {
                int64_t va = a[y*global_shape.x + x];
                int64_t vb = b[y*global_shape.x + x];

                if (va && vb) {
                    mapping_a[va].insert(vb); // Before as it should be included in the entry loop
                    if (rename_a[va]) {
                        int64_t lab = rename_a[va];
                        rename_b[vb] = lab;
                        if (rename_b[vb]) {
                            for (int64_t entry : mapping_a[va]) {
                                rename_b[entry] = lab;
                                for (int64_t entry2 : mapping_b[entry]) {
                                    rename_a[entry2] = lab;
                                }
                            }
                        }
                    } else if (rename_b[vb]) {
                        rename_a[va] = rename_b[vb];
                    } else {
                        rename_a[va] = rename_b[vb] = ++next;
                    }
                    mapping_b[vb].insert(va); // After as it shouldn't be included in the entry2 loop
                } else {
                    if (va && !rename_a[va]) {
                        if (last_a) {
                            rename_a[va] = next;
                        } else {
                            rename_a[va] = ++next;
                        }
                    }
                    if (vb && !rename_b[vb]) {
                        if (last_b) {
                            rename_b[vb] = next;
                        } else {
                            rename_b[vb] = ++next;
                        }
                    }
                }

                if (x == global_shape.x-1) {
                    last_a = 0;
                    last_b = 0;
                } else {
                    last_a = va;
                    last_b = vb;
                }
            }
        }

        // Ensure the labels are consecutive
        int64_t n_new_labels = recount_labels(rename_a, rename_b, next);

        assert (n_new_labels <= n_labels_a + n_labels_b && "New labels exceed the sum of the old labels");

        return { rename_a, rename_b, n_new_labels };
    }

} // namespace cpu