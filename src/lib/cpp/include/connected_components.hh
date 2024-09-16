/**
 * @file connected_components.hh
 * Header file for the connected components functions.
 */
#ifndef connected_components_h
#define connected_components_h

#include "datatypes.hh"

namespace NS {

    //
    // External Functions
    //
    /**
     * Apply the renaming LUT in `renames` to the labeled chunks in `base_path`.
     * The number of chunks is given by the length of `n_labels`.
     *
     * @param base_path The path to the directory and prefix of the labeled chunks.
     * @param n_labels The number of labels in each chunk.
     * @param total_shape The shape of the total volume.
     * @param global_shape The shape of a chunk.
     * @param renames The renaming LUT.
     * @param verbose Whether to print debug information.
     * @return The number of labels in the new volume.
     */
    int64_t apply_renamings(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &total_shape, const idx3d &global_shape, const std::vector<std::vector<int64_t>> &renames, const bool verbose);

    /**
     * Compute the connected components of the labeled chunks in `base_path`, essentially merging the labels so that there exist a global labeling across all chunks.
     * The number of chunks is given by the length of `n_labels`.
     *
     * @param base_path The path to the directory and prefix of the labeled chunks.
     * @param n_labels The number of labels in each chunk.
     * @param global_shape The shape of a chunk.
     * @param verbose Whether to print debug information.
     */
    std::vector<std::vector<int64_t>> connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &global_shape, const bool verbose);

    /**
     * Find the largest connected component in the labeled chunks in `base_path`.
     * The number of chunks is given by the length of `n_labels`.
     * The largest connected component is written to `mask`.
     * `mask` is assumed to be preallocated and have shape `total_shape`.
     *
     * @param base_path The path to the directory and prefix of the labeled chunks.
     * @param mask The mask to write the largest connected component to.
     * @param n_labels The number of labels in each chunk.
     * @param total_shape The shape of the total volume.
     * @param global_shape The shape of a chunk.
     * @param verbose Whether to print debug information.
     */
    void filter_largest(const std::string &base_path, bool *__restrict__ mask, const std::vector<std::vector<int64_t>> &renames, const int64_t largest, const idx3d &total_shape, const idx3d &global_shape, const bool verbose);

    /**
     * Find the largest connected component in the labeled chunks in `base_path`.
     * The number of chunks is given by the length of `n_labels`.
     *
     * @param base_path The path to the directory and prefix of the labeled chunks.
     * @param n_labels The number of labels in each chunk.
     * @param total_shape The shape of the total volume.
     * @param global_shape The shape of a chunk.
     * @param verbose Whether to print debug information.
     */
    int64_t largest_component(const std::string &base_path, const std::vector<std::vector<int64_t>> &renames, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape, const bool verbose = false);

    /**
     * Merge the labeled chunks in `chunks` to have a global labeling across all chunks.
     * This is essentially the in-memory version of `connected_components`.
     *
     * @param chunks Pointers to the labeled chunks to merge.
     * @param n_chunks The number of chunks.
     * @param n_labels The number of labels in each chunk.
     * @param global_shape The shape of a chunk.
     * @param total_z The total number of z-slices in the volume, i.e. across all chunks.
     * @param verbose Whether to print debug information.
     */
    int64_t merge_labeled_chunks(int64_t *chunks, const int64_t n_chunks, int64_t *n_labels, const idx3d &global_shape, const int64_t total_z, const bool verbose);

    //
    // Internal Functions
    //

    /**
     * Apply the renaming LUT in `renames` to the given chunk `img`.
     *
     * @param img The chunk to apply the renaming to.
     * @param to_rename The renaming LUT.
     */
    void apply_renaming(std::vector<int64_t> &img, const std::vector<int64_t> &to_rename);

    /**
     * Apply the renaming LUT in `renames` to the given chunk `img`.
     *
     * @param img The chunk to apply the renaming to.
     * @param n The number of elements in the chunk.
     * @param to_rename The renaming LUT.
     */
    void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename);

    /**
     * Find the canonical names and sizes of the labels in the collective volume stored at `path`.
     * It is assumed that the collective volume can be divided into chunks of shape `global_shape`.
     * A canonical name is the smallest 3D coordinate of a given label and the size is the number of voxels that has that label.
     * The canonical names and sizes are written to `out`.
     *
     * @param path The path to the volume.
     * @param out The output buffer for the canonical names and sizes.
     * @param n_labels The number of labels in each chunk.
     * @param total_shape The shape of the total volume.
     * @param global_shape The shape of a chunk.
     */
    void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape);

    /**
     * Get the mappings between a vector `a` and a vector `b`. A mapping from `a` to `b` describes which labels in `b` that are neighbours to a given label in `a`.
     * Both the mappings from `a` to `b` and from `b` to `a` are returned as a tuple.
     *
     * @param a The first vector.
     * @param n_labels_a The number of labels in `a`.
     * @param b The second vector.
     * @param n_labels_b The number of labels in `b`.
     * @param global_shape The shape of a chunk.
     * @return A tuple containing the mappings from `a` to `b` and from `b` to `a`.
     */
    std::tuple<mapping_t, mapping_t> get_mappings(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape);

    /**
     * Get the sizes of each label in `img`.
     *
     * @param img The chunk to get the sizes from.
     * @param n_labels The number of labels in the chunk.
     * @return A vector containing the sizes of each label.
     */
    std::vector<int64_t> get_sizes(const std::vector<int64_t> &img, const int64_t n_labels);

    /**
     * Generate the adjacency tree of the chunks.
     * The adjacency tree is a vector of vectors of tuples. Each tuple contains the indices of the chunks to compare at that level of the tree.
     * The tree is generated by recursively merging the chunks in pairs.
     * The first layer will have n_chunks // 2 pairs, the second layer will have n_chunks // 4 pairs, and so on until the final layer, which will have one pair.
     *
     * @param chunks The number of chunks. Must be a power of 2.
     * @return The adjacency tree.
     */
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> generate_adjacency_tree(const int64_t chunks);

    /**
     * Ensure that the labels in the renaming LUTs are consecutive.
     *
     * @param mapping_a The mapping from the first chunk to the second chunk.
     * @param mapping_b The mapping from the second chunk to the first chunk.
     * @param to_rename_a The renaming LUT for the first chunk.
     * @param to_rename_b The renaming LUT for the second chunk.
     * @return The number of labels after applying each LUT to their respective chunk.
     */
    int64_t recount_labels(const mapping_t &mapping_a, mapping_t &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b);

    /**
     * Build the renaming LUTs for each chunk. The renaming LUTs are built by merging the labels in the chunks so that there exist a global labeling across all chunks.
     *
     * @param a The first chunk.
     * @param n_labels_a The number of labels in the first chunk.
     * @param b The second chunk.
     * @param n_labels_b The number of labels in the second chunk.
     * @param global_shape The shape of a chunk.
     * @param verbose Whether to print debug information.
     * @return A tuple containing the renaming LUTs for each chunk and the number of labels in the new volume.
     */
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape, const bool verbose);

    //
    // Debugging functions
    //
    /**
     * Debug function for printing the canonical names of the labels in `names_a`.
     *
     * @param names_a The canonical names to print.
     */
    void print_canonical_names(const std::vector<idx3d> &names_a);

    /**
     * Debug function for printing the mappings between the labels in `mapping_`.
     *
     * @param mapping_ The mappings to print.
     */
    void print_mapping(const mapping_t &mapping_);

    /**
     * Debug function for printing the renaming LUT `to_rename`.
     *
     * @param to_rename The renaming LUT to print.
     */
    void print_rename(const std::vector<int64_t> &to_rename);

}

#endif // connected_components_h