#ifndef connected_components_h
#define connected_components_h

#include "datatypes.hh"

namespace NS {

    // External Functions
    int64_t apply_renamings(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &total_shape, const idx3d &global_shape, const std::vector<std::vector<int64_t>> &renames, const bool verbose);
    std::vector<std::vector<int64_t>> connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &total_shape, const idx3d &global_shape, const bool verbose);
    void filter_largest(const std::string &base_path, bool *__restrict__ mask, const std::vector<std::vector<int64_t>> &renames, const int64_t largest, const idx3d &total_shape, const idx3d &global_shape, const bool verbose);
    int64_t largest_component(const std::string &base_path, const std::vector<std::vector<int64_t>> &renames, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape, const bool verbose = false);

    // Internal Functions
    void apply_renaming(std::vector<int64_t> &img, const std::vector<int64_t> &to_rename);
    void apply_renaming(int64_t *__restrict__ img, const int64_t n, const std::vector<int64_t> &to_rename);
    void canonical_names_and_size(const std::string &path, int64_t *__restrict__ out, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape);
    std::tuple<mapping_t, mapping_t> get_mappings(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape);
    std::vector<int64_t> get_sizes(const std::vector<int64_t> &img, const int64_t n_labels);
    std::vector<std::vector<std::tuple<int64_t, int64_t>>> generate_adjacency_tree(const int64_t chunks);
    std::vector<idx3d> merge_canonical_names(const std::vector<idx3d> &names_a, const std::vector<idx3d> &names_b);
    std::vector<int64_t> merge_labels(mapping_t &mapping_a, const mapping_t &mapping_b, const std::vector<int64_t> &to_rename_b);
    int64_t recount_labels(const mapping_t &mapping_a, mapping_t &mapping_b, std::vector<int64_t> &to_rename_a, std::vector<int64_t> &to_rename_b);
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, int64_t> relabel(const std::vector<int64_t> &a, const int64_t n_labels_a, const std::vector<int64_t> &b, const int64_t n_labels_b, const idx3d &global_shape, const bool verbose);
    void rename_mapping(mapping_t &mapping_a, const std::vector<int64_t> &to_rename_other);

    // Debugging functions
    void print_canonical_names(const std::vector<idx3d> &names_a);
    void print_mapping(const mapping_t &mapping_);
    void print_rename(const std::vector<int64_t> &to_rename);

}

#endif // connected_components_h