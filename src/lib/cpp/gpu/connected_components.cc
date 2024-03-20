#include "connected_components.hh"

namespace gpu {

#pragma GCC diagnostic ignored "-Wunused-parameter"
int64_t apply_renamings(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &total_shape, const idx3d &global_shape, const std::vector<std::vector<int64_t>> &renames, const bool verbose) {
    throw std::runtime_error("Not implemented");
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
std::vector<std::vector<int64_t>> connected_components(const std::string &base_path, std::vector<int64_t> &n_labels, const idx3d &total_shape, const idx3d &global_shape, const bool verbose) {
    throw std::runtime_error("Not implemented");
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
void filter_largest(const std::string &base_path, bool *__restrict__ mask, const std::vector<std::vector<int64_t>> &renames, const int64_t largest, const idx3d &total_shape, const idx3d &global_shape, const bool verbose) {
    throw std::runtime_error("Not implemented");
}

#pragma GCC diagnostic ignored "-Wunused-parameter"
int64_t largest_component(const std::string &base_path, const std::vector<std::vector<int64_t>> &renames, const int64_t n_labels, const idx3d &total_shape, const idx3d &global_shape, const bool verbose) {
    throw std::runtime_error("Not implemented");
}

}