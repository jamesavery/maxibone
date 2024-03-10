#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace std;
namespace py = pybind11;

#include "histograms.cc"

namespace python_api {

    void axis_histograms(const np_array<voxel_type> np_voxels,
                         const tuple<int64_t,int64_t,int64_t> np_offset,
                         const tuple<int64_t,int64_t,int64_t> np_block_size,
                         np_array<uint64_t> &np_x_bins,
                         np_array<uint64_t> &np_y_bins,
                         np_array<uint64_t> &np_z_bins,
                         np_array<uint64_t> &np_r_bins,
                         const tuple<uint64_t, uint64_t> center,
                         const tuple<double, double> vrange,
                         const bool verbose) {

        py::buffer_info
            voxels_info = np_voxels.request(),
            x_info = np_x_bins.request(),
            y_info = np_y_bins.request(),
            z_info = np_z_bins.request(),
            r_info = np_r_bins.request();

        shape_t
            voxels_shape = { voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2] },
            offset = { get<0>(np_offset), get<1>(np_offset), get<2>(np_offset) },
            block_size = { get<0>(np_block_size), get<1>(np_block_size), get<2>(np_block_size) };

        const uint64_t
            voxel_bins = x_info.shape[1],
            Nr = r_info.shape[0];

        const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
        uint64_t
            *x_bins = static_cast<uint64_t*>(x_info.ptr),
            *y_bins = static_cast<uint64_t*>(y_info.ptr),
            *z_bins = static_cast<uint64_t*>(z_info.ptr),
            *r_bins = static_cast<uint64_t*>(r_info.ptr);

        NS::axis_histogram(voxels, voxels_shape, offset, block_size,
                           x_bins, y_bins, z_bins, r_bins,
                           voxel_bins, Nr, center, vrange, verbose);
    }

}

PYBIND11_MODULE(histograms, m) {
    m.doc() = "Functions for generating 2D histograms of either axis' or fields."; // optional module docstring

    m.def("axis_histograms", &python_api::axis_histograms,
        py::arg("np_voxels"),
        py::arg("np_offset"),
        py::arg("np_block_size"),
        py::arg("np_x_bins").noconvert(),
        py::arg("np_y_bins").noconvert(),
        py::arg("np_z_bins").noconvert(),
        py::arg("np_r_bins").noconvert(),
        py::arg("center"),
        py::arg("vrange"),
        py::arg("verbose"));
}