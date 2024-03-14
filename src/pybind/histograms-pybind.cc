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
            global_shape = { z_info.shape[0], y_info.shape[0], x_info.shape[0] },
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

        NS::axis_histogram(voxels, global_shape, offset, voxels_shape,
                           x_bins, y_bins, z_bins, r_bins,
                           voxel_bins, Nr, center, vrange, verbose);
    }

    void field_histogram(const np_array<voxel_type> &np_voxels,
                         const np_array<field_type> &np_field,
                         const tuple<int64_t,int64_t,int64_t> np_offset,
                         const tuple<int64_t,int64_t,int64_t> np_block_size,
                         np_array<uint64_t> &np_bins,
                         const tuple<double, double> vrange,
                         const tuple<double, double> frange) {
        py::buffer_info
            voxels_info = np_voxels.request(),
            field_info = np_field.request(),
            bins_info = np_bins.request();

        shape_t
            voxels_shape = { voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2] },
            field_shape = { field_info.shape[0], field_info.shape[1], field_info.shape[2] },
            offset = { get<0>(np_offset), get<1>(np_offset), get<2>(np_offset) },
            block_size = { get<0>(np_block_size), get<1>(np_block_size), get<2>(np_block_size) };

        const uint64_t
            voxel_bins = bins_info.shape[1],
            field_bins = bins_info.shape[0];

        const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
        const field_type *field = static_cast<field_type*>(field_info.ptr);
        uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

        NS::field_histogram(voxels, field, voxels_shape, field_shape, offset, block_size, bins, voxel_bins, field_bins, vrange, frange);
    }

    pair<int,int> masked_minmax(const np_array<voxel_type> np_voxels) {
        // Extract NumPy array basearray-pointer and length
        auto voxels_info    = np_voxels.request();
        size_t image_length = voxels_info.size;

        const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);

        voxel_type
            voxel_min = max(voxels[0], voxel_type(1)),
            voxel_max = voxels[0];

        #pragma omp parallel for reduction(min:voxel_min) reduction(max:voxel_max)
        for (size_t i = 0; i < image_length; i++) {
            voxel_min = min(voxel_min, voxels[i] > 0 ? voxels[i] : voxel_type(1));
            voxel_max = max(voxel_max, voxels[i]);
        }

        assert(voxel_min > 0);
        return make_pair(voxel_min,voxel_max);
    }

    void field_histogram_resample(const np_array<voxel_type> np_voxels,
                        const np_array<field_type> np_field,
                        const std::tuple<int64_t,int64_t,int64_t> np_offset,
                        const std::tuple<int64_t,int64_t,int64_t> np_block_size,
                        np_array<uint64_t> &np_bins,
                        const std::tuple<double,double> vrange,
                        const std::tuple<double,double> frange) {
        py::buffer_info
            voxels_info = np_voxels.request(),
            field_info = np_field.request(),
            bins_info = np_bins.request();

        shape_t
            offset = { get<0>(np_offset), get<1>(np_offset), get<2>(np_offset) },
            voxels_shape = { voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2] },
            field_shape = { field_info.shape[0], field_info.shape[1], field_info.shape[2] },
            block_size = { get<0>(np_block_size), get<1>(np_block_size), get<2>(np_block_size) };

        uint64_t
            voxel_bins = bins_info.shape[1],
            field_bins = bins_info.shape[0];

        const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
        const field_type *field = static_cast<field_type*>(field_info.ptr);
        uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

        NS::field_histogram_resample(voxels, field, voxels_shape, field_shape, offset, block_size, bins, voxel_bins, field_bins, vrange, frange);
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
    m.def("field_histogram", &python_api::field_histogram,
        py::arg("np_voxels"),
        py::arg("np_field"),
        py::arg("np_offset"),
        py::arg("np_block_size"),
        py::arg("np_bins").noconvert(),
        py::arg("vrange"),
        py::arg("frange"));
    m.def("field_histogram_resample", &python_api::field_histogram_resample,
        py::arg("np_voxels"),
        py::arg("np_field"),
        py::arg("np_offset"),
        py::arg("np_block_size"),
        py::arg("np_bins").noconvert(),
        py::arg("vrange"),
        py::arg("frange"));
    m.def("masked_minmax", &python_api::masked_minmax,
        py::arg("np_voxels"));
}