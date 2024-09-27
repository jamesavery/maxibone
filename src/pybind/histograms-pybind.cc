/**
 * @file histograms-pybind.cc
 * @author Carl-Johannes Johnsen (carl-johannes@di.ku.dk)
 * @brief Python bindings for the 2D histogram functions.
 * @version 0.1
 * @date 2024-09-17
 *
 * @copyright Copyright (c) 2024
 */
#include "histograms.cc"

namespace python_api {

    /**
     * Computes four 2D histograms of the voxels.
     * A 2D histogram is where each row is a histogram of the voxels in a slice of the 3D volume.
     * E.g. for the x-axis, a histogram is computed for each slice in the x-direction.
     * Assumes that the `*_bins` are pre-allocated and zeroed.
     *
     * @param np_voxels The 3D volume of voxels.
     * @param np_offset The global offset of the block. Used for out-of-core processing.
     * @param np_x_bins The output array for the x-axis histograms.
     * @param np_y_bins The output array for the y-axis histograms.
     * @param np_z_bins The output array for the z-axis histograms.
     * @param np_r_bins The output array for the radial histograms.
     * @param center The YX center of the volume. Used for computing the radial position of each voxel for the radial histogram.
     * @param vrange The value range of the voxels.
     * @param verbose The verbosity level. Default is 0.
     */
    void axis_histograms(const np_array<voxel_type> np_voxels,
                         const std::tuple<int64_t,int64_t,int64_t> np_offset,
                         np_array<uint64_t> &np_x_bins,
                         np_array<uint64_t> &np_y_bins,
                         np_array<uint64_t> &np_z_bins,
                         np_array<uint64_t> &np_r_bins,
                         const std::tuple<uint64_t, uint64_t> center,
                         const std::tuple<double, double> vrange,
                         const int verbose = 0) {

        py::buffer_info
            voxels_info = np_voxels.request(),
            x_info = np_x_bins.request(),
            y_info = np_y_bins.request(),
            z_info = np_z_bins.request(),
            r_info = np_r_bins.request();

        shape_t
            global_shape = { z_info.shape[0], y_info.shape[0], x_info.shape[0] },
            voxels_shape = { voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2] },
            offset = { std::get<0>(np_offset), std::get<1>(np_offset), std::get<2>(np_offset) };

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

    /**
     * Computes a 2D field histogram.
     * A field histogram is where for each `v = voxel[z,y,x]` value and corresponding `f = field[z,y,x]` value, index `i,j` in the resulting histogram is incremented, where `i` is the field bin, and `j` is the voxel bin.
     * If the field shape is less than the voxel shape, the field is upsampled to the voxel shape.
     *
     * @param np_voxels The 3D volume of voxels.
     * @param np_field The 3D volume of fields.
     * @param np_offset The global offset of the block. Used for out-of-core processing.
     * @param np_bins The output array for the histogram.
     * @param vrange The value range of the voxels.
     * @param frange The value range of the fields.
     * @param verbose The verbosity level. Default is 0.
     */
    void field_histogram(const np_array<voxel_type> &np_voxels,
                         const np_array<field_type> &np_field,
                         const std::tuple<int64_t,int64_t,int64_t> np_offset,
                         np_array<uint64_t> &np_bins,
                         const std::tuple<double, double> vrange,
                         const std::tuple<double, double> frange,
                         const int verbose) {
        py::buffer_info
            voxels_info = np_voxels.request(),
            field_info = np_field.request(),
            bins_info = np_bins.request();

        shape_t
            voxels_shape = { voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2] },
            field_shape = { field_info.shape[0], field_info.shape[1], field_info.shape[2] },
            offset = { std::get<0>(np_offset), std::get<1>(np_offset), std::get<2>(np_offset) };

        const uint64_t
            voxel_bins = bins_info.shape[1],
            field_bins = bins_info.shape[0];

        const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
        const field_type *field = static_cast<field_type*>(field_info.ptr);
        uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

        NS::field_histogram(voxels, field, voxels_shape, field_shape, offset, voxels_shape, bins, voxel_bins, field_bins, vrange, frange, verbose);
    }

    /**
     * Computes a 2D field histogram.
     * A field histogram is where for each `v = voxel[z,y,x]` value and corresponding `f = field[z,y,x]` value, index `i,j` in the resulting histogram is incremented, where `i` is the field bin, and `j` is the voxel bin.
     * This function differs from `field_histogram` in that it resamples the field to the voxel shape using the neighbouring field values, rather than just choosing the closest.
     *
     * @param np_voxels The 3D volume of voxels.
     * @param np_field The 3D volume of fields.
     * @param np_offset The global offset of the block. Used for out-of-core processing.
     * @param np_block_size The size of the block. Used for out-of-core processing.
     * @param np_bins The output array for the histogram.
     * @param vrange The value range of the voxels.
     * @param frange The value range of the fields.
     * @param verbose The verbosity level. Default is 0.
     */
    void field_histogram_resample(const np_array<voxel_type> np_voxels,
                        const np_array<field_type> np_field,
                        const std::tuple<int64_t,int64_t,int64_t> np_offset,
                        const std::tuple<int64_t,int64_t,int64_t> np_block_size,
                        np_array<uint64_t> &np_bins,
                        const std::tuple<double,double> vrange,
                        const std::tuple<double,double> frange,
                        const int verbose) {
        py::buffer_info
            voxels_info = np_voxels.request(),
            field_info = np_field.request(),
            bins_info = np_bins.request();

        shape_t
            offset = { std::get<0>(np_offset), std::get<1>(np_offset), std::get<2>(np_offset) },
            voxels_shape = { voxels_info.shape[0], voxels_info.shape[1], voxels_info.shape[2] },
            field_shape = { field_info.shape[0], field_info.shape[1], field_info.shape[2] },
            block_size = { std::get<0>(np_block_size), std::get<1>(np_block_size), std::get<2>(np_block_size) };

        uint64_t
            voxel_bins = bins_info.shape[1],
            field_bins = bins_info.shape[0];

        const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);
        const field_type *field = static_cast<field_type*>(field_info.ptr);
        uint64_t *bins = static_cast<uint64_t*>(bins_info.ptr);

        NS::field_histogram_resample(voxels, field, voxels_shape, field_shape, offset, block_size, bins, voxel_bins, field_bins, vrange, frange, verbose);
    }

}

PYBIND11_MODULE(histograms, m) {
    m.doc() = "Functions for generating 2D histograms of either axis' or fields."; // optional module docstring

    m.def("axis_histograms", &python_api::axis_histograms,
        py::arg("np_voxels"),
        py::arg("np_offset"),
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
        py::arg("np_bins").noconvert(),
        py::arg("vrange"),
        py::arg("frange"),
        py::arg("verbose"));
    m.def("field_histogram_resample", &python_api::field_histogram_resample,
        py::arg("np_voxels"),
        py::arg("np_field"),
        py::arg("np_offset"),
        py::arg("np_block_size"),
        py::arg("np_bins").noconvert(),
        py::arg("vrange"),
        py::arg("frange"),
        py::arg("verbose"));
}