#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
namespace py = pybind11;

typedef uint8_t  result_type;
typedef uint16_t voxel_type;
typedef float_t  prob_type;
typedef uint16_t field_type;

void material_prob(const py::array_t<voxel_type> np_voxels,
                    const py::array_t<prob_type> &np_Pxs,
                    const py::array_t<prob_type> &np_Pys,
                    const py::array_t<prob_type> &np_Pzs,
                    const py::array_t<prob_type> &np_Prs,
                    const py::array_t<prob_type> &np_Pfield,
                    const py::array_t<field_type> &np_field,
                    py::array_t<result_type> &np_result,
                    const py::array_t<float> &np_vrange,
                    const py::array_t<uint64_t> &np_ranges) {

    py::buffer_info
        voxels_info = np_voxels.request(),
        x_info = np_Pxs.request(),
        y_info = np_Pys.request(),
        z_info = np_Pzs.request(),
        r_info = np_Prs.request(),
        Pfield_info = np_Pfield.request(),
        field_info = np_field.request(),
        result_info = np_result.request(),
        vrange_info = np_vrange.request(),
        ranges_info = np_ranges.request();
    
    const uint64_t
        Nvoxel_bins = x_info.shape[1],
        Nx = x_info.shape[0],
        Ny = y_info.shape[0],
        Nz = z_info.shape[0],
        Nfield_bins = field_info.shape[1];

    const voxel_type *voxels = static_cast<voxel_type*>(voxels_info.ptr);

    const prob_type
        *Pxs = static_cast<prob_type*>(x_info.ptr),
        *Pys = static_cast<prob_type*>(y_info.ptr),
        *Pzs = static_cast<prob_type*>(z_info.ptr),
        *Prs = static_cast<prob_type*>(r_info.ptr),
        *Pfield = static_cast<prob_type*>(Pfield_info.ptr);
    
    const field_type *field = static_cast<field_type*>(field_info.ptr);

    result_type *result = static_cast<result_type*>(result_info.ptr);

    const uint64_t *ranges = static_cast<uint64_t*>(ranges_info.ptr);

    const float *vrange = static_cast<float*>(vrange_info.ptr);
    const float vmin = vrange[0];
    const float vmax = vrange[1];
    const float f_min = vrange[2];
    const float f_max = vrange[3];
    
    #pragma omp parallel for collapse(3) schedule(dynamic)
    for (uint64_t z = ranges[0]; z < ranges[1]; z++) {
        for (uint64_t y = ranges[2]; y < ranges[3]; y++) {
            for (uint64_t x = ranges[4]; x < ranges[5]; x++) {
                uint64_t flat_index = z*Ny*Nx + y*Nx + x;
                // Get the voxel value and the indices
                voxel_type voxel = voxels[flat_index];
                int64_t voxel_index = floor(static_cast<double>(Nvoxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );
                uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));

                field_type field_v = field[flat_index];
                int64_t field_index = floor(static_cast<double>(Nfield_bins-1) * ((field_v - f_min)/(f_max - f_min)) );

                // Get the probabilities
                prob_type 
                    Px = Pxs[x*Nvoxel_bins + voxel_index],
                    Py = Pys[y*Nvoxel_bins + voxel_index],
                    Pz = Pzs[z*Nvoxel_bins + voxel_index],
                    Pr = Prs[r*Nvoxel_bins + voxel_index],
                    Pf = Pfield[field_index*Nvoxel_bins + voxel_index];
                
                // Compute the dummy joint probability
                uint64_t ox = ranges[5] - ranges[4];
                uint64_t oy = ranges[3] - ranges[2];
                uint64_t oz = ranges[1] - ranges[0];
                uint64_t offset_index = (z-ranges[0])*oy*ox + (y-ranges[2])*ox + (x-ranges[4]);
                result[offset_index] = ((uint8_t) floor(((Px + Py + Pz + Pr + Pf) / 5) * 255));
            }
        }
    }
}

PYBIND11_MODULE(label, m) {
    // optional module docstring
    m.doc() = "Mapping material probability distributions to a tomography.";
    m.def("material_prob", &material_prob);
}