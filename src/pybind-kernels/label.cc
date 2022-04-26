#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
namespace py = pybind11;

typedef uint16_t voxel_type;
typedef float_t  prob_type;
typedef float    field_type;

void material_prob(const py::array_t<voxel_type> np_voxels,
                    const py::array_t<prob_type> &np_Pxs,
                    const py::array_t<prob_type> &np_Pys,
                    const py::array_t<prob_type> &np_Pzs,
                    const py::array_t<prob_type> &np_Prs,
                    const py::array_t<prob_type> &np_Pfield,
                    const py::array_t<field_type> &np_field,
                    py::array_t<prob_type> &np_result,
                    const double vmin, const double vmax,
                    const double fmin, const double fmax,
                    const bool verbose) {

    py::buffer_info
        voxels_info = np_voxels.request(),
        x_info = np_Pxs.request(),
        y_info = np_Pys.request(),
        z_info = np_Pzs.request(),
        r_info = np_Prs.request(),
        Pfield_info = np_Pfield.request(),
        field_info = np_field.request(),
        result_info = np_result.request();
    
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

    prob_type *result = static_cast<prob_type*>(result_info.ptr);
    
    uint64_t flat_index = 0;
    #pragma omp parallel for collapse(3)
    for (uint64_t z = 0; z < Nz; z++) {
        for (uint64_t y = 0; y < Ny; y++) {
            for (uint64_t x = 0; x < Nx; x++) {
                // Get the voxel value and the indices
                voxel_type voxel = voxels[flat_index];
                int64_t voxel_index = floor(static_cast<double>(Nvoxel_bins-1) * ((voxel - vmin)/(vmax - vmin)) );
                uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));

                field_type field_v = field[flat_index];
                int64_t field_index = floor(static_cast<double>(Nfield_bins-1) * ((field_v - fmin)/(fmax - fmin)) );

                // Get the probabilities
                prob_type 
                    Px = Pxs[x*Nvoxel_bins + voxel_index],
                    Py = Pys[y*Nvoxel_bins + voxel_index],
                    Pz = Pzs[z*Nvoxel_bins + voxel_index],
                    Pr = Prs[r*Nvoxel_bins + voxel_index],
                    Pf = Pfield[field_index*Nvoxel_bins + voxel_index];
                
                // Compute the dummy joint probability
                result[flat_index] = (Px + Py + Pz + Pr + Pf) / 5;
                flat_index++;
            }
        }
    }
}

PYBIND11_MODULE(label, m) {
    // optional module docstring
    m.doc() = "Mapping material probability distributions to a tomography.";
    m.def("material_prob", &material_prob);
}