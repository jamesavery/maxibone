#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <inttypes.h>
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <iostream>
namespace py = pybind11;

typedef uint16_t result_type;
typedef uint16_t voxel_type;
typedef float_t  prob_type;
typedef uint16_t field_type;

uint64_t number_of_set_bits(uint64_t number) {
    uint64_t count = 0;
    while (number) {
        count += number & 1;
        number >>= 1;
    }
    return count;
}

void material_prob(const py::array_t<voxel_type> &np_voxels,
                   const py::array_t<field_type> &np_field,
                   const py::list li_axes_probs,
                   const uint64_t axes_probs_mask,
                   const py::list li_field_probs,
                   const uint64_t field_probs_mask,
                   const py::array_t<prob_type> &np_weights,
                   py::array_t<result_type> &np_result,
                   const std::tuple<float, float> &vrange,
                   const std::tuple<float, float> &frange,
                   const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
                   const std::tuple<uint64_t, uint64_t, uint64_t> &ranges) {

    // Extract the probability arrays
    uint64_t n_axes = number_of_set_bits(axes_probs_mask);
    py::array_t<prob_type> np_axes_probs[n_axes];
    py::buffer_info axes_probs_info[n_axes];
    const prob_type* axes_probs[n_axes];
    uint64_t axes_access_idx[n_axes];
    int i = 0, j = 0;
    for (py::handle array: li_axes_probs) {
        if ((axes_probs_mask >> j) & 1) {
            np_axes_probs[i] = py::cast<py::array_t<prob_type>>(array);
            axes_probs_info[i] = np_axes_probs[i].request();
            axes_probs[i] = static_cast<const prob_type*>(axes_probs_info[i].ptr);
            axes_access_idx[i] = j;
            i++;
        }
        j++;
    }

    uint64_t n_fields = number_of_set_bits(field_probs_mask);
    py::array_t<prob_type> np_field_probs[li_field_probs.size()];
    py::buffer_info field_probs_info[li_field_probs.size()];
    const prob_type* field_probs[li_field_probs.size()];
    i = 0, j = 0;
    for (py::handle array: li_field_probs) {
        if ((field_probs_mask >> j) & 1) {
            np_field_probs[i] = py::cast<py::array_t<prob_type>>(array);
            field_probs_info[i] = np_field_probs[i].request();
            field_probs[i] = static_cast<const prob_type*>(field_probs_info[i].ptr);
            i++;
        }
        j++;
    }


    py::buffer_info
        voxels_info = np_voxels.request(),
        field_info = np_field.request(),
        weights_info = np_weights.request(),
        result_info = np_result.request();

    // TODO this might fail if the first of each is not enabled by the mask.
    const uint32_t
        Nvoxel_bins = n_axes > 0 ? axes_probs_info[0].shape[1] : 0, // TODO: Don't depend on axes being used
        Nfield_bins = n_fields > 0 ? field_probs_info[0].shape[2] : 0;

    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
    const field_type *field = static_cast<const field_type*>(field_info.ptr);
    const prob_type *weights = static_cast<prob_type*>(weights_info.ptr);
    result_type *result = static_cast<result_type*>(result_info.ptr);

    auto [sz, sy, sx] = offset;
    auto [Nz, Ny, Nx] = ranges;
    
    uint64_t
        fz = (Nz-sz) / 2, //TODO this is off // TODO: How so?
        fy = Ny / 2,
        fx = Nx / 2;
    auto [v_min, v_max] = vrange;
    auto [f_min, f_max] = frange;

    // fprintf(stderr,"(Nz,Ny,Nx) = (%d,%d,%d)\n",Nz,Ny,Nx);
    // fprintf(stderr,"(sz,sy,sx) = (%d,%d,%d)\n",sz,sy,sx);
    // fprintf(stderr,"n_axes = %ld, n_fields = %ld\n",n_axes,n_fields);
    

#pragma omp parallel for collapse(3)          
    for (uint64_t z = sz; z < Nz; z++) {
        for (uint64_t y = sy; y < Ny; y++) {
            for (uint64_t x = sx; x < Nx; x++) {
                // TODO Only compute the indices and such if they're actually used.
                uint64_t flat_index = (z-sz)*Ny*Nx + y*Nx + x; // TODO: Shouldn't y and x be offset, too??
                // get the voxel value and the indices
                voxel_type voxel = voxels[flat_index];
                if (voxel < v_min || voxel > v_max)
                    continue;
		
                int64_t voxel_index = floor(static_cast<double>(Nvoxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );
                uint64_t r = floor(sqrt((x-Nx/2.0)*(x-Nx/2.0) + (y-Ny/2.0)*(y-Ny/2.0)));

                prob_type p = 0;
                uint64_t axes_idxs[] = {x, y, z, r};
                for (uint64_t i = 0; i < n_axes; i++) {
                    p += axes_probs[i][axes_idxs[axes_access_idx[i]]*Nvoxel_bins + voxel_index] * weights[i];
                }

                for (uint64_t i = 0; i < n_fields; i++) {
                    uint64_t flat_field_index = ((z-sz)/2)*fy*fx + (y/2)*fx + (x/2);
                    field_type field_v = field[i*fz*fy*fx + flat_field_index];
                    if (field_v < f_min || field_v > f_max)
                        continue;
                    int64_t field_index = floor(static_cast<double>(Nfield_bins-1) * ((field_v - f_min)/(f_max - f_min)) );
                    p += field_probs[i][field_index*Nfield_bins + voxel_index] * weights[i + n_axes];
                }

                // Compute the joint probability and cast between [0:result_type_max_value]
                result[flat_index] = round(p * std::numeric_limits<result_type>::max());
            }
        }
    }
}

void material_prob_justonefieldthx(const py::array_t<voxel_type> &np_voxels,
				   const py::array_t<field_type> &np_field,
				   const py::array_t<prob_type>  &np_prob,			      
				   py::array_t<result_type> &np_result,
				   const std::pair<voxel_type, voxel_type> &vrange,
				   const std::pair<field_type, field_type> &frange,
				   const std::tuple<uint64_t, uint64_t, uint64_t> &offset,
				   const std::tuple<uint64_t, uint64_t, uint64_t> &ranges) {

    py::buffer_info
      voxels_info = np_voxels.request(),
      field_info  = np_field.request(),
      prob_info   = np_prob.request(),
      result_info = np_result.request();

    const voxel_type *voxels = static_cast<const voxel_type*>(voxels_info.ptr);
    const field_type *field  = static_cast<const field_type*>(field_info.ptr);
    const prob_type  *prob   = static_cast<prob_type*>(prob_info.ptr);
    result_type      *result = static_cast<result_type*>(result_info.ptr);

    const uint32_t Nfield_bins = prob_info.shape[0],
                   Nvoxel_bins = prob_info.shape[1];
    
    auto [sz, sy, sx] = offset;
    auto [Nz, Ny, Nx] = ranges;
    const auto nz = voxels_info.shape[0], ny = voxels_info.shape[1], nx = voxels_info.shape[2];
    const auto fz = field_info.shape[0],  fy = field_info.shape[1],  fx = field_info.shape[2];

    auto [v_min, v_max] = vrange;
    auto [f_min, f_max] = frange;

    printf("v_min, v_max = %d,%d\nf_min, f_max = %d,%d\n",v_min,v_max, f_min,f_max);
    
    for(size_t i=0;i<result_info.size;i++) result[i] = 0;

#pragma omp parallel for collapse(3)          
    for (uint64_t z = sz; z < Nz; z++) {
        for (uint64_t y = sy; y < Ny; y++) {
            for (uint64_t x = sx; x < Nx; x++) {
                uint64_t flat_index = (z-sz)*Ny*Nx + y*Nx + x;

		assert(flat_index < voxels_info.size);
		voxel_type voxel = voxels[flat_index];
		
                if (voxel < v_min || voxel > v_max)
                    continue;

                int64_t voxel_index = floor(static_cast<double>(Nvoxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );

		assert(voxel_index > 0 && voxel_index < Nvoxel_bins);
                prob_type p = 0;

                // TODO: Allow variable voxels scale and field scale (dx = Nx/float(fx), etc.)                                                                                                                     
                uint64_t flat_field_index = ((z-sz)/2)*fy*fx + (y/2)*fx + (x/2);
		assert(flat_field_index < field_info.size);
		
                field_type field_v        = field[flat_field_index];

                if (field_v < f_min || field_v > f_max)
                        continue;
                int64_t field_index = floor(static_cast<double>(Nfield_bins-1) * ((field_v - f_min)/(f_max - f_min)) );

		assert(field_index > 0 && field_index < Nfield_bins);
		
                p += prob[field_index*Nfield_bins + voxel_index]; //* weights[i + n_axes];                                                                                                                         
		if(p<0 || p>1){
		  fprintf(stderr,"p = %g; field_v = %d; voxel_voxel = %d\n",p, field_v,voxel);
		  abort();
		}
                result[flat_index] = round(p * std::numeric_limits<result_type>::max());

		
                // // get the voxel value and the indices
		// assert(flat_index < voxels_info.size);
		
                // voxel_type voxel = voxels[flat_index];

                // if (voxel == 0)	// Maskeret voxel
                //     continue;

		// voxel = std::max(v_min,voxel);
		// voxel = std::min(v_max,voxel);
		
                // int64_t voxel_index = floor(static_cast<double>(Nvoxel_bins-1) * ((voxel - v_min)/(v_max - v_min)) );
		// assert(voxel_index >= 0 && voxel_index < Nvoxel_bins);
		
                // prob_type p = 0;

		// // TODO: Allow variable voxels scale and field scale (dx = Nx/float(fx), etc.)
		// uint64_t flat_field_index = ((z-sz)/2)*fy*fx + (y/2)*fx + (x/2);

		// assert(flat_field_index < field_info.size);
		  
		// field_type field_v        = field[flat_field_index];

		// // if (field_v < f_min || field_v > f_max)
                // //         continue;
		
		// int64_t field_index = floor(static_cast<double>(Nfield_bins-1) * ((field_v - f_min)/(f_max - f_min)) );
		// assert(field_index >= 0 && field_index < Nfield_bins);
		// p += prob[field_index*Nfield_bins + voxel_index]; //* weights[i + n_axes];

		// assert(p>=0);
		// assert(p<1);
                // // cast between [0:result_type_max_value]
                // result[flat_index] = round(p * std::numeric_limits<result_type>::max());

            }
        }
    }
    // fprintf(stderr,"maxint = %d\n",std::numeric_limits<result_type>::max());
    // result_type r_min = 0, r_max = 0;
    // for(size_t i=0;i<result_info.size;i++){ r_min = std::min(r_min,result[i]); r_max = std::max(r_max,result[i]); }
    // fprintf(stderr,"result min,max = %d,%d\n",r_min,r_max);
}

PYBIND11_MODULE(label, m) {
    // optional module docstring
    m.doc() = "Mapping material probability distributions to a tomography.";
    m.def("material_prob", &material_prob);
    m.def("material_prob_justonefieldthx",&material_prob_justonefieldthx);
}
