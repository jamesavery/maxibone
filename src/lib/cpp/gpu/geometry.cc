#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "geometry.hh"

array<real_t,3> center_of_mass(const input_ndarray<mask_type> voxels) {
    // nvc++ doesn't support OpenACC 2.7 array reductions yet.  
    uint64_t cmx = 0, cmy = 0, cmz = 0;
    size_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    int64_t image_length = Nx*Ny*Nz;

    print_timestamp("center_of_mass start");

    uint64_t total_mass = 0;
    
    #pragma acc data copy(total_mass,cmx,cmy,cmz)
    {
        for (int64_t block_start = 0; block_start < image_length; block_start += acc_block_size) {
            const mask_type *buffer = voxels.data + block_start;
            ssize_t this_block_size = min(acc_block_size, image_length-block_start);

            #pragma acc data copyin(buffer[:this_block_size]) 
            {
                #pragma acc parallel loop reduction(+:total_mass,cmx,cmy,cmz)
                for (int64_t k = 0; k < this_block_size; k++) {
                    mask_type m = buffer[k];

                    int64_t flat_idx = block_start + k;
                    int64_t x = flat_idx / (Ny*Nz);
                    int64_t y = (flat_idx / Nz) % Ny;
                    int64_t z = flat_idx % Nz;

                    total_mass += m;
                    cmx += m*x; cmy += m*y; cmz += m*z;
                }
            }
        }
    }
    real_t 
        rcmx = cmx / ((real_t) total_mass),
        rcmy = cmy / ((real_t) total_mass),
        rcmz = cmz / ((real_t) total_mass);
  
    print_timestamp("center_of_mass end");  

    return array<real_t,3>{rcmx, rcmy, rcmz};
}

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm) {
    // nvc++ doesn't support OpenACC 2.7 array reductions yet, so must name each element.
    real_t
        Ixx = 0, Ixy = 0, Ixz = 0,
                 Iyy = 0, Iyz = 0,
                          Izz = 0;
  
    size_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    ssize_t image_length = Nx*Ny*Nz;

    print_timestamp("inertia_matrix start");

    #pragma acc data copy(Ixx, Ixy, Ixz, Iyy, Iyz, Izz) 
    {
        for (ssize_t block_start = 0; block_start < image_length; block_start += acc_block_size) {
            const mask_type *buffer = voxels.data + block_start;
            ssize_t this_block_size = min(acc_block_size, image_length - block_start);

            #pragma acc data copyin(buffer[:this_block_size]) 
            {
                #pragma acc parallel loop reduction(+:Ixx,Iyy,Izz) reduction(-:Ixy,Ixz,Iyz)
                for (int64_t k = 0; k < this_block_size; k++) {    //\if (buffer[k] != 0)
                    mask_type m = buffer[k];

                    // m guards this, and GPUs doesn't like branches
                    //if (m != 0)
                    int64_t 
                        flat_idx = block_start + k,
                        X = flat_idx / (Ny * Nz),
                        Y = ((flat_idx) / Nz) % Ny,
                        Z = flat_idx % Nz;
                    
                    real_t 
                        x = X - cm[0], 
                        y = Y - cm[1], 
                        z = Z - cm[2];
                
                    Ixx += m * (y*y + z*z);
                    Iyy += m * (x*x + z*z);
                    Izz += m * (x*x + y*y);    
                    Ixy -= m * x*y;
                    Ixz -= m * x*z;
                    Iyz -= m * y*z;
                }
            }
        }
    }

    print_timestamp("inertia_matrix end");

    return array<real_t,9> {
        Ixx, Ixy, Ixz,
        Ixy, Iyy, Iyz,
        Ixz, Iyz, Izz
    };
}