#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>
using namespace std;

#include "geometry.hh"

array<real_t,3> center_of_mass(const input_ndarray<mask_type> voxels) {
    uint64_t cmx = 0, cmy = 0, cmz = 0;
    size_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];
    int64_t image_length = Nx*Ny*Nz;

    print_timestamp("center_of_mass start");
    
    uint64_t total_mass = 0;  
    
    #pragma omp parallel for reduction(+:total_mass,cmx,cmy,cmz)
    for (int64_t k = 0; k < image_length; k++) {
        mask_type m = voxels.data[k];      

        int64_t x = k / (Ny*Nz);
        int64_t y = (k / Nz) % Ny;
        int64_t z = k % Nz;

        total_mass += m;
        cmx += m*x; cmy += m*y; cmz += m*z;
    }
    real_t
        rcmx = cmx / ((real_t) total_mass),
        rcmy = cmy / ((real_t) total_mass),
        rcmz = cmz / ((real_t) total_mass);
  
    print_timestamp("center_of_mass end");  

    return array<real_t,3>{ rcmx, rcmy, rcmz };
}