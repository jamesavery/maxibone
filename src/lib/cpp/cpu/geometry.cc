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

array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm) {
    real_t
        Ixx = 0, Ixy = 0, Ixz = 0,
                 Iyy = 0, Iyz = 0,
                          Izz = 0;
  
    ssize_t Nx = voxels.shape[0], Ny = voxels.shape[1], Nz = voxels.shape[2];

    print_timestamp("inertia_matrix_serial start");
    
    #pragma omp parallel for collapse(3) reduction(+:Ixx,Iyy,Izz) reduction(-:Ixy,Ixz,Iyz)
    for (int64_t X = 0; X < Nx; X++) {
        for (int64_t Y = 0; Y < Ny; Y++) {
            for (int64_t Z = 0; Z < Nz; Z++) {

                // TODO shouldn't the loops be interchanged to match the access pattern? (Naming-wise, that is)
                int64_t k = X*Ny*Nz + Y*Nz + Z; 
                mask_type m = voxels.data[k];
                
                // m guards this, and this removes branches
                // if (m != 0) 
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
  
    print_timestamp("inertia_matrix_serial end");

    return array<real_t,9> {
        Ixx, Ixy, Ixz,
        Ixy, Iyy, Iyz,
        Ixz, Iyz, Izz
    };
}