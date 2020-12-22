#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// On entry, x_bins[Nx*voxel_bins], y_bins[Ny*voxel_bins], z_bins[Nz*voxel_bins], r_bins[Nr*voxel_bins] must be allocated and zeroed
void execute(uint64_t       *restrict x_bins,
	     uint64_t       *restrict y_bins,
	     uint64_t       *restrict z_bins,
	     uint64_t       *restrict r_bins,
            %(voxel_type)s *voxels)
{
  const uint64_t voxel_bins = %(voxel_bins)d, Nx = %(Nx)d, Ny = %(Ny)d, Nz = %(Nz)d, Nr = %(Nr)d;
  const double vmin = %(vmin)f, vmax = %(vmax)f;

  uint64_t i;

#pragma omp parallel for private(i) reduction(+:x_bins[:Nx*voxel_bins], y_bins[:Ny*voxel_bins], z_bins[:Nz*voxel_bins], r_bins[:Nr*voxel_bins])  
  for(i=0;i< %(image_length)d;i++){
    uint64_t x = i %% Nx;
    uint64_t y = (i / Nx) %% Ny;
    uint64_t z = i / (Nx*Ny);
    uint64_t r = floor(sqrt(x*x + y*y));
    
    uint64_t voxel_index = floor((voxel_bins-1) * (voxels[i] - vmin)/(vmax - vmin) );
    
    x_bins[x*voxel_bins + voxel_index]++;
    y_bins[y*voxel_bins + voxel_index]++;    
    z_bins[z*voxel_bins + voxel_index]++;    
    r_bins[r*voxel_bins + voxel_index]++;
  }
}
		     
