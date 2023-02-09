#include <h5pp/h5pp.h>
#include <string>
#include <vector>
#include <inttypes.h>
using namespace std;

typedef uint8_t byte;

typedef uint16_t data_int;
typedef uint16_t field_int;
typedef uint64_t count_int;

void bincount(data_int *data, uint64_t data_length, data_int value_length, count_int *output)
{
  count_int bins[value_length];
  for(int i=0;i<value_length;i++) bins[i] = 0;

#pragma omp parallel for reduction(+: bins)
  for(uint64_t i=0;i<data_length;i++){
    bins[data[i]]++;
  }

  for(int i=0;i<value_length;i++) output[i] = bins[i];
}

void bincount_axes(data_int *voxels, data_int n_values,
		   field_int *field,   data_int n_field_segments, 
		   count_int *zbins,count_int *ybins, count_int *xbins, count_int *fbins, 
		   uint64_t nz,      uint64_t ny,      uint64_t nx, 
		   uint64_t stridez, uint64_t stridey, uint64_t stridex)
{
  for(int i=0;i<n_values;i++){
    zbins[i] = 0;
    ybins[i] = 0;
    zbins[i] = 0;
    fbins[i] = 0;
  }
  
#pragma omp parallel for reduction(+: xbins, ybins, zbins)  
  for(uint64_t z=0;z<nz;z++){
    for(uint64_t y=0;y<ny;y++){
      for(uint64_t x=0;x<nx;x++){
	uint64_t index = z*stridey*stridex + y*stridex + x;
	data_int value = voxels[index];
	zbins[value*nz + z]++;
	ybins[value*ny + y]++;
	xbins[value*nx + x]++;
	fbins[value*n_field_segments + field[index]]++;
      }
    }
  }
}


void bincount_field(data_int *voxels,    field_int *field,
		    data_int n_segments, data_int n_values,
		    count_int *output,
		    uint64_t nz,      uint64_t ny,      uint64_t nx, 
		    uint64_t stridez, uint64_t stridey, uint64_t stridex)
{
  
}


int main(int ac, char **av)
{
  string sample    = string(av[1]);
  string field     = string(av[2]);
  string scale     = string(av[3]);  
  string hdf5_root = string(av[4]);

  h5pp::File voxel_msb_file(hdf5_root+"hdf5-byte/msb/"+scale+"x/"+sample+".h5",h5pp::filePermission::READONLY);
  h5pp::File voxel_lsb_file(hdf5_root+"hdf5-byte/lsb/"+scale+"x/"+sample+".h5",h5pp::filePermission::READONLY);
  h5pp::File field_file(hdf5_root+"processed/implant/"+scale+"x/"+sample+"-"+field+".h5",h5pp::filePermission::READONLY);  

  auto voxels_msb = voxel_msb_file.readDataset< vector<uint8> >("voxels");
  auto voxels_lsb = voxel_lsb_file.readDataset< vector<uint8> >("voxels");

  
}
