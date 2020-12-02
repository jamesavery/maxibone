#include <h5pp/h5pp.h>
#include <iostream>
#include <vector>

using namespace std;
// This example shows how to use to write data into a portion of a dataset, a so-called "hyperslab", using h5pp::Options

/********************************************************************
   Note that the HDF5 C-API uses row-major layout!
*********************************************************************/


// TODO: Template over in-type and out-type
void process_h5(string filename_in, string filename_out, auto process,
	       size_t chunk_size=256, size_t padding=28, string dsetname="voxels")
{
  using namespace h5pp;

  File file_in (filename_in, FilePermission::READONLY);
  File file_out(filename_out,FilePermission::REPLACE);

  size_t Nz, Ny, Nx; // = file_in['voxels'].shape
  vector<int> dset_shape = {Nz, Ny, Nx};
  
  for(size_t z0=0;z0<Nz;z0+=chunk_size){
    size_t z1 = min(Nz, z0+chunk_size);              // z0:z1 is inner region (written to result)
    size_t top_padding = (z0!=0) *padding;	     // Pad top    except for first chunk
    size_t bot_padding = (z1!=Nz)*padding;	     // Pad bottom except for last  chunk
    size_t Z0 = z0-top_padding, Z1 = z1+bot_padding; // Z0:Z1 is padded region (read from input and processed)

    vector<float> chunk_in  = file_in.readHyperslab< vector<float> > (dsetname, Hyperslab({Z0,0,0},{Z1-Z0,Ny,Nx}));
    vector<float> chunk_out = process(chunk_in, dset_shape, z0, z1, Z0, Z1);
    file_in.writeHyperslab<vector<float>>(chunk_out, dsetname, Hyperslab({z0,0,0},{z1-z0,Ny,Nx}));
  }

  //  file_in.close();
  //  file_out.close();
}
  
