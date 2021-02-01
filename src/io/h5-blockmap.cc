#include <h5pp/h5pp.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <inttypes.h>

using namespace std;
// This example shows how to use to write data into a portion of a dataset, a so-called "hyperslab", using h5pp::Options

/********************************************************************
   Note that the HDF5 C-API uses row-major layout!
*********************************************************************/

float mean(const vector<float> &data, const h5pp::Hyperslab &slab)
{
  double sum = 0;
  for(float x: data) sum += x;
  return sum/data.size();
}

// TODO: Template over in-type and out-type
// TODO: Create dataset in file_out
void process_h5(string filename_in, string filename_out, 
	       int chunk_size=256, int padding=28, string dsetname="voxels")
{
  using namespace h5pp;

  File file_in (filename_in, FilePermission::READONLY);
  //  File file_out(filename_out,FilePermission::REPLACE);

  auto dset = file_in.getDatasetInfo(dsetname);
  auto dset_shape = dset.dsetDims.value();
  int Nz = dset_shape[0], Ny = dset_shape[1], Nx = dset_shape[2];
  
  for(int z0=0;z0<Nz;z0+=chunk_size){
    int z1 = min(Nz, z0+chunk_size);              // z0:z1 is inner region (written to result)
    int Z0 = max(0,z0-padding), Z1 = min(Nz,z1+padding);
    int top_padding = Z0-z0, bot_padding = Nz-Z1;	  

    Hyperslab slab({Z0,0,0},{Z1-Z0,Ny,Nx});
    
    vector<float> chunk_in  = file_in.readHyperslab< vector<float> >(dsetname, slab);
    float m = mean(chunk_in, slab);
    printf("Mean of data[%d:%d,%d:%d,%d:%d] = %f\n",Z0,Z1,0,Ny,0,Nx,m);
    
    //    vector<float> chunk_out = process(chunk_in, dset_shape, z0, z1, Z0, Z1);
    //    file_in.writeHyperslab<vector<float>>(chunk_out, dsetname, Hyperslab({z0,0,0},{z1-z0,Ny,Nx}));
  }

  //  file_in.close();
  //  file_out.close();
}


int main(int ac, char **av)
{
  if(ac<=2){
    fprintf(stderr,"Syntax: %s <h5_input> <h5_output>\n", av[0]);
    exit(-1);
  }
  
  string h5_infile = av[1], h5_outfile = av[2];

  process_h5(h5_infile, h5_outfile);
  return 0;
}
