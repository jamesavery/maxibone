#ifndef geometry_h
#define geometry_h

using namespace std;

#include "datatypes.hh"
#include <chrono>
#include <string>

#define dot(a,b) (a[0]*b[0] + a[1]*b[1] + a[2]*b[2])

void print_timestamp(string message) {
    //auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    //tm local_tm = *localtime(&now);
    //fprintf(stderr,"%s at %02d:%02d:%02d\n", message.c_str(), local_tm.tm_hour, local_tm.tm_min, local_tm.tm_sec);
}

array<real_t,3> center_of_mass(const input_ndarray<mask_type> voxels);
array<real_t,9> inertia_matrix(const input_ndarray<mask_type> &voxels, const array<real_t,3> &cm);

#endif