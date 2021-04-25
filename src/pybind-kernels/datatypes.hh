#include <array>
#include <vector>

typedef uint8_t voxel_type;	// TODO: Template + explicit instantiation
typedef double real_t;

constexpr ssize_t acc_block_size =  1024 * 1024 * 1024/sizeof(voxel_type); // 1 GB

struct plane_t {
  array<real_t,3> cm, u_axis, v_axis;
};

template <typename T> struct ndarray_input {
  const T *data;
  const vector<ssize_t> shape;

  ndarray_input(const T *data, const vector<ssize_t> &shape): data(data), shape(shape) {}
  ndarray_input(const void *data, const vector<ssize_t> &shape): data(static_cast<const T*>(data)), shape(shape) {}  
};

template <typename T> struct ndarray_output {
  T *data;
  const vector<ssize_t> shape;

  ndarray_output(T *data, const vector<ssize_t> &shape): data(data), shape(shape) {}
  ndarray_output(void *data, const vector<ssize_t> &shape): data(static_cast<T*>(data)), shape(shape) {}    
};



