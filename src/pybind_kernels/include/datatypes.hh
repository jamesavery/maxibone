#include <array>
#include <vector>

typedef uint8_t mask_type;	// TODO: Template + explicit instantiation
typedef double real_t;

constexpr ssize_t acc_block_size =  1024 * 1024 * 1024/sizeof(mask_type); // 1 GB

struct plane_t {
  array<real_t,3> cm, u_axis, v_axis;
};

template <typename T> struct input_ndarray {
  const T *data;
  const vector<ssize_t> shape;

  input_ndarray(const T *data, const vector<ssize_t> &shape): data(data), shape(shape) {}
  input_ndarray(const void *data, const vector<ssize_t> &shape): data(static_cast<const T*>(data)), shape(shape) {}  
};

template <typename T> struct output_ndarray {
  T *data;
  const vector<ssize_t> shape;

  output_ndarray(T *data, const vector<ssize_t> &shape): data(data), shape(shape) {}
  output_ndarray(void *data, const vector<ssize_t> &shape): data(static_cast<T*>(data)), shape(shape) {}    
};



