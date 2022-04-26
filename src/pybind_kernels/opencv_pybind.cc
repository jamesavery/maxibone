#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <inttypes.h>

using namespace std;
using namespace cv;
namespace py = pybind11;

void opencv_tester(const py::array_t<uint8_t>& np_image)
{
  py::buffer_info image_info = np_image.request();
  uint64_t 
    Ny  = image_info.shape[0],
    Nx  = image_info.shape[1];

  Mat img(Ny, Nx, CV_8UC1, image_info.ptr);

  imshow("opencv_tester window",img);
  int k = waitKey(0);
}


PYBIND11_MODULE(opencv_pybind, m) {
    m.doc() = "Test of C++ OpenCV through pybind"; // optional module docstring

    m.def("tester",  &opencv_tester);
}

