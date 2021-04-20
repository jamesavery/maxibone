#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <stdio.h>
#include <inttypes.h>

using namespace std;

int main(int ac, char **av)
{
  fprintf(stderr,"Just starting up, doing nothing.\n");
  
  if(ac<2) return -1;
  
  cv::Mat img = cv::imread(av[1], cv::IMREAD_COLOR);
  
  if(img.empty()) return -2;

  cv::imshow("Window",img);

  int k = cv::waitKey(0);

  return 0;

}
