#ifndef CALIB_H
#define CALIB_H

#include <stdio.h>
#include <map>
#include <iostream>
#include <unordered_map>
#include <exception>

#include <hdmarker/hdmarker.hpp>
#include <hdmarker/subpattern.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <boost/filesystem.hpp>

#include <libraw/libraw.h>

namespace hdcalib {
using namespace std;
using namespace hdmarker;
using namespace cv;
namespace fs = boost::filesystem;

class Calib
{
  bool verbose = true;
  std::vector<std::string> input_images;
  int grid_width = 1;
  int grid_height = 1;

  bool use_rgb = false;
public:
  Calib();

  void setInputImages(std::vector<std::string> const& files) {

  }

  cv::Mat normalize_raw_per_channel(cv::Mat const& input);

  void normalize_raw_per_channel_inplace(cv::Mat & input);

  cv::Mat read_raw(std::string const& filename);

  /**
   * @brief getCorners
   * @param input_file
   * @param effort
   * @param demosaic
   * @param recursion_depth
   * @param raw
   * @return
   */
  vector<Corner> getCorners(
      const std::string input_file,
      const float effort,
      const bool demosaic,
      const int recursion_depth,
      const bool raw);
};

}

#endif // CALIB_H
