#include <iostream>
#include "hdcalib.h"


int main(int argc, char ** argv) {

    hdcalib::Calib c;
    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        cv::Mat_<cv::Vec3b> img = cv::imread(argv[ii]);
        cv::Mat_<cv::Vec3b> filled = c.fillHoles(img);
        cv::imwrite(std::string(argv[ii]) + "-filled.png", filled);
    }
}
