#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/optflow.hpp>

#include <runningstats/runningstats.h>

#include "libplotoptflow.h"

int main(int argc, char ** argv) {

    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        cv::Mat_<cv::Vec2f> flow = cv::readOpticalFlow(argv[ii]);
        std::cout << "Input file: " << argv[ii] << std::endl;
        //hdflow::fitSpline<9,3>(std::string(argv[ii]) + "-", flow, -1, 100);
        cv::imwrite(std::string(argv[ii]) + ".png", hdflow::plotWithArrows(flow, -1, 100));
        hdflow::run(std::string(argv[ii]) + "-", flow, -1, 100, false);
    }

}
