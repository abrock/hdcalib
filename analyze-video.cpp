#include <iostream>
#include <fstream>

#include <opencv2/highgui.hpp>

void analyzeVideo(std::string const& filename) {
    cv::VideoCapture capture(filename);
    cv::Mat image;
    std::ofstream out(filename + ".stats");
    while (capture.read(image)) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(image, mean, stddev);
        double mean_sum = 0;
        double stddev_sum = 0;
        for (size_t ii = 0; ii < 3; ++ii) {
            mean_sum += mean[ii] / 3.0;
            stddev_sum += stddev[ii] / 3.0;
        }
        out << mean_sum << "\t" << stddev_sum << "\t" << mean << "\t" << stddev << std::endl;
    }
}

int main(int argc, char ** argv) {
    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        analyzeVideo(argv[ii]);
    }
}
