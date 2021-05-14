#include <iostream>
#include <opencv2/highgui.hpp>

#include "randutils.hpp"

int main(int argc, char ** argv) {
    randutils::mt19937_rng rng;
    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        cv::Mat_<uint16_t> img = cv::imread(argv[ii], cv::IMREAD_UNCHANGED);
        for (uint16_t & val : img) {
            double new_val = val;
            new_val += rng.variate<double, std::normal_distribution>(0, std::sqrt(double(val)));
            val = cv::saturate_cast<uint16_t>(std::round(new_val));
        }
        cv::imwrite(argv[ii], img);
    }
}
