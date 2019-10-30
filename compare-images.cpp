#include <iostream>

#include <opencv2/highgui.hpp>

#include <runningstats/runningstats.h>

int main(int argc, char ** argv) {

    if (argc <3) {
        std::cout << "Usage: " << argv[0] << " <image a> <image b>" << std::endl;
        return EXIT_FAILURE;
    }

    cv::Mat_<cv::Vec3b> a,b;

    a = cv::imread(argv[1]);
    b = cv::imread(argv[2]);

    if (a.size() != b.size()) {
        std::cout << "Sizes differ, can not compare. " << std::endl;
        return EXIT_FAILURE;
    }


    runningstats::RunningStats stats[3];

    for (int ii = 0; ii < a.rows; ++ii) {
        for (int jj = 0; jj < a.cols; ++jj) {
            for (int kk = 0; kk < 3; ++kk) {
                stats[kk].push(a(ii,jj)[kk] - b(ii,jj)[kk]);
            }
        }
    }

    std::cout << "Residual stats for channels BGR:" << std::endl;

    for (int kk = 0; kk < 3; ++kk) {
        std::cout << stats[kk].print() << std::endl;
    }

    std::cout << std::endl;

}
