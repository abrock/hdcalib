#include <iostream>

#include "hdcalib.h"

std::string basename(std::string const& name) {
    size_t const pos = name.find_last_of('.');
    if (pos == name.npos) {
        return name;
    }
    return name.substr(0, pos);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " any number of raw image files..." << std::endl;
        return 0;
    }

    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        hdcalib::Calib c;
        std::string const filename(argv[ii]);
        std::string const base = basename(filename);

        cv::Mat img = c.read_raw(argv[ii]);

        cv::cvtColor(img, img, cv::COLOR_BayerBG2BGR);
        cv::Mat split[3];
        cv::split(img, split);
        img = split[1];
        cv::imwrite(base + "-16.png", img);

        double min = 0, max = 0;
        cv::minMaxIdx(img, &min, &max);
        std::cout << "original min/max: " << min << " / " << max << std::endl;

        img.convertTo(img, CV_8U, 0.00390625);
        cv::imwrite(base + "-8.png", img);

        cv::minMaxIdx(img, &min, &max);
        std::cout << "8 bit min/max: " << min << " / " << max << std::endl;

    }

}
