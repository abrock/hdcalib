#include <iostream>

#include <tclap/CmdLine.h>

#include "hdcalib.h"

std::string basename(std::string const& name) {
    size_t const pos = name.find_last_of('.');
    if (pos == name.npos) {
        return name;
    }
    return name.substr(0, pos);
}

int main(int argc, char ** argv) {
    try {
        TCLAP::CmdLine cmd("hdcalib calibration tool", ' ', "0.1");

        TCLAP::ValueArg<double> factor_arg("f", "factor",
                                          "Factor for image normalization.",
                                          false, 1, "image normalization factor");
        cmd.add(factor_arg);

        TCLAP::UnlabeledMultiArg<std::string> input_img_arg("input_img",
                                                            "Input images.",
                                                            false,
                                                            "Input images.");
        cmd.add(input_img_arg);

        cmd.parse(argc, argv);

        double const factor = std::max(0.0, factor_arg.getValue());
        std::vector<std::string> images = input_img_arg.getValue();

        for (std::string const& image : images) {
            hdcalib::Calib c;
            std::string const base = basename(image);

            cv::Mat img = c.read_raw(image);

            cv::cvtColor(img, img, cv::COLOR_BayerBG2BGR);
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
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    return EXIT_SUCCESS;
}
