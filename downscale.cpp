#include <iostream>
#include <opencv2/highgui.hpp>
#include <tclap/CmdLine.h>

#include "libdownscale.h"

std::string tostringLZ(size_t num, size_t min_digits = 2) {
    std::string result = std::to_string(num);
    if (result.size() < min_digits) {
        result = std::string(min_digits - result.size(), '0') + result;
    }
    return result;
}

int main(int argc, char ** argv) {

    TCLAP::CmdLine cmd("HDCalib downscale tool for integer downscaling");

    TCLAP::ValueArg<size_t> factor_arg("f", "factor", "Integer factor", false, 1, "integer");
    cmd.add(factor_arg);

    TCLAP::ValueArg<size_t> crop_x_arg("x", "shift-x", "Crop the top x lines of the input image before downscaling", false, 1, "integer");
    cmd.add(crop_x_arg);

    TCLAP::ValueArg<size_t> crop_y_arg("y", "shift-y", "Crop the top y lines of the input image before downscaling", false, 1, "integer");
    cmd.add(crop_y_arg);

    TCLAP::UnlabeledMultiArg<std::string> images_arg("images", "Input images", true, "filename");
    cmd.add(images_arg);


    cmd.parse(argc, argv);

    for (std::string const& file : images_arg.getValue()) {
        std::cout << "Reading file " << file << "..." << std::flush;
        cv::Mat_<float> img = cv::imread(file, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE);
        std::cout << " done. Size: " << img.size() << std::endl;
        if (img.empty()) {
            std::cout << "File " << file << " is empty" << std::endl;
            continue;
        }
        std::cout << "Cropping..." << std::flush;
        img = hdcalib::crop(img, crop_x_arg.getValue(), crop_y_arg.getValue());
        std::cout << " done. Size: " << img.size() << std::endl;
        std::cout << "Scaling..." << std::flush;
        img = hdcalib::downscale(img, factor_arg.getValue());
        std::cout << " done. Size: " << img.size() << std::endl;
        std::cout << "Saving..." << std::flush;
        cv::imwrite(file
                    + "-f-" + tostringLZ(factor_arg.getValue())
                    + "-x-" + tostringLZ(crop_x_arg.getValue())
                    + "-y-" + tostringLZ(crop_y_arg.getValue()) + ".png",
                    img
                    );
        std::cout << " done." << std::endl;

        std::cout << std::endl;
    }

}
