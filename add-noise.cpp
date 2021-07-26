#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/filesystem.hpp>

#include <tclap/CmdLine.h>

#include "randutils.hpp"

#include "libdownscale.h"

int main(int argc, char ** argv) {

    TCLAP::CmdLine cmd("Tool for adding noise and distorting images");

    TCLAP::UnlabeledMultiArg<std::string> images_arg("images", "List of images to add noise and / or distort", true, "filename");
    cmd.add(images_arg);

    TCLAP::ValueArg<double> noise_arg("n", "noise", "noise factor, noise added is normal distributed with zero mean and a standard deviation of n*sqrt(val)", false, 1.0, "float, >0");
    cmd.add(noise_arg);

    TCLAP::ValueArg<double> border_arg("a", "add-border", "add border (in percent)", false, 0, "percentage, >=0");
    cmd.add(border_arg);

    TCLAP::ValueArg<std::string> output_dir_arg("o", "output-directory", "output directory", false, "out", "directory name");
    cmd.add(output_dir_arg);

    TCLAP::ValueArg<double> distortion_arg("d", "distortion", "k1 parameter of the radial distortion model", false, 0.0, "float");
    cmd.add(distortion_arg);

    TCLAP::ValueArg<double> downscale_arg("s", "downscale", "downscale image by an integer factor", false, 0, "integer > 0");
    cmd.add(downscale_arg);

    TCLAP::ValueArg<double> gauss_arg("g", "gaussian", "Use Gaussian smoothing as preprocessing", false, 0, "float > 0");
    cmd.add(gauss_arg);

    cmd.parse(argc, argv);

    double const noise = std::max<double>(0, noise_arg.getValue());

    int const downscale = downscale_arg.getValue();

    std::vector<std::string> const images = images_arg.getValue();

    namespace fs = boost::filesystem;
    fs::path output = output_dir_arg.getValue();
    boost::system::error_code ignore_error_code;
    fs::create_directories(output, ignore_error_code);

    cv::Mat_<cv::Vec2f> map1, map2;
    randutils::mt19937_rng rng;
    for (size_t ii = 0; ii < images.size(); ++ii) {
        std::cout << "Reading " << images[ii] << "..." << std::flush;
        cv::Mat_<uint16_t> img = cv::imread(images[ii], cv::IMREAD_UNCHANGED);
        std::cout << "done." << std::endl;

        if (border_arg.isSet() && border_arg.getValue() > 0) {
            int const vertical = std::round(double(img.rows) * border_arg.getValue() / 100);
            int const horizontal = std::round(double(img.cols) * border_arg.getValue() / 100);
            cv::copyMakeBorder(img, img, vertical, vertical, horizontal, horizontal, cv::BORDER_CONSTANT);
        }
        cv::Mat_<float> camera_matrix(3, 3, float(0.0));
        camera_matrix(0,0) = camera_matrix(1,1) = std::min(img.rows, img.cols);
        camera_matrix(0,2) = double(img.cols + 1)/2;
        camera_matrix(1,2) = double(img.rows + 1)/2;
        camera_matrix(2,2) = 1;
        if (distortion_arg.isSet() && distortion_arg.getValue() != 0) {
            cv::Mat_<float> distCoeffs(1,4, float(0));
            distCoeffs(0,0) = distortion_arg.getValue();
            if (map1.size() != img.size()) {
                std::cout << "Calculating distortion map..." << std::flush;
                cv::initInverseRectificationMap(camera_matrix, distCoeffs, cv::Mat(), camera_matrix, img.size(), CV_32FC2, map1, map2);
                std::cout << std::endl;
            }
            //cv::initUndistortRectifyMap(camera_matrix, distCoeffs, cv::Mat(), camera_matrix, img.size(), CV_32FC2, map1, map2);
            std::cout << "Distorting..." << std::flush;
            cv::remap(img, img, map1, map2, cv::INTER_LINEAR);
            std::cout << "done." << std::endl;
        }
        if (gauss_arg.isSet()) {
            double const gauss = std::max(0.01, gauss_arg.getValue());
            size_t size = std::max<size_t>(3, std::ceil(gauss*3));
            if (0 == size % 2) {
                size++;
            }
            cv::GaussianBlur(img, img, cv::Size(size, size), gauss, gauss);
        }
        if (downscale > 1) {
            img = hdcalib::downscale(img, downscale);
        }
        std::cout << "Adding noise..." << std::flush;
        if (noise_arg.isSet()) {
            for (uint16_t & val : img) {
                double new_val = val;
                new_val += rng.variate<double, std::normal_distribution>(0, noise*std::sqrt(double(val)));
                val = cv::saturate_cast<uint16_t>(std::round(new_val));
            }
        }
        std::cout << std::endl;
        std::cout << "Writing output..." << std::flush;
        cv::imwrite((output / images[ii]).string(), img);
        std::cout << std::endl << std::endl;
    }
}
