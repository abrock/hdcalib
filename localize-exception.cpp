/**
* @file localize-exception.cpp
* @brief  hdmarker/hdcalib
*
* @author Alexander Brock
* @date 02/20/2019
*/

#include <exception>

#include <tclap/CmdLine.h>

#include "hdcalib.h"

#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>

#include <pstreams/pstream.h>

std::string exec(std::string const command) {
    // run a process and create a streambuf that reads its stdout and stderr
    redi::ipstream proc(command, redi::pstreams::pstdout | redi::pstreams::pstderr);
    std::string line;
    // read child's stdout
    std::string result;
    while (std::getline(proc.out(), line)) {
        std::cout << "stdout: " << line << std::endl;
        result += line;
    }
    // read child's stderr
    while (std::getline(proc.err(), line)) {
        std::cout << "stderr: " << line << std::endl;
        result += line;
    }
    return result;
}

void intersect(cv::Mat const& img, cv::Rect & roi) {
    if (roi.x < 0) roi.x = 0;
    if (roi.y < 0) roi.y = 0;
    if (roi.x > img.cols) roi.x = img.cols;
    if (roi.y > img.rows) roi.y = img.rows;
    if (roi.x + roi.width > img.cols) roi.width = img.cols - roi.x;
    if (roi.y + roi.height > img.rows) roi.height = img.rows - roi.y;
}

void localizeException(cv::Mat _img, std::string const name, int const recursion_depth = 1, float const effort = .5) {
    if (_img.cols < 10 || _img.rows < 10) {
        return;
    }
    std::cout << "Testing " << name << " with size " << _img.size << std::endl;
    cv::Mat img = _img.clone();
    cv::imwrite(name + ".png", img);
    std::vector<hdmarker::Corner> corners;
    std::vector<hdmarker::Corner> corners2;
    bool const use_rgb = false;
    try {
        hdmarker::detect(img, corners, use_rgb, 0, 10, effort, 3);
        if (recursion_depth > 0) {
            double msize = 1.0;
            hdmarker::refine_recursive(img, corners, corners2, 3, &msize);
        }
    }
    catch (hdmarker::runaway_subpattern const& e) {
        cv::imwrite(name + ".png", img);
        std::ofstream out(name + ".txt");
        out << "File: " << name + ".png" << std::endl
            << "size: " << img.size << std::endl
            << "Exception: " << e.what() << std::endl;
        out.close();
        if (img.rows > img.cols) {
            cv::Rect top(0,0, img.cols, img.rows/2);
            cv::Rect bottom(0,img.rows/2, img.cols, img.rows/2+1);
            cv::Rect middle(0,img.rows/4, img.cols, (img.rows/4)*3);
            intersect(img, top);
            intersect(img, bottom);
            intersect(img, middle);
            cv::Mat t = img(top).clone();
            cv::Mat b = img(bottom).clone();
            cv::Mat m = img(middle).clone();
            localizeException(t, name + "-t", recursion_depth, effort);
            localizeException(b, name + "-b", recursion_depth, effort);
            localizeException(m, name + "-m", recursion_depth, effort);
        }
        else {
            cv::Rect left(0,0, img.cols/2, img.rows);
            cv::Rect right(img.cols/2,0, img.cols/2+1, img.rows);
            cv::Rect middle(img.rows/4,0, (img.cols/4)*3, img.rows);
            intersect(img, left);
            intersect(img, right);
            intersect(img, middle);
            cv::Mat l = img(left).clone();
            cv::Mat r = img(right).clone();
            cv::Mat m = img(middle).clone();
            localizeException(l, name + "-l", recursion_depth, effort);
            localizeException(r, name + "-r", recursion_depth, effort);
            localizeException(m, name + "-m", recursion_depth, effort);
        }
        return;
    }
}

void localizeExceptionExternal(
        cv::Mat _img,
        std::string const name,
        std::string const executable,
        int const recursion_depth = 1,
        float const effort = .5) {
    if (_img.cols < 10 || _img.rows < 10) {
        return;
    }
    std::cout << "Testing " << name << " with size " << _img.size << std::endl;
    cv::Mat img = _img.clone();
    cv::imwrite("tmp.png", img);
    std::vector<hdmarker::Corner> corners;
    std::vector<hdmarker::Corner> corners2;
    bool const use_rgb = false;
    std::string result = exec(std::string("\"") + executable + + "\" tmp.png tmp-out.png");
    bool exists = result.find("hdmarker::runaway_subpattern") != std::string::npos;
    exists |= (result.find("ERROR: area already covered from different idx!") != std::string::npos);
    if (exists) {
        cv::imwrite(name + ".png", img);
        std::ofstream out(name + ".txt");
        out << "File: " << name + ".png" << std::endl
            << "size: " << img.size << std::endl
            << "output: " << result << std::endl;
        out.close();
        if (img.rows > img.cols) {
            cv::Rect top(0,0, img.cols, img.rows/2);
            cv::Rect bottom(0,img.rows/2, img.cols, img.rows/2+1);
            cv::Rect middle(0,img.rows/4, img.cols, (img.rows/4)*3);
            intersect(img, top);
            intersect(img, bottom);
            intersect(img, middle);
            cv::Mat t = img(top).clone();
            cv::Mat b = img(bottom).clone();
            cv::Mat m = img(middle).clone();
            localizeExceptionExternal(t, name + "-t", executable, recursion_depth, effort);
            localizeExceptionExternal(b, name + "-b", executable, recursion_depth, effort);
            localizeExceptionExternal(m, name + "-m", executable, recursion_depth, effort);
        }
        else {
            cv::Rect left(0,0, img.cols/2, img.rows);
            cv::Rect right(img.cols/2,0, img.cols/2+1, img.rows);
            cv::Rect middle(img.rows/4,0, (img.cols/4)*3, img.rows);
            intersect(img, left);
            intersect(img, right);
            intersect(img, middle);
            cv::Mat l = img(left).clone();
            cv::Mat r = img(right).clone();
            cv::Mat m = img(middle).clone();
            localizeExceptionExternal(l, name + "-l", executable, recursion_depth, effort);
            localizeExceptionExternal(r, name + "-r", executable, recursion_depth, effort);
            localizeExceptionExternal(m, name + "-m", executable, recursion_depth, effort);
        }
        return;
    }
}
int main(int argc, char* argv[]) {

    hdcalib::Calib calib;
    std::string input_file, executable;
    int recursion_depth = -1;
    float effort = 0.5;
    bool demosaic = false;
    bool libraw = false;
    bool plot_markers = false;
    bool only_green = false;
    try {
        TCLAP::CmdLine cmd("hdcalib calibration tool", ' ', "0.1");

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 false, -1, "int");
        cmd.add(recursive_depth_arg);

        TCLAP::ValueArg<float> effort_arg("e", "effort",
                                          "Effort value for the marker detection.",
                                          false, .5, "float");
        cmd.add(effort_arg);

        TCLAP::SwitchArg demosaic_arg("d", "demosaic",
                                      "Use this flag if the input images are raw images and demosaicing should be used.",
                                      false);
        cmd.add(demosaic_arg);

        TCLAP::SwitchArg read_raw_arg("", "raw",
                                      "Use this flag if the input images are raw images which must be read using LibRaw since OpenCV cannot read them. This implies -d.",
                                      false);
        cmd.add(read_raw_arg);

        TCLAP::SwitchArg only_green_arg("", "only-green", "Set this flag true if only the green channel of a bayer image should be used. This implies demosaic, in this case bilinear demosaicing of the green channel only.", false);
        cmd.add(only_green_arg);

        TCLAP::ValueArg<std::string> input_img_arg("i", "input", "Input image, should contain markers", true, "", "Input image");
        cmd.add(input_img_arg);

        TCLAP::ValueArg<std::string> executable_arg("", "ex", "path of the extractMarker executable", true, "", "path of the extractMarker executable");
        cmd.add(executable_arg);

        cmd.parse(argc, argv);


        input_file = input_img_arg.getValue();
        recursion_depth = recursive_depth_arg.getValue();
        effort = effort_arg.getValue();
        libraw = read_raw_arg.getValue();
        only_green = only_green_arg.getValue();
        demosaic = demosaic_arg.getValue() || libraw || only_green;
        executable = executable_arg.getValue();

        std::cout << "Parameters: " << std::endl
                  << "Input file: " << input_file << std::endl
                  << "recursion depth: " << recursion_depth << std::endl
                  << "effort: " << effort << std::endl
                  << "demosaic: " << (demosaic ? "true" : "false") << std::endl
                  << "use libraw: " << (libraw ? "true" : "false") << std::endl
                  << "only green channel: " << (only_green ? "true" : "false") << std::endl;

        calib.setPlotMarkers(plot_markers);
        calib.only_green(only_green);
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    cv::Mat img;
    if (demosaic) {
        if (libraw) {
            img = calib.read_raw(input_file);
        }
        else {
            img = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
        }
        double min_val = 0, max_val = 0;
        cv::minMaxIdx(img, &min_val, &max_val);
        std::cout << "Image min/max: " << min_val << " / " << max_val << std::endl;
        img = img * (65535 / max_val);

        //cv::normalize(img, img, 0, 255, NORM_MINMAX, CV_8UC1);
        cvtColor(img, img, cv::COLOR_BayerBG2BGR); // RG BG GB GR
        if (only_green) {
            cv::Mat split[3];
            cv::split(img, split);
            img = split[1];
        }
        cv::minMaxIdx(img, &min_val, &max_val);
        std::cout << "Image min/max: " << min_val << " / " << max_val << std::endl;
        img = img * (255.0 / max_val);
        img.convertTo(img, CV_8UC1);
    }
    else {
        img = cv::imread(input_file);
    }

    localizeExceptionExternal(img, input_file, executable, recursion_depth, effort);


    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
