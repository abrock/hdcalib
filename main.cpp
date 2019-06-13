/**
* @file main.cpp
* @brief demo calibration application of hdmarker/hdcalib
*
* @author Alexander Brock
* @date 02/20/2019
*/

#include <exception>
#include <boost/filesystem.hpp>

#include <tclap/CmdLine.h>

#include "hdcalib.h"

namespace fs = boost::filesystem;

void trim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

int main(int argc, char* argv[]) {

    hdcalib::Calib calib;
    std::vector<std::string> input_files;
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

        TCLAP::SwitchArg plot_markers_arg("p", "plot", "Use this flag if the detected markers should be painted into the input images", false);
        cmd.add(plot_markers_arg);

        TCLAP::SwitchArg only_green_arg("g", "only-green", "Set this flag true if only the green channel of a bayer image should be used."
                                                           "In the case of demosaicing this means that the missing green pixels"
                                                           "are interpolated bilinear.", false);
        cmd.add(only_green_arg);

        TCLAP::MultiArg<std::string> textfile_arg("i",
                                                  "input",
                                                  "Text file containing a list of image paths relative to the working directory.",
                                                  false,
                                                  "Text file with a list of input images.");
        cmd.add(textfile_arg);

        TCLAP::UnlabeledMultiArg<std::string> input_img_arg("input_img", "Input images, should contain markers.", false, "Input images.");
        cmd.add(input_img_arg);

        cmd.parse(argc, argv);


        input_files = input_img_arg.getValue();
        recursion_depth = recursive_depth_arg.getValue();
        effort = effort_arg.getValue();
        libraw = read_raw_arg.getValue();
        only_green = only_green_arg.getValue();
        demosaic = demosaic_arg.getValue() || libraw;
        plot_markers = plot_markers_arg.getValue();
        std::vector<std::string> const textfiles = textfile_arg.getValue();

        for (std::string const& file : textfiles) {
            if (!fs::is_regular_file(file)) {
                continue;
            }
            std::ifstream in(file);
            std::string line;
            while (std::getline(in, line)) {
                trim(line);
                if (fs::is_regular_file(line)) {
                    input_files.push_back(line);
                }
            }
        }

        if (input_files.empty()) {
            std::cerr << "Fatal error: No input files specified." << std::endl;
            cmd.getOutput()->usage(cmd);

            return EXIT_FAILURE;
        }

        std::cout << "Parameters: " << std::endl
                  << "Number of input files: " << input_files.size() << std::endl
                  << "recursion depth: " << recursion_depth << std::endl
                  << "effort: " << effort << std::endl
                  << "demosaic: " << (demosaic ? "true" : "false") << std::endl
                  << "use libraw: " << (libraw ? "true" : "false") << std::endl
                  << "plot markers: " << (plot_markers ? "true" : "false") << std::endl
                  << "only green channel: " << (only_green ? "true" : "false") << std::endl;

        calib.plotMarkers(plot_markers);
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

    std::map<std::string, std::vector<hdmarker::Corner> > detected_markers;

#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < input_files.size(); ++ii) {
        std::string const& input_file = input_files[ii];
        try {
            detected_markers[input_file] = calib.getCorners(input_file, effort, demosaic, recursion_depth, libraw);
        }
        catch (const std::exception &e) {
            std::cout << "Reading file " << input_file << " failed with an exception: " << std::endl
                      << e.what() << std::endl;
        }
    }

    std::ofstream duplicate_markers("duplicate-markers.log");
    for (auto const& it : detected_markers) {
        for (size_t ii = 0; ii < it.second.size(); ++ii) {
            for (size_t jj = ii+1; jj < it.second.size(); ++jj) {
                cv::Point2f residual = it.second[ii].p - it.second[jj].p;
                double const dist = std::sqrt(residual.dot(residual));
                if (dist < (it.second[ii].size + it.second[jj].size)/20) {
                    duplicate_markers << it.first << ": "
                                      << it.second[ii].id << ", p. " << it.second[ii].page << " vs. "
                                      << it.second[jj].id << ", p. " << it.second[jj].page << std::endl;
                }
            }
        }
    }

    for (auto const& it : detected_markers) {
        calib.addInputImage(it.first, it.second);
    }

    calib.openCVCalib();

    calib.plotReprojectionErrors("", "initial");

    calib.removeOutliers(1.5);

    calib.plotReprojectionErrors("", "outliers");

    calib.CeresCalib();

    calib.plotReprojectionErrors("", "ceres");

    calib.CeresCalibFlexibleTarget();

    calib.printObjectPointCorrectionsStats();

    calib.plotReprojectionErrors("", "ceres2");

    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
