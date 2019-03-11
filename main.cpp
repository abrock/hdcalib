/**
* @file main.cpp
* @brief demo calibration application of hdmarker/hdcalib
*
* @author Alexander Brock
* @date 02/20/2019
*/

#include <exception>

#include <tclap/CmdLine.h>

#include "calib.h"

int main(int argc, char* argv[]) {

    hdcalib::Calib cal;
    std::vector<std::string> input_files;
    int recursion_depth = -1;
    float effort = 0.5;
    bool demosaic = false;
    bool libraw = false;
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

        TCLAP::UnlabeledMultiArg<std::string> input_img_arg("input", "Input images, should contain markers", true, "string");
        cmd.add(input_img_arg);


        cmd.parse(argc, argv);


        input_files = input_img_arg.getValue();
        recursion_depth = recursive_depth_arg.getValue();
        effort = effort_arg.getValue();
        libraw = read_raw_arg.getValue();
        demosaic = demosaic_arg.getValue() || libraw;

        std::cout << "Parameters: " << std::endl
                  << "Number of input files: " << input_files.size() << std::endl
                  << "recursion depth: " << recursion_depth << std::endl
                  << "effort: " << effort << std::endl
                  << "demosaic: " << (demosaic ? "true" : "false") << std::endl
                  << "use libraw: " << (libraw ? "true" : "false") << std::endl;
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
            detected_markers[input_file] = cal.getCorners(input_file, effort, demosaic, recursion_depth, libraw);
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


    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
