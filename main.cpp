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
#include <ParallelTime/paralleltime.h>

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

#define TIMELOG(descr) { \
    time_log << descr << ": " << t.print() << std::endl;\
    t.start();\
    std::cout << time_log.str() << "Total time: " << total_time.print() << std::endl << std::endl;\
    }

int main(int argc, char* argv[]) {
    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    bool removed = false;

    hdcalib::Calib calib;
    std::vector<std::string> input_files;
    int recursion_depth = -1;
    float effort = 0.5;
    bool demosaic = false;
    bool libraw = false;
    bool plot_markers = false;
    bool only_green = false;
    bool verbose = true;
    bool gnuplot = false;
    std::vector<int> valid_pages;
    std::string cache_file;
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

        TCLAP::ValueArg<std::string> cache_arg("c", "cache",
                                               "Cache file for the calibration results. This makes use of the opencv filestorage capabilities so filename extension should be .xml/.xml.gz/.yaml/.yaml.gz",
                                               false, "", "Calibration cache.");
        cmd.add(cache_arg);

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

        TCLAP::SwitchArg gnuplot_arg("", "gnuplot", "Use gnuplot for plotting residuals etc."
                                     , false);
        cmd.add(gnuplot_arg);

        TCLAP::MultiArg<std::string> textfile_arg("i",
                                                  "input",
                                                  "Text file containing a list of image paths relative to the working directory.",
                                                  false,
                                                  "Text file with a list of input images.");
        cmd.add(textfile_arg);

        TCLAP::MultiArg<int> valid_pages_arg("",
                                                  "valid",
                                                  "Page number of a valid corner.",
                                                  false,
                                                  "Page number of a valid corner.");
        cmd.add(valid_pages_arg);

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
        cache_file = cache_arg.getValue();
        gnuplot = gnuplot_arg.getValue();
        valid_pages = valid_pages_arg.getValue();

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
                  << "only green channel: " << (only_green ? "true" : "false") << std::endl
                  << "Gnuplot: " << (gnuplot ? "true" : "false") << std::endl
                  << "Valid pages: ";
        for (const auto it : valid_pages) {
            std::cout << it << "  ";
        }
        std::cout << std::endl;
        calib.setValidPages(valid_pages);


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

    std::map<std::string, std::vector<hdmarker::Corner> > detected_markers;

    calib.setRecursionDepth(recursion_depth);

    TIMELOG("Argument parsing");

    bool has_cached_calib = false;
    if (fs::is_regular_file(cache_file)) {
        try {
            if (verbose) {
                std::cout << "Reading cached calibration results..." << std::flush;
            }
            cv::FileStorage fs(cache_file, cv::FileStorage::READ);
            cv::FileNode n = fs["calibration"];
            n >> calib;
            has_cached_calib = true;
            fs.release();
            std::cout << " done." << std::endl;
            calib.purgeInvalidPages();
        }
        catch (std::exception const& e) {
            std::cout << "Reading cache file failed with exception:" << std::endl
                      << e.what() << std::endl;
        }
        TIMELOG("Reading cached result");
    }

    if (has_cached_calib) {
        removed = calib.removeOutliers(2);

        TIMELOG("removeOutliers() #1");

        if (removed) {
            calib.CeresCalibFlexibleTarget();

            TIMELOG("CeresCalibFlexibleTarget()");
        }
        bool found_new_files = false;
#pragma omp parallel for schedule(dynamic)
        for (size_t ii = 0; ii < input_files.size(); ++ii) {
            std::string const& input_file = input_files[ii];
            if (!calib.hasFile(input_file)) {
                try {
                    detected_markers[input_file] = calib.getCorners(input_file, effort, demosaic, libraw);
                }
                catch (const std::exception &e) {
                    std::cout << "Reading file " << input_file << " failed with an exception: " << std::endl
                              << e.what() << std::endl;
                }
            }
        }

        TIMELOG("Reading missing files");

        for (auto const& it : detected_markers) {
            if (!calib.hasFile(it.first)) {
                calib.addInputImageAfterwards(it.first, it.second);
                found_new_files = true;
            }
        }

        TIMELOG("Adding missing files");

        if (found_new_files) {
            calib.CeresCalibFlexibleTarget();

            TIMELOG("CeresCalibFlexibleTarget()");

            removed = calib.removeOutliers(2);

            TIMELOG("removeOutliers");

            if (removed) {
                calib.CeresCalibFlexibleTarget();

                TIMELOG("CeresCalibFlexibleTarget()");
            }
        }

        calib.printObjectPointCorrectionsStats();

        TIMELOG("printObjectPointCorrectionsStats");

        if (gnuplot) {
            calib.plotReprojectionErrors("", "ceres3");
            TIMELOG("plotReprojectionErrors ceres3");
        }

        cv::FileStorage fs(cache_file, cv::FileStorage::WRITE);
        fs << "calibration" << calib;
        fs.release();
        // */
        return EXIT_SUCCESS;
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < input_files.size(); ++ii) {
        std::string const& input_file = input_files[ii];
        try {
            detected_markers[input_file] = calib.getCorners(input_file, effort, demosaic, libraw);
        }
        catch (const std::exception &e) {
            std::cout << "Reading file " << input_file << " failed with an exception: " << std::endl
                      << e.what() << std::endl;
        }
    }

    TIMELOG("Reading markers")


    for (auto const& it : detected_markers) {
        calib.addInputImage(it.first, it.second);
    }

    TIMELOG("Adding input images")

    calib.purgeInvalidPages();

    TIMELOG("Purging invalid pages")

    calib.openCVCalib();

    TIMELOG("openCVCalib")

    if (gnuplot) {
        calib.plotReprojectionErrors("", "initial");
        TIMELOG("plotReprojectionErrors initial")
    }

    calib.prepareCalibration();

    TIMELOG("prepareCalibration")

    if (gnuplot) {
        calib.plotReprojectionErrors("", "initial-all-markers");
        TIMELOG("plotReprojectionErrors initial all markers")
    }

    calib.CeresCalib();

    TIMELOG("CeresCalib")

    if (gnuplot) {
        calib.plotReprojectionErrors("", "ceres");
        TIMELOG("plotReprojectionErrors ceres")
    }

    calib.removeOutliers(150);

    TIMELOG("removeOutliers(150)")

    calib.CeresCalib();

    TIMELOG("CeresCalib")

    if (gnuplot) {
        calib.plotReprojectionErrors("", "ceres2");
        TIMELOG("plotReprojectionErrors ceres2")
    }

    calib.CeresCalibFlexibleTarget();

    TIMELOG("CeresCalibFlexibleTarget")

    removed = calib.removeOutliers(2);

    TIMELOG("removeOutliers(2)")

    if (removed) {
        calib.CeresCalibFlexibleTarget();

        TIMELOG("CeresCalibFlexibleTarget")
    }

    calib.printObjectPointCorrectionsStats();

    TIMELOG("printObjectPointCorrectionsStats")

    if (gnuplot) {
        calib.plotReprojectionErrors("", "ceres3");
        TIMELOG("plotReprojectionErrors ceres3")
    }

    if (!cache_file.empty()) {
        cv::FileStorage fs(cache_file, cv::FileStorage::WRITE);
        fs << "calibration" << calib;
        fs.release();
        TIMELOG("Writing cache file")
    }

    std::cout << "Level 1 log entries: " << std::endl;
    clog::Logger::getInstance().printAll(std::cout, 1);

    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
