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

template<class T>
std::vector<T> commaSeparate(std::vector<std::string> const& args) {
    std::vector<T> result;
    for (std::string const& s : args) {
        std::string current;
        for (char c : s) {
            if (c == ',' || c == ';') {
                trim(current);
                if (!current.empty()) {
                    std::stringstream stream(current);
                    T val;
                    stream >> val;
                    result.push_back(val);
                    current = "";
                }
            }
            else {
                current += c;
            }
        }
        trim(current);
        if (!current.empty()) {
            std::stringstream stream(current);
            T val;
            stream >> val;
            result.push_back(val);
        }
    }
    return result;
}

#define TIMELOG(descr) { \
    time_log << descr << ": " << t.print() << std::endl;\
    t.start();\
    std::cout << time_log.str() << "Total time: " << total_time.print() << std::endl << std::endl;\
    }

int main(int argc, char* argv[]) {
    std::ofstream logfile("hdcalib.log", std::ofstream::out);

    clog::Logger::getInstance().addListener(std::cout);
    clog::Logger::getInstance().addListener(logfile);

    ParallelTime t, total_time;
    std::stringstream time_log;

    bool calib_updated = false;

    hdcalib::Calib calib;
    std::vector<std::string> input_files;
    int recursion_depth = -1;
    float effort = 0.5;
    bool demosaic = false;
    bool libraw = false;
    bool plot_markers = false;
    bool only_green = false;
    bool gnuplot = false;
    std::string del = "";
    std::vector<int> valid_pages;
    std::vector<std::string> calibration_types;
    std::vector<std::string> calibration_types_eval;
    std::vector<std::string> same_pos_suffixes;
    std::string cache_file;
    std::string cache_file_prefix;
    double outlier_threshold = -1;
    double max_outlier_percentage = 105;
    double cauchy_param = 1;
    double marker_size = 1;
    double min_snr = 5;
    try {
        TCLAP::CmdLine cmd("hdcalib calibration tool", ' ', "0.1");

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. "
                                                 "Set this to the actual recursion depth of the used target.",
                                                 false, -1, "int");
        cmd.add(recursive_depth_arg);

        TCLAP::ValueArg<float> effort_arg("e", "effort",
                                          "Effort value for the marker detection.",
                                          false, .5, "float");
        cmd.add(effort_arg);

        TCLAP::ValueArg<float> cauchy_param_arg("", "cauchy",
                                                "Scale parameter (px) for the cauchy loss. Use negative numbers to disable the robust solving (not recommended "
                                                "except you know exactly what you're doing).",
                                                false, 1, "cauchy loss scale [px]");
        cmd.add(cauchy_param_arg);

        TCLAP::ValueArg<float> marker_size_arg("m", "marker-size",
                                               "Physical size (width/height) of the main markers.",
                                               false, 1, "physical marker size");
        cmd.add(marker_size_arg);

        TCLAP::ValueArg<float> max_outlier_arg("", "mout",
                                               "Maximum outlier percentage for an image to be included in the calibration.",
                                               false, 105, "max. outlier percentage");
        cmd.add(max_outlier_arg);

        TCLAP::ValueArg<double> ceres_tolerance_arg("", "ceres-tol",
                                                    "Value to be used for function, gradient- and parameter tolerance by the Ceres solver.",
                                                    false, 1e-10, "max. outlier percentage");
        cmd.add(ceres_tolerance_arg);

        TCLAP::ValueArg<float> outlier_threshold_arg("", "thresh",
                                                     "Maximum error from previous calibrations for a marker to be included in the calibration.",
                                                     false, -1, "max. error threshold");
        cmd.add(outlier_threshold_arg);

        TCLAP::ValueArg<std::string> cache_arg("c", "cache",
                                               "Cache filename prefix for the calibration results. "
                                               ".yaml.gz will be appended.",
                                               false, "", "Calibration cache.");
        cmd.add(cache_arg);

        TCLAP::ValueArg<double> min_snr_arg("", "min-snr",
                                               "Minimum SNR (x sigma) for a marker to be used.",
                                               false, 5, "Minimum SNR value.");
        cmd.add(min_snr_arg);

        TCLAP::ValueArg<double> max_iter_arg("", "max-iter",
                                               "Maximum number of iterations used in the calibrations using ceres.",
                                               false, 1000, "Max #iterations.");
        cmd.add(max_iter_arg);

        TCLAP::ValueArg<std::string> delete_arg("", "delete",
                                                "Specify a calibration result to delete from the cached calibration. ",
                                                false, "", "Calibration type.");
        cmd.add(delete_arg);

        TCLAP::MultiArg<std::string> type_arg("t", "type",
                                              "Type of the calibration(s) to run. "
                                              "Possibilities in increasing order of computational complexity:"
                                              "SimpleOpenCV, SimpleCeres, OpenCV, Ceres, Flexible, SemiFlexible ",
                                              false, "Calibration type.");
        cmd.add(type_arg);

        TCLAP::MultiArg<std::string> eval_arg("", "eval",
                                              "Type of the calibration(s) to evaluate. "
                                              "Possibilities in increasing order of computational complexity:"
                                              "SimpleOpenCV, SimpleCeres, OpenCV, Ceres, Flexible, SemiFlexible ",
                                              false, "Calibration types for evaluation.");
        cmd.add(eval_arg);

        TCLAP::MultiArg<std::string> same_pos_arg("", "same",
                                                  "This option allows the user to verify if two shots show the same target position. "
                                                  "The option specifies a suffix for the full path of the shot. "
                                                  "Specify at least twice in order to name two path suffixes which are expected to show identical target positions. "
                                                  "This is meant to be used for checking if the target accidentally moved while a motion stage was moving the camera, "
                                                  "or to test the repeatability of the motion stage.",
                                                  false, "Same position suffix.");
        cmd.add(same_pos_arg);

        TCLAP::SwitchArg demosaic_arg("d", "demosaic",
                                      "Use this flag if the input images are raw images and demosaicing should be used.",
                                      false);
        cmd.add(demosaic_arg);

        TCLAP::SwitchArg read_raw_arg("", "raw",
                                      "Use this flag if the input images are raw images "
                                      "which must be read using LibRaw since OpenCV cannot read them. "
                                      "This implies -d.",
                                      false);
        cmd.add(read_raw_arg);

        TCLAP::SwitchArg plot_markers_arg("p", "plot",
                                          "Use this flag if the detected markers "
                                          "should be painted into the input images", false);
        cmd.add(plot_markers_arg);

        TCLAP::SwitchArg only_green_arg("g", "only-green",
                                        "Set this flag true if only the green channel of a bayer image should be used."
                                        "In the case of demosaicing this means that the missing green pixels"
                                        "are interpolated bilinear.", false);
        cmd.add(only_green_arg);

        TCLAP::SwitchArg gnuplot_arg("", "gnuplot", "Use gnuplot for plotting residuals etc."
                                     , false);
        cmd.add(gnuplot_arg);

        TCLAP::MultiArg<std::string> textfile_arg("i", "input",
                                                  "Text file containing a list of image paths "
                                                  "relative to the working directory.",
                                                  false,
                                                  "Text file with a list of input images.");
        cmd.add(textfile_arg);


        TCLAP::MultiArg<std::string> valid_pages_arg("",
                                             "valid",
                                             "Page number(s) of valid corners.",
                                             false,
                                             "Page number(s) of valid corners.");
        cmd.add(valid_pages_arg);

        TCLAP::UnlabeledMultiArg<std::string> input_img_arg("input_img",
                                                            "Input images, should contain markers.",
                                                            false,
                                                            "Input images.");
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
        cache_file_prefix = cache_arg.getValue();
        cache_file = cache_file_prefix + ".yaml.gz";
        gnuplot = gnuplot_arg.getValue();
        valid_pages = commaSeparate<int>(valid_pages_arg.getValue());
        calibration_types = commaSeparate<std::string>(type_arg.getValue());
        calibration_types_eval = commaSeparate<std::string>(eval_arg.getValue());
        same_pos_suffixes = same_pos_arg.getValue();
        max_outlier_percentage = max_outlier_arg.getValue();
        del = delete_arg.getValue();
        outlier_threshold = outlier_threshold_arg.getValue();
        cauchy_param = cauchy_param_arg.getValue();
        min_snr = min_snr_arg.getValue();
        calib.setMaxIter(std::max<size_t>(1, max_iter_arg.getValue()));

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

        std::stringstream str_pages;
        for (const auto it : valid_pages) {
            str_pages << it << "  ";
        }
        clog::L("tclap", 2) << "Parameters: " << std::endl
                            << "Number of input files: " << input_files.size() << std::endl
                            << "recursion depth: " << recursion_depth << std::endl
                            << "effort: " << effort << std::endl
                            << "demosaic: " << (demosaic ? "true" : "false") << std::endl
                            << "use libraw: " << (libraw ? "true" : "false") << std::endl
                            << "plot markers: " << (plot_markers ? "true" : "false") << std::endl
                            << "only green channel: " << (only_green ? "true" : "false") << std::endl
                            << "Gnuplot: " << (gnuplot ? "true" : "false") << std::endl
                            << "Valid pages: " << str_pages.str() << std::endl
                            << "Min SNR*sigma: " << min_snr;
        calib.setValidPages(valid_pages);
        calib.setMinSNR(min_snr);

        calib.setCeresTolerance(ceres_tolerance_arg.getValue());


        calib.setPlotMarkers(plot_markers);
        calib.only_green(only_green);
        marker_size = marker_size_arg.getValue();
        if (marker_size > 0) {
            calib.setMarkerSize(marker_size);
        }
        calib.setCauchyParam(cauchy_param);
        calib.setRecursionDepth(recursion_depth);
        calib.setMaxOutlierPercentage(max_outlier_percentage);
        calib.setUseRaw(libraw);
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

    TIMELOG("Argument parsing");

    bool has_cached_calib = false;
    if (!cache_file.empty() && fs::is_regular_file(cache_file)) {
#if CATCH_STORAGE
        try {
#endif
            clog::L("main", 2) << "Reading cached calibration results from file " << cache_file << std::endl;
            cv::FileStorage fs(cache_file, cv::FileStorage::READ);
            cv::FileNode n = fs["calibration"];
            n >> calib;
            if (marker_size > 0) {
                calib.setMarkerSize(marker_size);
            }
            has_cached_calib = true;
            fs.release();
            calib.purgeInvalidPages();
            if (!del.empty()) {
                calib.deleteCalib(del);
                calib_updated = true;
            }
            clog::L(__func__, 2) << calib.printAllCameraMatrices() << std::endl;

#if CATCH_STORAGE
        }
        catch (std::exception const& e) {
            clog::L("main", 1) << "Reading cache file failed with exception:" << std::endl
                               << e.what() << std::endl;
        }
#endif
        TIMELOG("Reading cached result");
    }

    if (has_cached_calib) {
        bool found_new_files = false;
#pragma omp parallel for schedule(dynamic)
        for (size_t ii = 0; ii < input_files.size(); ++ii) {
            std::string const& input_file = input_files[ii];
            if (!calib.hasFile(input_file)) {
#if CATCH_EXCEPTIONS
                try {
#endif
                    std::vector<hdmarker::Corner> corners = calib.getCorners(input_file, effort, demosaic, libraw);
#pragma omp critical
                    {
                        detected_markers[input_file] = corners;
                    }
#if CATCH_EXCEPTIONS
                }
                catch (const std::exception &e) {
                    clog::L("main", 1) << "Reading file " << input_file << " failed with an exception: " << std::endl
                                       << e.what() << std::endl;
                }
#endif
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
    }
    else {
#pragma omp parallel for schedule(dynamic)
        for (size_t ii = 0; ii < input_files.size(); ++ii) {
            std::string const& input_file = input_files[ii];
            try {
                std::vector<hdmarker::Corner> const corners = calib.getCorners(input_file, effort, demosaic, libraw);
#pragma omp critical
                {
                    detected_markers[input_file] = corners;
                }
            }
            catch (const std::exception &e) {
                clog::L("main", 1) << "Reading file " << input_file << " failed with an exception: " << std::endl
                                   << e.what() << std::endl;
            }
        }
        TIMELOG("Reading markers");
        clog::L(__func__, 2) << "Adding images to the calibration..." << std::endl;
        std::cout << std::string(detected_markers.size(), '-') << std::endl;
        for (auto const& it : detected_markers) {
            calib.addInputImage(it.first, it.second);
            std::cout << "." << std::flush;
        }
        std::cout << std::endl;
        TIMELOG("Adding input images");
        if (!cache_file.empty()) {
            calib.save(cache_file);
            TIMELOG("Writing cache file");
        }
    }


    calib.purgeInvalidPages();
    TIMELOG("Purging invalid pages");

    calib.purgeUnlikelyByDetectedRectangles();
    TIMELOG("purgeUnlikelyByDetectedRectangles");

    for (std::string const& calibration_type : calibration_types) {
        clog::L(__func__, 2) << "Running calib " << calibration_type << std::endl;
        calib.runCalib(calibration_type, outlier_threshold);
        calib_updated = true;
        TIMELOG(std::string("Calib ") + calibration_type);
        if (!cache_file.empty()) {
            calib.save(cache_file);
            TIMELOG("Writing cache file");
        }
        calib.exportPointClouds(calibration_type);
        TIMELOG("exportPointClouds");

        clog::L(__func__, 2) << calib.printAllCameraMatrices() << std::endl;
    }


    for (std::string const& calibration_type : calibration_types_eval) {
        calib.plotReprojectionErrors(calibration_type, calibration_type);
        TIMELOG(std::string("plotReprojectionErrors for ") + calibration_type);
    }

    if (!same_pos_suffixes.empty()) {
        clog::L(__func__, 2) << "Checking same positions 2D";
        calib.checkSamePosition2D(same_pos_suffixes);
        TIMELOG("checkSamePosition2D");
        clog::L(__func__, 2) << "Checking same positions 3D";
        calib.checkSamePosition(same_pos_suffixes);
        TIMELOG("checkSamePosition");
    }

    if (calib_updated && !cache_file.empty()) {
        calib.save(cache_file);
        TIMELOG("Writing cache file");
    }


    std::cout << "Level 1 log entries: " << std::endl;
    clog::Logger::getInstance().printAll(std::cout, 1);
    TIMELOG("print all log entries");

    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
