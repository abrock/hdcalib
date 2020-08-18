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
    std::ofstream logfile("hdcalib.log", std::ofstream::out);

    clog::Logger::getInstance().addListener(std::cout);
    clog::Logger::getInstance().addListener(logfile);

    ParallelTime t, total_time;
    std::stringstream time_log;

    bool calib_updated = false;

    hdcalib::Calib calib;
    std::vector<std::string> calibration_types;
    std::string cache_file;
    std::string grids_file;
    std::vector<hdcalib::GridDescription> descriptions;
    try {
        TCLAP::CmdLine cmd("hdcalib calibration tool", ' ', "0.1");

        TCLAP::ValueArg<std::string> cache_arg("c", "cache",
                                               "Cache filename the calibration results to use.",
                                               false, "", "Calibration cache.");
        cmd.add(cache_arg);

        TCLAP::ValueArg<std::string> grids_arg("g", "grids",
                                               "Filename of the grid decriptions file.",
                                               false, "", "Grid descriptions file.");
        cmd.add(grids_arg);

        TCLAP::MultiArg<std::string> type_arg("t", "type",
                                              "Type of the calibration(s) to run. "
                                              "Possibilities in increasing order of computational complexity:"
                                              "SimpleOpenCV, SimpleCeres, OpenCV, Ceres, Flexible, SemiFlexible ",
                                              false, "Calibration type.");
        cmd.add(type_arg);

        cmd.parse(argc, argv);


        cache_file = cache_arg.getValue();
        calibration_types = type_arg.getValue();
        grids_file = grids_arg.getValue();
        if (!grids_arg.isSet() || grids_file.empty()) {
            grids_file = "grids-example.yaml";
            hdcalib::GridDescription d;
            d.name = "forward";
            d.points.push_back({"/0/00.tif", {0,0,0}});
            d.points.push_back({"/1/00.tif", {0,0,10}});
            descriptions.push_back(d);
            d.name = "lateral";
            d.points.clear();
            d.points.push_back({"/5/00.tif", {0,0,0}});
            d.points.push_back({"/6/00.tif", {10,0,0}});
            descriptions.push_back(d);

            hdcalib::GridDescription::writeFile(grids_file, descriptions);

            std::cout << "Please specify a grid descriptions file, an example has been written to " << grids_file << std::endl;
            return 0;
        }
        hdcalib::GridDescription::readFile(grids_file, descriptions);

        clog::L("tclap", 2) << "Parameters: " << std::endl;
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
        clog::L("main", 2) << "Reading cached calibration results from file " << cache_file << std::endl;
        cv::FileStorage fs(cache_file, cv::FileStorage::READ);
        cv::FileNode n = fs["calibration"];
        n >> calib;
        has_cached_calib = true;
        fs.release();
        clog::L(__func__, 2) << calib.printAllCameraMatrices() << std::endl;
        TIMELOG("Reading cached result");
    }

    calib.purgeUnlikelyByDetectedRectangles();
    TIMELOG("purgeUnlikelyByDetectedRectangles");

    for (std::string const& calibration_type : calibration_types) {
        clog::L(__func__, 2) << "Running fitGrid on calib " << calibration_type << std::endl;

        hdcalib::FitGrid fit;
        fit.runFit(calib, calib.getCalib(calibration_type), descriptions);

        TIMELOG(std::string("Calib ") + calibration_type);
    }

    return EXIT_SUCCESS;
}
