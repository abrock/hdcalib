#include <iostream>
#include <vector>

#include <tclap/CmdLine.h>
#include <ParallelTime/paralleltime.h>
#include <boost/filesystem.hpp>

#include "hdcalib.h"
#include "cornercolor.h"

#include "gnuplot-iostream.h"

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

bool stringEndsWith(std::string const& str, std::string const& search) {
    if (search.size() > str.size()) {
        return false;
    }
    return str.substr(str.size() - search.size()) == search;
}

void analyzeOffsets(
        hdcalib::CornerStore const& a,
        hdcalib::CornerStore const& b,
        std::string const& result_prefix,
        int const recursion) {
    runningstats::QuantileStats<float> stat_x, stat_y;
    runningstats::QuantileStats<float> stat_x_color[4], stat_y_color[4];
    for (size_t ii = 0; ii < a.size(); ++ii) {
        hdmarker::Corner const& _a = a.get(ii);
        hdmarker::Corner _b;
        if (b.hasID(_a, _b)) {
            size_t const color = CornerColor::getColor(_a, recursion);
            if (color > 3) {
                throw std::runtime_error("Color value unexpectedly >3, aborting.");
            }
            stat_x.push_unsafe(_b.p.x - _a.p.x);
            stat_x_color[color].push_unsafe(_b.p.x - _a.p.x);
            stat_y.push_unsafe(_b.p.y - _a.p.y);
            stat_y_color[color].push_unsafe(_b.p.y - _a.p.y);
        }
    }
    stat_x.plotHistAndCDF(result_prefix + "-offset-x", 0.1);
    stat_y.plotHistAndCDF(result_prefix + "-offset-y", 0.1);
    for (size_t ii = 0; ii < 4; ++ii) {
        stat_x_color[ii].plotHistAndCDF(result_prefix + "-offset-x-color-" + std::to_string(ii), 0.1);
        stat_y_color[ii].plotHistAndCDF(result_prefix + "-offset-x-color-" + std::to_string(ii), 0.1);
    }
}

void analyzeDirectory(std::string const& dir, int const recursion) {
    std::map<double, std::string> data;
    std::vector<fs::path> files;
    for (fs::directory_iterator itr(dir); itr!=fs::directory_iterator(); ++itr) {
        if (fs::is_regular_file(itr->status())
                && stringEndsWith(itr->path().filename().string(), "-submarkers.yaml.gz")) {
            files.push_back(itr->path());
        }
    }
    if (files.size() < 2) {
        std::cout << "Number of files in given directory is smaller than two." << std::endl;
        return;
    }
    std::sort(files.begin(), files.end());
    hdcalib::CornerStore first_corners(hdcalib::Calib::readCorners(files.front().string()));
#pragma omp parallel for
    for (size_t ii = 1; ii < files.size(); ++ii) {
        hdcalib::CornerStore cmp_corners(hdcalib::Calib::readCorners(files[ii].string()));
        analyzeOffsets(first_corners, cmp_corners, files[ii].string(), recursion);
    }

    return;

    std::ofstream logfile(dir + "-log");
    for (auto const& it : data) {
        if (it.first > 0) {
            logfile << it.second << std::endl;
        }
    }
    gnuplotio::Gnuplot plt;
    std::stringstream cmd;

    cmd << "set term svg enhanced background rgb \"white\";\n"
        << "set output \"" << dir << "-log.svg\";\n"
        << "set title 'Marker detection rates';\n"
        << "set xrange [6:15];\n"
        << "set yrange [0:100];\n"
        << "set xlabel 'submarker distance [px]';\n"
        << "set ylabel 'detection rate [%]';\n"
        << "plot '" << dir << "-log' u 1:2 w lp title 'combined rate',"
           << "'' u 1:3 w lp title 'black rate',"
           << "'' u 1:4 w lp title 'white rate'\n";
    plt << cmd.str();
    std::ofstream gpl_file(dir + "-log.gpl");
    gpl_file << cmd.str();
}

int main(int argc, char ** argv) {

    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    std::vector<std::string> input_dirs;

    int recursion = 0;

    try {
        TCLAP::CmdLine cmd("hdcalib tool for analyzing offsets of detected markers between images", ' ', "0.1");

        TCLAP::MultiArg<std::string> directories_arg("i",
                                                     "input",
                                                     "Directory containing extracted corners.",
                                                     false,
                                                     "directory");
        cmd.add(directories_arg);

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 true, -1, "int");
        cmd.add(recursive_depth_arg);


        cmd.parse(argc, argv);

        recursion = recursive_depth_arg.getValue();
        input_dirs = directories_arg.getValue();

        if (input_dirs.empty()) {
            std::cerr << "Fatal error: No input directories specified." << std::endl;
            cmd.getOutput()->usage(cmd);

            return EXIT_FAILURE;
        }

        std::cout << "Parameters: " << std::endl
                  << "Number of input directories: " << input_dirs.size() << std::endl;
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    for (auto const& dir : input_dirs) {
        analyzeDirectory(dir, recursion);
    }


    std::cout << "Total time: " << total_time.print() << std::endl;

    return EXIT_SUCCESS;
}
