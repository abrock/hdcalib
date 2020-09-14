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
        int const recursion,
        runningstats::Stats2D<float> &local_stat_2d,
        runningstats::Stats2D<float> local_stat_2d_color[3][4]
) {
    std::set<std::string> suffixes;
    for (size_t ii = 0; ii < a.size(); ++ii) {
        hdmarker::Corner const& _a = a.get(ii);
        hdmarker::Corner _b;
        if (b.hasIDLevel(_a, _b, _a.level)) {
            size_t const color = CornerColor::getColor(_a, recursion);
            if (color > 3) {
                throw std::runtime_error("Color value unexpectedly >3, aborting.");
            }
            std::pair<double, double> const val(_b.p.x - _a.p.x, _b.p.y - _a.p.y);
            local_stat_2d.push_unsafe(val);
            if (_a.level == 0) { // main marker
                if (color < 2) {
                    throw std::runtime_error("Error: Main marker with color < 2");
                }
                if (color > 4) {
                    throw std::runtime_error("Error: Main marker with color > 4");
                }
            }
            else {
                if (color > 1) {
                    throw std::runtime_error("Error: Submarker with color > 1");
                }
            }
            size_t const color_index = (_a.level == 0 ? color-2 : color);
            local_stat_2d_color[2][_a.level].push_unsafe(val);
            local_stat_2d_color[color_index][_a.level].push_unsafe(val);
        }
    }
    runningstats::HistConfig conf2d;
    conf2d.setMinMaxX(-2,2)
            .setMinMaxY(-2,2)
            .setXLabel("x offset [px]")
            .setYLabel("y offset [px]")
            .addVerticalLine(0, "white")
            .addHorizontalLine(0, "white");

    runningstats::HistConfig conf1d;
    conf1d.setMinMaxX(-2,2)
            .setMinMaxY(-2,2)
            .addVerticalLine(0, "black");

    std::string suffix;
    for (size_t color_index = 0; color_index < 3; ++color_index) {
        for (size_t level = 0; level < 4; ++level) {
            runningstats::Stats2D<float> & stats = local_stat_2d_color[color_index][level];
            if (stats.empty()) {
                continue;
            }
            size_t const color = (level == 0 ? color_index + 2 : color_index);
            suffix = "-offset2d-c" + std::to_string(color) + "-l" + std::to_string(level);
            suffixes.insert(suffix);
            stats.plotHist(result_prefix + suffix,
                           stats.FreedmanDiaconisBinSize(),
                           conf2d);
            suffix = "-offset2d-log-c" + std::to_string(color) + "-l" + std::to_string(level);
            suffixes.insert(suffix);
            stats.plotHist(result_prefix + suffix,
                           stats.FreedmanDiaconisBinSize(),
                           conf2d.clone()
                           .setLogCB());
            for (std::string const axis : {"x", "y"}) {
                runningstats::QuantileStats<float> axis_stats = stats.get(axis);
                suffix = "-offset-" + axis + "-c" + std::to_string(color) + "-l" + std::to_string(level);
                suffixes.insert(suffix);
                axis_stats.plotHistAndCDF(result_prefix + suffix,
                                          axis_stats.FreedmanDiaconisBinSize(),
                                          conf1d.clone()
                                          .setDataLabel(axis + " offset [px]"));
            }
        }
    }

    suffix = "-offset2d";
    suffixes.insert(suffix);
    local_stat_2d.plotHist(result_prefix + suffix,
                     local_stat_2d.FreedmanDiaconisBinSize(),
                     conf2d);
    suffix = "-offset2d-log";
    suffixes.insert(suffix);
    local_stat_2d.plotHist(result_prefix + suffix,
                     local_stat_2d.FreedmanDiaconisBinSize(),
                     conf2d.clone()
                     .setLogCB());

#pragma omp single
    for (std::string const& suffix : suffixes) {
        std::string const dirname = "suffix-" + suffix;
        std::cout << "mkdir \"" << dirname << "\"" << std::endl;
        std::cout << "for i in *" << suffix << "*; do ln -s $(pwd)/$i \"" << dirname << "\"/$i; done " << std::endl;
    }
}

void analyzeFileList(std::vector<std::string> const& files, int const recursion) {
    if (files.size() < 2) {
        std::cout << "Number of files in given directory is smaller than two." << std::endl;
        return;
    }
    hdcalib::CornerStore first_corners(hdcalib::Calib::readCorners(files.front()));
    runningstats::Stats2D<float> stat_2d, stat_2d_color[3][4];
#pragma omp parallel for
    for (size_t ii = 1; ii < files.size(); ++ii) {
        runningstats::Stats2D<float> local_stat_2d, local_stat_2d_color[3][4];
        hdcalib::CornerStore cmp_corners(hdcalib::Calib::readCorners(files[ii]));
        analyzeOffsets(first_corners, cmp_corners, files[ii], recursion, local_stat_2d, local_stat_2d_color);
    }
}

int main(int argc, char ** argv) {

    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    std::vector<std::string> input_dirs;

    std::vector<std::vector<std::string> > files;

    int recursion = 0;

    try {
        TCLAP::CmdLine cmd("hdcalib tool for analyzing offsets of detected markers between images", ' ', "0.1");

        TCLAP::MultiArg<std::string> files_arg("f",
                                                     "file",
                                                     "Text file containing images. The first image is the reference and is compared to all other images.",
                                                     false,
                                                     "directory");
        cmd.add(files_arg);

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 true, -1, "int");
        cmd.add(recursive_depth_arg);


        cmd.parse(argc, argv);

        recursion = recursive_depth_arg.getValue();

        if (input_dirs.empty()) {
            std::cerr << "Fatal error: No input directories specified." << std::endl;
            cmd.getOutput()->usage(cmd);

            return EXIT_FAILURE;
        }

        for (std::string const& src : files_arg.getValue()) {
            std::ifstream in(src);
            std::vector<std::string> local_files;
            std::string name;
            while (in >> name) {
                trim(name);
                if (!name.empty() && fs::is_regular_file(name)) {
                    local_files.push_back(name);
                }
            }
            if (local_files.size() > 1) {
                files.push_back(local_files);
            }
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

    for (auto const& dir : files) {
        analyzeFileList(dir, recursion);
    }


    std::cout << "Total time: " << total_time.print() << std::endl;

    return EXIT_SUCCESS;
}
