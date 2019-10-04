#include <iostream>
#include <vector>

#include <tclap/CmdLine.h>
#include <ParallelTime/paralleltime.h>
#include <boost/filesystem.hpp>

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

bool stringEndsWith(std::string const& str, std::string const& search) {
    if (search.size() > str.size()) {
        return false;
    }
    return str.substr(str.size() - search.size()) == search;
}

std::pair<double, std::string> analyzeRates(fs::path const& file, int const recursion) {
    std::pair<double, std::string> result;
    std::vector<hdmarker::Corner> corners = hdcalib::Calib::readCorners(file.string());
    std::cout << "File " << file.string() << " has " << corners.size() << " corners." << std::endl;
    int const factor = hdcalib::Calib::computeCornerIdFactor(recursion);

    hdcalib::CornerStore store(corners);

    runningstats::RunningStats distances;
    runningstats::BinaryStats rate;

    for (auto const& c : corners) {
        if (c.id.x % factor == 0 && c.id.y % factor == 0) {
            bool has_square = true;
            hdmarker::Corner neighbour;
            for (cv::Point2i const & id_offset : {
                 cv::Point2i(0, factor),
                 cv::Point2i(factor, 0),
                 cv::Point2i(factor, factor)}) {
                hdmarker::Corner search = c;
                search.id += id_offset;
                if (store.hasID(search, neighbour)) {
                    cv::Point2f const residual = c.p - neighbour.p;
                    double const dist = double(std::sqrt(residual.dot(residual)));
                    distances.push_unsafe(dist/std::sqrt(id_offset.dot(id_offset)));
                }
                else {
                    has_square = false;
                }
            }
            if (has_square) {
                for (int xx = 1; xx < factor; xx += 2) {
                    for (int yy = 1; yy < factor; yy += 2) {
                        hdmarker::Corner search = c;
                        search.id += cv::Point2i(xx, yy);
                        if (store.hasID(search)) {
                            rate.push(true);
                        }
                        else {
                            rate.push(false);
                        }
                    }
                }
            }
        }
    }

    std::cout << "distances: " << distances.print() << std::endl;
    std::cout << "rate: " << rate.getPercent() << "%" << std::endl;

    std::cout << std::endl;
    return result;
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
    std::sort(files.begin(), files.end());
    for (auto const& p : files) {
        analyzeRates(p, recursion);
    }
}

int main(int argc, char ** argv) {

    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    std::vector<std::string> input_dirs;
    int recursion_depth = -1;

    try {
        TCLAP::CmdLine cmd("hdcalib tool for analyzing submarker detection rates", ' ', "0.1");

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 false, -1, "int");
        cmd.add(recursive_depth_arg);

        TCLAP::MultiArg<std::string> directories_arg("i",
                                                     "input",
                                                     "Directory containing extracted corners.",
                                                     false,
                                                     "directory");
        cmd.add(directories_arg);


        cmd.parse(argc, argv);


        recursion_depth = recursive_depth_arg.getValue();
        if (recursion_depth < 1) {
            throw std::runtime_error("Submarker detection rates can not be analyzed when the recursion depth is smaller than 1.");
        }
        input_dirs = directories_arg.getValue();

        if (input_dirs.empty()) {
            std::cerr << "Fatal error: No input directories specified." << std::endl;
            cmd.getOutput()->usage(cmd);

            return EXIT_FAILURE;
        }

        std::cout << "Parameters: " << std::endl
                  << "Number of input files: " << input_dirs.size() << std::endl
                  << "recursion depth: " << recursion_depth << std::endl;
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
        analyzeDirectory(dir, recursion_depth);
    }


    return EXIT_SUCCESS;
}
