#include <iostream>
#include <vector>

#include <tclap/CmdLine.h>
#include <ParallelTime/paralleltime.h>
#include <boost/filesystem.hpp>

#include "hdcalib.h"
#include "cornercolor.h"

#include "gnuplot-iostream.h"

#undef NDEBUG
#include <cassert>

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

struct Rates {
    size_t num_corners;
    std::vector<double> rates_by_color;
    double rate;
    std::string filename;
    double mean_dist;

    double mean_main_edge;

    double density;

    /**
     * @brief write Function needed for serializating a Corner using the OpenCV FileStorage system.
     * @param fs
     */
    void write(cv::FileStorage& fs) const {
        fs << "{"
           << "num_corners" << int(num_corners)
           << "rates_by_color" << rates_by_color
           << "rate" << rate
           << "filename" << filename
           << "mean_dist" << mean_dist
           << "mean_main_edge" << mean_main_edge
           << "}";
    }

    /**
     * @brief read Method needed for reading a serialized Corner using the OpenCV FileStorage system.
     * @param node
     */
    void read(const cv::FileNode& node) {
        int _num_corners = -1;
        node["num_corners"] >> _num_corners;
        if (_num_corners < 0) {
            throw std::runtime_error("Reading num_corners failed");
        }
        num_corners = _num_corners;
        node["rates_by_color"] >> rates_by_color;
        node["rate"] >> rate;
        node["filename"] >> filename;
        node["mean_dist"] >> mean_dist;
        node["mean_main_edge"] >> mean_main_edge;
        if (rates_by_color.size() != 2) {
            throw std::runtime_error(std::string("rates_by_color has unexpected size ") + std::to_string(rates_by_color.size()) + ", expected 2.");
        }
    }

    std::string getString() const {
        return std::to_string(mean_dist) + "\t"
                + std::to_string(rate) + "\t"
                + std::to_string(rates_by_color[0]) + "\t"
                + std::to_string(rates_by_color[1]) + "\t"
                + std::to_string(num_corners);
    }

    Rates() {
        rates_by_color.resize(2);
    }
};

void write(cv::FileStorage& fs, const std::string&, const Rates& x){
    x.write(fs);
}
void read(const cv::FileNode& node, Rates& x, const Rates& default_value = Rates()){
    if(node.empty()) {
        throw std::runtime_error("Could not recover rates cache, file storage is empty.");
    }
    else
        x.read(node);
}

bool isInSquare(hdmarker::Corner const& c, std::vector<hdmarker::Corner> const& square) {
    assert(square.size() == 4);
    int min_x = std::min(std::min(square[0].id.x, square[1].id.x), std::min(square[2].id.x, square[3].id.x));
    int min_y = std::min(std::min(square[0].id.y, square[1].id.y), std::min(square[2].id.y, square[3].id.y));

    int max_x = std::max(std::max(square[0].id.x, square[1].id.x), std::max(square[2].id.x, square[3].id.x));
    int max_y = std::max(std::max(square[0].id.y, square[1].id.y), std::max(square[2].id.y, square[3].id.y));

    return c.id.x < max_x && c.id.y < max_y && c.id.x > min_x && c.id.y > min_y;
}

double polynomeArea(std::vector<hdmarker::Corner> const& vec) {
    double result = 0;
    int const n = vec.size();
    for (int ii = 0; ii < n; ++ii) {
        result += vec[ii].p.x*(vec.at((ii+1)%n).p.y - vec.at((ii+n-1)%n).p.y);
    }
    return std::abs(result/2);
}

Rates analyzeRates(fs::path const& file, int8_t const recursion, int const page) {
    Rates result;
    fs::path const rates_cache_file = file.string() + "-rates-r" + std::to_string(recursion) + "-p" + std::to_string(page) + ".yaml";
    if (false) {
        if (fs::is_regular_file(rates_cache_file)) {
            try {
                cv::FileStorage cache(rates_cache_file.string(), cv::FileStorage::READ);
                cache["cache"] >> result;
                return result;
            } catch (std::exception const& e) {
                std::cout << "Reading file " << rates_cache_file .string() << " failed with exception:" << std::endl
                          << e.what() << std::endl;
            }
        }
    }
    std::vector<hdmarker::Corner> corners;
    hdmarker::Corner::readGzipFile(file.string(), corners);
    corners = hdcalib::Calib::purgeInvalidPages(corners, {page});
    std::cout << "File " << file.string() << " has " << corners.size() << " corners." << std::endl;

    hdcalib::CornerStore store(corners);
    int const factor = store.getCornerIdFactorFromMainMarkers();

    runningstats::RunningStats distances;
    runningstats::BinaryStats rate;
    runningstats::BinaryStats rates_by_color[2];

    for (auto const& squares : store.getSquares(factor)) {
        if (squares.size() != 4) {
            continue;
        }
        hdmarker::Corner const& c = squares[0];
        for (size_t ii = 0; ii < 4; ++ii) {
            distances.push_unsafe(hdcalib::Calib::distance(squares[ii], squares[(ii + 1) % 4]) / (factor/2));
        }
        for (int xx = 1; xx < factor; xx += 2) {
            for (int yy = 1; yy < factor; yy += 2) {
                hdmarker::Corner search = c;
                search.id += cv::Point2i(xx, yy);
                size_t color = CornerColor::getColor(search, recursion);
                if (color > 1) {
                    throw std::runtime_error("Color value unexpectedly high, something is wrong.");
                }
                if (store.hasIDLevel(search, recursion)) {
                    rate.push(true);
                    rates_by_color[color].push(true);
                }
                else {
                    rate.push(false);
                    rates_by_color[color].push(false);
                }
            }
        }
    }

    std::cout << "distances: " << distances.print() << std::endl;
    std::cout << "rate: " << rate.getPercent() << "%" << std::endl;
    std::cout << "black rate: " << rates_by_color[0].getPercent() << "%" << std::endl;
    std::cout << "white rate: " << rates_by_color[1].getPercent() << "%" << std::endl;

    std::cout << std::endl;

    result.rate = rate.getPercent();
    result.filename = file.string();
    result.mean_dist = distances.getMean();
    result.num_corners = corners.size();
    result.rates_by_color.resize(2);
    result.rates_by_color[0] = rates_by_color[0].getPercent();
    result.rates_by_color[1] = rates_by_color[1].getPercent();

    cv::FileStorage cache(rates_cache_file.string(), cv::FileStorage::WRITE);
    cache << "cache" << result;
    return result;
}

double getDensity(fs::path const& file, Rates& rates, int const page) {
    std::vector<hdmarker::Corner> corners;
    hdmarker::Corner::readGzipFile(file.string(), corners);
    corners = hdcalib::Calib::purgeInvalidPages(corners, {page});
    std::cout << "File " << file.string() << " has " << corners.size() << " corners." << std::endl;

    hdcalib::CornerStore store(corners);
    int const factor = store.getCornerIdFactorFromMainMarkers();

    runningstats::RunningStats distances, areas;
    runningstats::BinaryStats rate;
    runningstats::BinaryStats rates_by_color[2];

    std::vector<std::vector<hdmarker::Corner> > squares = store.getSquares(factor);
    double total_pixels = 0;
    for (auto const& square : squares) {
        assert(square.size() == 4);
        for (size_t ii = 0; ii < 4; ++ii) {
            distances.push_unsafe(hdcalib::Calib::distance(square[ii], square[(ii + 1) % 4]));
        }
        double const area = polynomeArea(square);
        areas.push_unsafe(area);
        total_pixels += area;
    }
    size_t in_square_count = 0;
    for (hdmarker::Corner const& c : corners) {
        size_t local_in_square_count = 0;
        for (auto const& square: squares) {
            if (isInSquare(c, square)) {
                local_in_square_count++;
            }
        }
        if (local_in_square_count > 1) {
            std::cout << "Warning: corner " << c.id << " found in multiple squares?" << std::endl;
        }
        if (local_in_square_count > 0) {
            in_square_count++;
        }
    }

    double const mean_dist = distances.getMean();

    std::cout << "Mean dist: " << mean_dist << ", square: " << mean_dist * mean_dist << std::endl;
    std::cout << "Mean area: " << areas.getMean() << std::endl;

    rates.mean_main_edge = mean_dist;
    double const megapixels = double(total_pixels) / 1'000'000;
    rates.density = double(in_square_count)/megapixels;

    return 0;
}

void plotDensity(std::string prefix, std::map<double, Rates> & rates) {
    prefix += "-density";
    std::string logfile = prefix + ".data";
    std::ofstream log(logfile);
    log << "# 1. mean main marker square edge length, 2. density in markers/megapixels" << std::endl;
    std::map<double, double> density_by_main_edge;
    for (auto const& it : rates) {
        if (it.second.density > 0) {
            density_by_main_edge[it.second.mean_main_edge] = it.second.density;
        }
    }
    for (auto const& it : density_by_main_edge) {
        log << it.first << "\t" << it.second << std::endl;
    }
    for (auto const& it : rates) {
        std::cout << it.second.filename << " has density " << it.second.density << ", main edge: " << it.second.mean_main_edge << std::endl;
    }
    gnuplotio::Gnuplot plt("tee " + prefix + ".gpl | gnuplot -persist");
    plt << "set term svg enhanced background rgb \"white\";\n"
        << "set output \"" << prefix << ".svg\";\n"
        << "set title 'Marker detection density';\n"
        //<< "set xrange [4.5:10];\n"
        //<< "set yrange [0:100];\n"
        << "set logscale x;\n"
        << "set xlabel 'main marker edge [px]';\n"
        << "set ylabel 'detection density[1/MP]';\n"
        << "plot '" << logfile << "' u 1:2 w lp title 'density'";
}

void analyzeDirectory(std::string const& dir, int const recursion, int const page) {
    std::map<double, std::string> data;
    std::vector<fs::path> files;
    for (fs::recursive_directory_iterator itr(dir); itr!=fs::recursive_directory_iterator(); ++itr) {
        if (fs::is_regular_file(itr->status())
                && stringEndsWith(itr->path().filename().string(), "-submarkers.hdmarker.gz")) {
            files.push_back(itr->path());
        }
    }
    std::sort(files.begin(), files.end());

    std::cout << "Directory " << dir << " has " << files.size() << " relevant files" << std::endl;

    std::map<double, std::string> overview;

    std::map<double, Rates> rates;

#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < files.size(); ++ii) {
        fs::path const& p = files[ii];
        Rates result = analyzeRates(p, recursion, page);
        getDensity(p, result, page);
#pragma omp critical
        {
            if (true || (result.num_corners > 0 && result.rate > 0)) {
                rates[result.mean_dist] = result;
            }
        }
    }
    if (rates.empty()) {
        return;
    }
    std::string prefix = dir + std::to_string(page);
    std::ofstream logfile(prefix + "-log");
    Rates previous = rates.begin()->second;
    for (auto const& it : rates) {
        if (it.first > 0) {
            logfile << it.second.getString() << "\t" << it.second.filename << std::endl;
            overview[it.first] = it.second.getString() + "\t" + it.second.filename;
            previous = it.second;
        }
    }
    plotDensity(prefix, rates);
    gnuplotio::Gnuplot plt;
    std::stringstream cmd;

    std::cout << "Overview for directory " << dir << ":" << std::endl;
    for (auto const& it : overview) {
        std::cout << it.first << "\t" << it.second << std::endl;
    }

    cmd << "set term svg enhanced background rgb \"white\";\n"
        << "set output \"" << prefix << "-log.svg\";\n"
        << "set title 'Marker detection rates';\n"
        << "set xrange [4.5:10];\n"
        << "set yrange [0:100];\n"
        << "set xlabel 'submarker distance [px]';\n"
        << "set ylabel 'detection rate [%]';\n"
        << "plot '" << prefix << "-log' u 1:2 w lp title 'combined rate',"
        << "'' u 1:3 w lp title 'black rate',"
        << "'' u 1:4 w lp title 'white rate'\n";
    plt << cmd.str();
    std::ofstream gpl_file(prefix + "-log.gpl");
    gpl_file << cmd.str();
}

int main(int argc, char ** argv) {

    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    std::vector<std::string> input_dirs;
    int recursion_depth = -1;
    std::vector<int> pages;

    try {
        TCLAP::CmdLine cmd("hdcalib tool for analyzing submarker detection rates", ' ', "0.1");

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 false, -1, "int");
        cmd.add(recursive_depth_arg);

        TCLAP::MultiArg<int> pages_arg("p", "pages",
                                       "page number (s) of the calibration target.",
                                       true, "int");
        cmd.add(pages_arg);

        TCLAP::MultiArg<std::string> directories_arg("i",
                                                     "input",
                                                     "Directory containing extracted corners.",
                                                     false,
                                                     "directory");
        cmd.add(directories_arg);


        cmd.parse(argc, argv);

        pages = pages_arg.getValue();

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
                  << "Number of input directories: " << input_dirs.size() << std::endl
                  << "recursion depth: " << recursion_depth << std::endl;
    }
    catch (std::exception const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    for (int const page : pages) {
        for (auto const& dir : input_dirs) {
            analyzeDirectory(dir, recursion_depth, page);
        }
    }
    for (auto const& dir : input_dirs) {
        { // Plot detection probabilities
            gnuplotio::Gnuplot plt;
            std::stringstream cmd;
            cmd << "set term svg enhanced background rgb 'white';\n"
            << "set output 'all-" << dir << ".svg';\n"
            << "set title 'Marker detection rates';\n"
            << "set xrange [4:10];\n"
            << "set yrange [0:100];\n"
            << "set xlabel 'submarker distance [px]';\n"
            << "set ylabel 'detection rate [%]';\n"
            << "set key out horiz;\n"
            << "plot '" << dir << pages[0] << "-log' u 1:2 w lp title '" << pages[0] << "'";
            for (size_t ii = 1; ii < pages.size(); ++ii) {
                cmd << ", '" << dir << pages[ii] << "-log' u 1:2 w lp title '" << pages[ii] << "'";
            }
            plt << cmd.str();
            std::ofstream gpl_file(std::string("all-") + dir + "-log.gpl");
            gpl_file << cmd.str();
        }
        { // plot detection rates
            gnuplotio::Gnuplot plt;
            std::stringstream cmd;
            cmd << "set term svg enhanced background rgb 'white';\n"
            << "set output 'all-" << dir << "-density.svg';\n"
            << "set title 'Marker detection densities';\n"
            << "set xlabel 'main marker edge [px]';\n"
            << "set ylabel 'detection density [1/MP]';\n"
            << "set logscale x;\n"
            << "set key out horiz;\n"
            << "plot '" << dir << pages[0] << "-density.data' u 1:2 w l title '" << pages[0] << "'";
            for (size_t ii = 1; ii < pages.size(); ++ii) {
                cmd << ", '" << dir << pages[ii] << "-density.data' u 1:2 w l title '" << pages[ii] << "'";
            }
            plt << cmd.str();
            std::ofstream gpl_file(std::string("all-") + dir + "-density-log.gpl");
            gpl_file << cmd.str();
        }
    }

    std::cout << "Total time: " << total_time.print() << std::endl;

    return EXIT_SUCCESS;
}
