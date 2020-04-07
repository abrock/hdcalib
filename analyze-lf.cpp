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
    clog::Logger::getInstance().addListener(std::cout);

    hdcalib::Calib calib;
    std::vector<std::string> input_files;
    size_t rows = 0;
    size_t cols = 0;
    bool demosaic = false;
    bool libraw = false;
    bool only_green = false;
    std::string cache_file;
    TCLAP::CmdLine cmd("hdcalib analyzation tool for captured light fields", ' ', "0.1");
    try {

        TCLAP::ValueArg<size_t> rows_arg("r", "rows",
                                         "Number of rows in the light field. Usually 9.",
                                         false, 9, "Number of rows in the light field.");
        cmd.add(rows_arg);

        TCLAP::ValueArg<size_t> cols_arg("c", "cols",
                                         "Number of columns in the light field. Usually 9.",
                                         false, 9, "Number of columns in the light field.");
        cmd.add(cols_arg);

        TCLAP::ValueArg<std::string> cache_arg("", "calib",
                                               "Cache file for the calibration results. This makes use of the opencv filestorage capabilities so filename extension should be .xml/.xml.gz/.yaml/.yaml.gz",
                                               true, "", "Calibration cache.");
        cmd.add(cache_arg);

        TCLAP::SwitchArg demosaic_arg("d", "demosaic",
                                      "Use this flag if the input images are raw images and demosaicing should be used.",
                                      false);
        cmd.add(demosaic_arg);

        TCLAP::SwitchArg read_raw_arg("", "raw",
                                      "Use this flag if the input images are raw images which must be read using LibRaw since OpenCV cannot read them. This implies -d.",
                                      false);
        cmd.add(read_raw_arg);

        TCLAP::SwitchArg only_green_arg("g", "only-green", "Set this flag true if only the green channel of a bayer image should be used."
                                                           "In the case of demosaicing this means that the missing green pixels"
                                                           "are interpolated bilinear.", false);
        cmd.add(only_green_arg);

        TCLAP::ValueArg<std::string> textfile_arg("i",
                                                  "input",
                                                  "Text file containing a list of image paths relative to the working directory. Note that the the files must be in row-major order.",
                                                  true,
                                                  "", "Text file with a list of input images. Row-major order.");
        cmd.add(textfile_arg);


        cmd.parse(argc, argv);

        rows = rows_arg.getValue();
        cols = cols_arg.getValue();
        libraw = read_raw_arg.getValue();
        only_green = only_green_arg.getValue();
        demosaic = demosaic_arg.getValue() || libraw;
        std::string const textfile = textfile_arg.getValue();
        cache_file = cache_arg.getValue();
        if (!fs::is_regular_file(cache_file)) {
            throw std::runtime_error(("Specified cache file (") + cache_file + ") is not a regular file.");
        }

        if (!fs::is_regular_file(textfile)) {
            throw std::runtime_error("Specified file containing the input images is not a regular file.");
        }
        std::ifstream in(textfile);
        std::string line;
        while (std::getline(in, line)) {
            trim(line);
            if (fs::is_regular_file(line)) {
                input_files.push_back(line);
            }
            else {
                throw std::runtime_error(std::string("Specified lightfield image (") + line + ") is not a regular file.");
            }
        }

        if (input_files.empty()) {
            throw std::runtime_error("Fatal error: No input files specified.");
        }

        if (rows < 1) {
            throw std::runtime_error("Number of rows must be positive.");
        }
        if (cols < 1) {
            throw std::runtime_error("Number of columns must be positive.");
        }
        if (rows * cols < 2) {
            throw std::runtime_error("Need at least two images for a light field.");
        }

        if (input_files.size() != rows * cols) {
            throw std::runtime_error(std::string("Number of input images (") + std::to_string(input_files.size()) + ") doesn't match number of images in the specified light field configuration (" + std::to_string(rows) + " * " + std::to_string(cols) + " = " + std::to_string(rows * cols));
        }

        std::cout << "Parameters: " << std::endl
                  << "Number of input files: " << input_files.size() << std::endl
                  << "demosaic: " << (demosaic ? "true" : "false") << std::endl
                  << "use libraw: " << (libraw ? "true" : "false") << std::endl
                  << "only green channel: " << (only_green ? "true" : "false") << std::endl
                  << "rows: " << rows << std::endl
                  << "columns: " << cols << std::endl;

        calib.only_green(only_green);
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (std::exception const & e) {
        std::cerr << e.what() << std::endl;
        cmd.getOutput()->usage(cmd);
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return EXIT_FAILURE;
    }

    cv::FileStorage fs(cache_file, cv::FileStorage::READ);
    cv::FileNode n = fs["calibration"];
    n >> calib;
    fs.release();

    calib.printObjectPointCorrectionsStats("Flexible");

    calib.analyzeGridLF("Flexible", rows, cols, input_files);

    std::cout << "Level 1 log entries: " << std::endl;
    clog::Logger::getInstance().printAll(std::cout, 1);

    if (!cache_file.empty()) {
        cv::FileStorage fs(cache_file, cv::FileStorage::WRITE);
        fs << "calibration" << calib;
        fs.release();
    }

    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
