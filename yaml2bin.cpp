#include <iostream>

#include <hdmarker/hdmarker.hpp>
#include "hdcalib.h"

#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;


void convert(std::string const& input, std::string const& output) {
    int width = 0;
    int height = 0;
    try {
        std::vector<hdmarker::Corner> const orig = hdcalib::Calib::readCorners(input, width, height);
        hdmarker::Corner::writeFile(output, orig);
    }
    catch (std::exception const& e) {
        std::cout << "Got exception for input " << input << ", output " << output << std::endl
                  << e.what() << std::endl;
    }
}

void convertSuffix(std::string const & filename, std::vector<std::string> const& suffixes) {
    for (std::string const& name : {"-pointcache", "-submarkers"}) {
        for (std::string const& suffix : suffixes) {
            std::string const src = filename + name + suffix;
            if (fs::is_regular_file(src)) {
                std::string const dst = filename + name + ".hdmarker.gz";
                convert(src, dst);
            }
        }
    }
    for (std::string const& suffix : suffixes) {
        if (suffix.size() > filename.size()) {
            continue;
        }
        if (filename.substr(filename.size() - suffix.size()) == suffix) {
            std::string const prefix = filename.substr(0, filename.size() - suffix.size());
            std::string const target = prefix + ".hdmarker.gz";
            convert(filename, target);
            return;
        }
    }
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <image file>" << std::endl;
    }
    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        std::string const filename = argv[ii];
        if (!fs::is_regular_file(filename)) {
            std::cout << "Specified file " << filename << " doesn't exist or is not a regular file." << std::endl;
            continue;
        }
        std::vector<std::string> const suffixes = {".yaml", ".yaml.gz", ".json", ".json.gz", ".xml", ".xml.gz"};
        convertSuffix(filename, suffixes);
    }
}
