#ifndef CORNERCACHE_H
#define CORNERCACHE_H

#include <vector>
#include <string>
#include <hdmarker.hpp>

#include "hdcalib.h"

#include <ParallelTime/paralleltime.h>

namespace hdm = hdmarker;

namespace hdcalib {

class CornerCache {
public:
    static CornerCache& getInstance();

    std::vector<hdm::Corner> & operator[](std::string const& filename);

    static void setSNRSigmaMin(double const val);

    std::vector<hdmarker::Corner> getGZ(const std::string &filename);
private:
    std::mutex access_mutex;
    std::map<std::string, std::mutex> access_mutex_by_file;
    CornerCache() {}
    CornerCache(CornerCache const&) = delete;
    void operator=(CornerCache const&) = delete;

    double snr_sigma_min = 0;

    std::map<std::string, std::vector<hdm::Corner>> data;
};

}

#endif // CORNERCACHE_H
