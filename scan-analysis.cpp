#undef NDEBUG
#include <assert.h>   // reinclude the header to update the definition of assert()

#include <iostream>

#include "hdcalib.h"

#include <runningstats/runningstats.h>

using namespace hdmarker;
using namespace hdcalib;

void printIQR(runningstats::QuantileStats<double> & stats) {
    std::cout << "median: " << stats.getMedian()
              << ", IQR: [" << stats.getQuantile(.25) << ", "
              << stats.getQuantile(.75) << "], count: " << stats.getCount() << std::endl;
}

void printStats(std::map<int, std::pair<Corner, Corner> > const& data) {
    runningstats::QuantileStats<double> pixels_per_id, x_per_id, y_per_id;
    for (auto const& _d : data) {
        auto const& d = _d.second;
        assert(d.first.id.x == d.second.id.x || d.first.id.y == d.second.id.y);
        if (d.first.id == d.second.id) {
            continue;
        }
        double const id_diff =    cv::norm(d.first.id - d.second.id);
        cv::Point2d p_diff = d.first.p - d.second.p;
        double const pixel_diff = cv::norm(p_diff);
        assert(id_diff > 0);
        assert(pixel_diff > 0);
        pixels_per_id.push_unsafe(pixel_diff / id_diff);
        x_per_id.push_unsafe(std::abs(p_diff.x) / id_diff);
        y_per_id.push_unsafe(std::abs(p_diff.y) / id_diff);
    }
    std::cout << "length ";
    printIQR(pixels_per_id);
    std::cout << "x ";
    printIQR(x_per_id);
    std::cout << "y ";
    printIQR(y_per_id);
}

void printAngles(
        std::map<int, std::pair<Corner, Corner> > const& a,
        std::map<int, std::pair<Corner, Corner> > const& b) {
    runningstats::QuantileStats<double> stats_alpha;
    for (auto const& _a1 : a) {
        auto const& a1 = _a1.second;
        cv::Point2d const v_a = a1.second.p - a1.first.p;
        for (auto const& _b1 : b) {
            auto const& b1 = _b1.second;
            cv::Point2d const v_b = b1.second.p - b1.first.p;
            double const cos_alpha = v_a.dot(v_b) / std::sqrt(v_a.dot(v_a) * v_b.dot(v_b));
            double const alpha = 180*std::acos(cos_alpha)/M_PI;
            stats_alpha.push_unsafe(alpha);
        }
    }
    printIQR(stats_alpha);
}

void analyze(std::vector<hdmarker::Corner> const& corners, int const layer, int const color) {
    std::map<int, std::pair<Corner, Corner> > rows, columns;
    for (Corner const& c : corners) {
        if (c.layer != layer || c.color != color) {
            continue;
        }
        int const row = c.id.y;
        int const column = c.id.x;

        if (rows.find(row) == rows.end()) {
            rows[row] = std::make_pair(c, c);
        }
        else {
            if (column < rows[row].first.id.x) {
                rows[row].first = c;
            }
            if (column  > rows[row].second.id.x) {
                rows[row].second = c;
            }
        }

        if (columns.find(column) == columns.end()) {
            columns[column] = std::make_pair(c, c);
        }
        else {
            if (column < columns[column].first.id.y) {
                columns[column].first = c;
            }
            if (column  > columns[column].second.id.y) {
                columns[column].second = c;
            }
        }
    }
    std::cout << "Columns, layer " << layer << ", color " << color << std::endl;
    printStats(columns);
    std::cout << "Rows, layer " << layer << ", color " << color << std::endl;
    printStats(rows);
    std::cout << "Angles, layer " << layer << ", color " << color << std::endl;
    printAngles(rows, columns);
    std::cout << std::endl;
}

int main(int argc, char ** argv) {

    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        std::cout << "Analyzing file " << argv[ii] << std::endl;
        std::vector<hdmarker::Corner> const corners = hdcalib::Calib().getCorners(argv[ii], 0.5, false, false);
        for (int layer : {1,2}) {
            for (int color : {0,1}) {
                analyze(corners, layer, color);
            }
        }
        for (int color : {2,3}) {
            analyze(corners, 0, color);
        }
    }
}
