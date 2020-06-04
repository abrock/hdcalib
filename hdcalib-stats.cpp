#include "hdcalib.h"

#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>
#include "gnuplot-iostream.h"

namespace {
boost::system::error_code ignore_error_code;
}

namespace hdcalib {

void Calib::printObjectPointCorrectionsStats(
        const std::map<Point3i, Point3f, cmpSimpleIndex3<Point3i> > &corrections) const {
    runningstats::RunningStats dx, dy, dz, abs_dx, abs_dy, abs_dz, length;
    for (std::pair<cv::Point3i, cv::Point3f> const& it : corrections) {
        dx.push(it.second.x);
        dy.push(it.second.y);
        dz.push(it.second.z);

        abs_dx.push(std::abs(it.second.x));
        abs_dy.push(std::abs(it.second.y));
        abs_dz.push(std::abs(it.second.z));

        length.push(std::sqrt(it.second.dot(it.second)));
    }
    std::cout << "Object point correction stats: " << std::endl
              << "dx: " << dx.print() << std::endl
              << "dy: " << dy.print() << std::endl
              << "dz: " << dz.print() << std::endl
              << "abs(dx): " << abs_dx.print() << std::endl
              << "abs(dy): " << abs_dy.print() << std::endl
              << "abs(dz): " << abs_dz.print() << std::endl
              << "length: " << length.print() << std::endl
              << std::endl;
}

void Calib::plotReprojectionErrors(
        const std::string & calibName,
        const size_t image_index,
        MarkerMap &residuals_by_marker,
        const std::string prefix,
        const std::string suffix,
        std::vector<float>& res_x,
        std::vector<float>& res_y) {
    auto & calib = getCalib(calibName);
    std::string const& filename = imageFiles[image_index];


    std::stringstream plot_command;

    CornerStore const& store = data[filename];

    std::string plot_name = prefix + filename + ".marker-residuals";

    fs::create_directories(fs::path(plot_name).parent_path(), ignore_error_code);

    std::vector<double> errors;

    runningstats::Histogram error_hist(.1);

    runningstats::RunningCovariance proj_x, proj_y;

    runningstats::QuantileStats<double> error_stats;

    std::vector<cv::Point2d> markers, reprojections;

    std::vector<std::vector<double> > data;

#pragma omp critical (plotReprojectionErrors)
    {
        prepareCalibration();
        getReprojections(calib, image_index, markers, reprojections);

        for (size_t ii = 0; ii < markers.size() && ii < reprojections.size(); ++ii) {
            cv::Point2d const& marker = markers[ii];
            cv::Point2d const& reprojection = reprojections[ii];
            double const error = distance(marker, reprojection);
            data.push_back({marker.x, marker.y,
                            reprojection.x, reprojection.y,
                            error});
            proj_x.push(marker.x, reprojection.x);
            proj_y.push(marker.y, reprojection.y);
            res_x.push_back(marker.x - reprojection.x);
            res_y.push_back(marker.y - reprojection.y);
            auto const id = getSimpleId(store.get(ii));
            residuals_by_marker[id].push_back(std::make_pair(marker, reprojection));
            errors.push_back(error);
            error_stats.push(error);
            error_hist.push(error);
        }

/*
        std::cout << "Error stats for image " << filename << ": "
                  << std::endl << error_hist.printBoth() << ", quantiles for .25, .5, .75, .9, .95: "
                  << error_stats.getQuantile(.25) << ", "
                  << error_stats.getQuantile(.5) << ", "
                  << error_stats.getQuantile(.75) << ", "
                  << error_stats.getQuantile(.9) << ", "
                  << error_stats.getQuantile(.95) << ", "
                  << std::endl;

        std::cout << "Covariance between marker values and reprojection values: " << proj_x.getCorr() << " for x and "
                  << proj_y.getCorr() << " for y" << std::endl;

        std::cout << std::endl;
        */

        std::sort(errors.begin(), errors.end());

    } // #pragma omp critical (plotReprojectionErrors)

    gnuplotio::Gnuplot plot;
    std::string const residuals_name = plot_name + ".residuals." + suffix;
    std::string const residuals_data = residuals_name + ".data";
    std::string const error_hist_data = plot_name + ".errors-hist." + suffix + ".data";
    plot_command << std::setprecision(16);
    plot_command << "set term svg enhanced background rgb \"white\";\n"
                 << "set output \"" << residuals_name << ".svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "plot " << plot.file1d(data, residuals_data)
                 << " u ($1-$3):($2-$4) w p pt 7 ps 0.17 notitle;\n"
                 << "set output \"" << plot_name << ".residuals-log." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "set logscale xy;\n"
                 << "plot \"" << residuals_data << "\""
                 << " u (abs($1-$3)):(abs($2-$4)) w p pt 7 ps 0.17 notitle;\n"
                 << "reset;\n"
                 << "set output \"" << plot_name + ".vectors." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "plot \"" << residuals_data << "\""
                 << " u 1:2:($3-$1):($4-$2) w vectors notitle;\n"
                 << "reset;\n"
                 << "set output \"" << plot_name + ".vectors." << suffix << ".2.svg\";\n"
                 << "set title 'Reprojection Residuals';\n"
                 << "plot \"" << residuals_data << "\""
                 << " u 3:4:($1-$3):($2-$4) w vectors notitle;\n"
                 << "reset;\n"
                 << "set key out horiz;\n"
                 << "set output \"" << plot_name + ".images." << suffix << ".svg\";\n"
                 << "set title 'Reprojection vs. original';\n"
                 << "plot \"" << residuals_data << "\""
                 << " u 1:2 w p pt 7 ps 0.17 title 'detected', \"" << plot_name + ".residuals." + suffix << ".data\" u 3:4 w p pt 7 ps 0.17 title 'reprojected';\n"
                 << "set output \"" << plot_name + ".error-dist." << suffix << ".svg\";\n"
                 << "set title 'CDF of the Reprojection Error';\n"
                 << "set xlabel 'error';\n"
                 << "set ylabel 'CDF';\n"
                 << "plot " << plot.file1d(errors, plot_name + ".errors." + suffix + ".data") << " u 1:($0/" << errors.size()-1 << ") w l notitle;\n"
                 << "set logscale x;\n"
                 << "set output \"" << plot_name + ".error-dist-log." << suffix << ".svg\";\n"
                 << "replot;\n"
                 << "reset;\n"
                 << "set output \"" << plot_name + ".error-hist." << suffix << ".svg\";\n"
                 << "set title 'Reprojection Error Histogram';\n"
                 << "set xlabel 'error';\n"
                 << "set ylabel 'absolute frequency';\n"
                 << "plot " << plot.file1d(error_hist.getAbsoluteHist(), error_hist_data)
                 << " w boxes notitle;\n"
                 << "set output \"" << plot_name + ".error-hist-log." << suffix << ".svg\";\n"
                 << "set logscale xy;\n"
                 << "plot \"" << error_hist_data << "\""
                 << "w boxes notitle;\n";

    plot << plot_command.str();

    std::ofstream out(plot_name + "." + suffix + ".gpl");
    out << plot_command.str();

}

void Calib::plotErrorsByMarker(
        const Calib::MarkerMap &map,
        const string prefix,
        const string suffix) {

    for (auto const& it : map) {
        if (it.second.size() < 2) {
            continue;
        }
        std::stringstream _id;
        _id << it.first;
        auto const id = _id.str();

        std::vector<std::vector<float> > local_data;
        for (auto const& d : it.second) {
            local_data.push_back({d.first.x, d.first.y, d.second.x, d.second.y});
        }

        std::string plot_name = "markers." + id;
        gnuplotio::Gnuplot plot;
        std::stringstream plot_command;
        plot_command << std::setprecision(16);
        plot_command << "set term svg enhanced background rgb \"white\";\n"
                     << "set output \"" << plot_name << ".residuals." << suffix << ".svg\";\n"
                     << "set title 'Reprojection Residuals for marker " << id << "';\n"
                     << "plot " << plot.file1d(local_data, plot_name + ".residuals." + suffix + ".data")
                     << " u ($1-$3):($2-$4) w p notitle;\n";


        plot << plot_command.str();

        std::ofstream out(plot_name + "." + suffix + ".gpl");
        out << plot_command.str();
    }
}

void Calib::plotResidualsByMarkerStats(
        const Calib::MarkerMap &map,
        const string prefix,
        const string suffix) {

    if (map.empty()) {
        clog::L("plotResidualsByMarkerStats", 1) << "Marker map is empty." << std::endl;
        return;
    }

    std::vector<std::pair<double, double> > mean_residuals_by_marker;
    int max_x = 1, max_y = 1;
    int min_x = 0, min_y = 0;
    for (auto const& it : map) {
        cv::Point3f const f_coord = getInitial3DCoord(it.first) / markerSize;
        max_x = std::max(max_x, 1+int(std::ceil(f_coord.x)));
        max_y = std::max(max_y, 1+int(std::ceil(f_coord.y)));
        min_x = std::min(min_x, int(std::floor(f_coord.x)));
        min_y = std::min(min_y, int(std::floor(f_coord.y)));
    }
    cv::Size const plot_size(max_x - min_x, max_y - min_y);
    cv::Mat_<cv::Vec2f> residuals(plot_size, cv::Vec2f(0,0));
    cv::Mat_<uint8_t> errors(plot_size, uint8_t(0));
    for (auto const& it : map) {
        const cv::Point2f mean_res = meanResidual(it.second);
        mean_residuals_by_marker.push_back({mean_res.x, mean_res.y});
        cv::Point3f const f_coord = getInitial3DCoord(it.first) / markerSize;
        int ii = int(std::round(f_coord.y)) - min_y;
        int jj = int(std::round(f_coord.x)) - min_x;
        if (ii < 0 || ii >= plot_size.height) {
            continue;
        }
        if (jj < 0 || jj >= plot_size.width) {
            continue;
        }
        residuals.at<cv::Vec2f>(ii, jj) = cv::Vec2f(mean_res);
        errors.at<uint8_t>(int(f_coord.y), int(f_coord.x)) = cv::saturate_cast<uint8_t>(255*std::sqrt(mean_res.dot(mean_res)));
    }

    std::string plot_name = prefix + "residuals-by-marker";

    cv::writeOpticalFlow(plot_name + "." + suffix + ".flo", residuals);
    cv::imwrite(plot_name + ".errors." + suffix + ".png", errors);

    residuals.deallocate();
    errors.deallocate();

    std::stringstream plot_command;
    gnuplotio::Gnuplot plot;

    plot_command << "set term svg enhanced background rgb \"white\";\n"
                 << "set output \"" << plot_name << "." << suffix << ".svg\";\n"
                 << "set title 'Mean reprojection residuals of each marker';\n"
                 << "set size ratio -1;\n"
                 << "plot " << plot.file1d(mean_residuals_by_marker, plot_name + "." + suffix + ".data")
                 << " u 1:2 w p notitle;\n";

    plot << plot_command.str();
    std::ofstream out(plot_name + ".gpl");
    out << plot_command.str();
}

void Calib::plotReprojectionErrors(std::string const& calibName, const string prefix, const string suffix) {
    MarkerMap residuals_by_marker;
    getCalib(calibName);
    runningstats::QuantileStats<float> res_x, res_y, res_all;
#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < imagePoints.size(); ++ii) {
        std::vector<float> local_res_x, local_res_y;
        plotReprojectionErrors(calibName, ii, residuals_by_marker, prefix, suffix, local_res_x, local_res_y);
#pragma omp critical
        {
            for (auto const it : local_res_x) {
                if (std::abs(it) < 3) {
                    res_x.push_unsafe(it);
                    res_all.push_unsafe(it);
                }
            }
            for (auto const it : local_res_y) {
                if (std::abs(it) < 3) {
                    res_y.push_unsafe(it);
                    res_all.push_unsafe(it);
                }
            }
        }
    }
    std::string name_x = prefix + "all-x" + suffix;
    res_x.plotHist(name_x, res_x.FriedmanDiaconisBinSize(), false);
    res_y.plotHist(prefix + "all-y" + suffix, res_y.FriedmanDiaconisBinSize(), false);
    res_all.plotHist(prefix + "all-xy" + suffix, res_all.FriedmanDiaconisBinSize(), false);
    //plotErrorsByMarker(residuals_by_marker);
    plotResidualsByMarkerStats(residuals_by_marker, prefix, suffix);
}

void Calib::printHist(std::ostream& out, runningstats::Histogram const& h, double const threshold) {
    auto hist = h.getRelativeHist();
    double threshold_sum = 0;
    double last_thresh_key = hist.front().first;
    double prev_key = last_thresh_key;
    for (auto const& it : hist) {
        if (it.second > threshold) {
            if (threshold_sum > 0) {
                if (last_thresh_key == prev_key) {
                    out << prev_key << ": " << threshold_sum << std::endl;
                }
                else {
                    out << last_thresh_key << " - " << prev_key << ": " << threshold_sum << std::endl;
                }
            }
            out << it.first << ": " << it.second << std::endl;
            threshold_sum = 0;
            last_thresh_key = it.first;
        }
        else {
            threshold_sum += it.second;
            if (threshold_sum > threshold) {
                out << last_thresh_key << " - " << prev_key << ": " << threshold_sum << std::endl;
                threshold_sum = 0;
                last_thresh_key = it.first;
            }
        }
        prev_key = it.first;
    }
}

void Calib::printObjectPointCorrectionsStats(std::string const& calibName) {
    printObjectPointCorrectionsStats(calibrations[calibName].objectPointCorrections);
}


} // namespace hdcalib
