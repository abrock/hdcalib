#include "hdcalib.h"

#include <opencv2/optflow.hpp>
#include <opencv2/video/tracking.hpp>
#include <runningstats/runningstats.h>
#include "gnuplot-iostream.h"

#undef NDEBUG
#include <assert.h>

#include <ParallelTime/paralleltime.h>

namespace {
boost::system::error_code ignore_error_code;
}

namespace hdcalib {

void Calib::printObjectPointCorrectionsStats(const std::map<Point3i, Point3f, cmpPoint3i> &corrections) const {
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

    runningstats::Stats2D<float> reprojection_residuals;

    std::vector<std::vector<double> > data;

#pragma omp critical
    {
        getReprojections(calib, image_index, markers, reprojections);

        for (size_t ii = 0; ii < markers.size() && ii < reprojections.size(); ++ii) {
            cv::Point2d const& marker = markers[ii];
            cv::Point2d const& reprojection = reprojections[ii];
            double const error = distance(marker, reprojection);
            data.push_back({marker.x, marker.y,
                            reprojection.x, reprojection.y,
                            error});
            proj_x.push_unsafe(marker.x, reprojection.x);
            proj_y.push_unsafe(marker.y, reprojection.y);
            double const residual_x = marker.x - reprojection.x;
            double const residual_y = marker.y - reprojection.y;
            res_x.push_back(residual_x);
            res_y.push_back(residual_y);
            reprojection_residuals.push_unsafe(residual_x, residual_y);
            auto const id = getSimpleId(store.get(ii));
            residuals_by_marker[id].push_back(std::make_pair(marker, reprojection));
            errors.push_back(error);
            error_stats.push_unsafe(error);
            error_hist.push_unsafe(error);
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
    std::string const residuals_name = plot_name + ".residuals." + suffix;

    runningstats::HistConfig conf;
    conf.setTitle("Reprojection Residuals")
            .setXLabel("x-residual")
            .setYLabel("y-residual");
    std::pair<double, double> reprojection_residuals_bin = reprojection_residuals.FreedmanDiaconisBinSize();
    reprojection_residuals.plotHist(residuals_name, reprojection_residuals_bin, conf);
    reprojection_residuals.plotHist(residuals_name + "-log", reprojection_residuals_bin, conf.setLogCB());

    conf.setTitle("Reprojection Errors")
            .setXLabel("error [px]")
            .setYLabel("CDF");
    error_stats.plotCDF(plot_name, conf);

    conf.setTitle("Reprojection Errors Histogram")
            .setXLabel("error [px]")
            .setYLabel("estimated PD")
            .setRelative();
    error_stats.plotHist(plot_name, error_stats.FreedmanDiaconisBinSize(), conf);
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
    cv::Mat_<float> errors_raw(plot_size, uint8_t(0));
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
        errors.at<uint8_t>(ii, jj) = cv::saturate_cast<uint8_t>(255*std::sqrt(mean_res.dot(mean_res)));
        errors_raw.at<float>(ii, jj) = std::sqrt(mean_res.dot(mean_res));
    }

    std::string plot_name = prefix + "residuals-by-marker";

    cv::writeOpticalFlow(plot_name + "." + suffix + ".flo", residuals);
    cv::imwrite(plot_name + ".errors." + suffix + ".png", errors);
    cv::imwrite(plot_name + ".errors." + suffix + ".pfm", errors_raw);

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

void Calib::plotReprojectionErrors(std::string const& calibName, string prefix, const string suffix) {
    MarkerMap residuals_by_marker;
    getCalib(calibName);
    fs::create_directory("plots", ignore_error_code);
    prefix = std::string("plots/") + prefix;
    runningstats::QuantileStats<float> res_x, res_y, res_all, errors_x, errors_y, errors_all;
    runningstats::Stats2D<float> res_xy, errors_xy;
    std::multimap<double, std::string> error_overview;
    if ("OpenCV" == calibName || "SimpleOpenCV" == calibName) {
        prepareOpenCVCalibration();
    }
    else {
        prepareCalibration();
    }
    std::cout << "Plotting reprojection errors for calib " << calibName << ":" << std::endl
              << std::string(imagePoints.size(), '-') << std::endl;
#pragma omp parallel for schedule(dynamic)
    for (size_t ii = 0; ii < imagePoints.size(); ++ii) {
        std::vector<float> local_res_x, local_res_y;
        plotReprojectionErrors(calibName, ii, residuals_by_marker, prefix, suffix, local_res_x, local_res_y);
#pragma omp critical
        {
            assert(local_res_x.size() == local_res_y.size());
            runningstats::QuantileStats<float> error_lengths;
            for (size_t ii = 0; ii < local_res_x.size(); ++ii) {
                double const length = std::sqrt(local_res_x[ii] * local_res_x[ii] + local_res_y[ii] * local_res_y[ii]);
                error_lengths.push_unsafe(length);
                res_x.push_unsafe(local_res_x[ii]);
                res_all.push_unsafe(local_res_x[ii]);

                res_xy.push_unsafe(local_res_x[ii], local_res_y[ii]);

                res_y.push_unsafe(local_res_y[ii]);
                res_all.push_unsafe(local_res_y[ii]);

                errors_x.push_unsafe(std::abs(local_res_x[ii]));
                errors_y.push_unsafe(std::abs(local_res_y[ii]));
                errors_all.push_unsafe(std::abs(local_res_x[ii]));
                errors_all.push_unsafe(std::abs(local_res_y[ii]));
                errors_xy.push_unsafe(std::abs(local_res_x[ii]), std::abs(local_res_y[ii]));
            }
            error_overview.insert({error_lengths.getMedian(), imageFiles[ii]});
        }
        std::cout << "." << std::flush;
    }
    std::cout << std::endl;
    clog::L(__func__, 2) << "Median error length overview:" << std::endl;
    for (auto const& it : error_overview) {
        clog::L(__func__, 2) << it.first << "\t" << it.second << std::endl;
    }
    ParallelTime t;
    { // Plot all the residuals
        std::string name_x = prefix + "all-x" + suffix;
        std::string name_y = prefix + "all-y" + suffix;
        std::string name_all = prefix + "all-xy" + suffix;
        runningstats::HistConfig conf;
        conf.setMinMaxX(-3, 3);
        conf.setMinMaxY(-3, 3);
        conf.setXLabel("Residual [px]")
                .setYLabel("Estimated probability density");
        res_x.plotHist(name_x, res_x.FreedmanDiaconisBinSize(), conf.setTitle("x-residuals"));
        res_y.plotHist(name_y, res_y.FreedmanDiaconisBinSize(), conf.setTitle("y-residuals"));
        res_all.plotHist(name_all, res_all.FreedmanDiaconisBinSize(), conf.setTitle("Residuals"));

        res_x.plotCDF(name_x + "-cdf", conf.setTitle("x-residuals [px]").setXLabel("x-residual [px]").setYLabel("Estimated PDF"));
        res_y.plotCDF(name_y + "-cdf", conf.setTitle("y-residuals [px]").setXLabel("y-residual [px]").setYLabel("Estimated PDF"));
        res_all.plotCDF(name_all + "-cdf", conf.setTitle("Residuals"));

        std::pair<double, double> bins = res_xy.FreedmanDiaconisBinSize();
        conf.setIgnoreAmount(0.001);
        res_xy.plotHist(prefix + "hm" + suffix, bins, conf.setXLabel("x").setYLabel("y").setTitle("Residuals heatmap"));
        res_xy.plotHist(prefix + "hmlog" + suffix, bins, conf.setXLabel("x").setYLabel("y").setLogCB().setTitle("Residuals heatmap"));
        plotResidualsByMarkerStats(residuals_by_marker, prefix, suffix);
    }
    { // Plot all the errors
        std::string name_error_x = prefix + "-errors-all-x" + suffix;
        std::string name_error_y = prefix + "-errors-all-y" + suffix;
        std::string name_error_all = prefix + "-errors-all-xy" + suffix;
        runningstats::HistConfig conf;
        conf.setMinMaxX(-3, 3);
        conf.setMinMaxY(-3, 3);
        conf.setXLabel("Error [px]")
                .setYLabel("Estimated probability density");
        errors_x.plotHist(name_error_x, errors_x.FreedmanDiaconisBinSize(), conf.setTitle("x-errors"));
        errors_y.plotHist(name_error_y, errors_y.FreedmanDiaconisBinSize(), conf.setTitle("y-errors"));
        errors_all.plotHist(name_error_all, errors_all.FreedmanDiaconisBinSize(), conf.setTitle("Errors [px]"));

        errors_x.plotCDF(name_error_x + "-cdf", conf.setTitle("x-errors [px]").setXLabel("x-error [px]").setYLabel("Estimated PDF"));
        errors_y.plotCDF(name_error_y + "-cdf", conf.setTitle("y-errors [px]").setXLabel("y-error [px]").setYLabel("Estimated PDF"));
        errors_all.plotCDF(name_error_all + "-cdf", conf.setTitle("Errors [px]"));

        std::pair<double, double> bins_errors = errors_xy.FreedmanDiaconisBinSize();
        conf.setIgnoreAmount(0.001);
        errors_xy.plotHist(prefix + "hm" + suffix, bins_errors, conf.setXLabel("x").setYLabel("y").setTitle("Errors heatmap"));
        errors_xy.plotHist(prefix + "hmlog" + suffix, bins_errors, conf.setXLabel("x").setYLabel("y").setLogCB().setTitle("Errors heatmap"));
        //plotErrorsByMarker(residuals_by_marker);
    }
    clog::L("plotReprojectionErrors", 2) << "Time for global plots: " << t.print();
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
