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
#include <ParallelTime/paralleltime.h>

#include "hdcalib.h"

#include <random>

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

int main(int argc, char* argv[]) {

    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    std::string cache_file = "";

    bool verbose2 = false;
    bool plot_synthetic_markers = true;

    try {
        TCLAP::CmdLine cmd("hdcalib simulation tool", ' ', "0.1");

        TCLAP::ValueArg<std::string> cache("c", "cache", "cache file", false, "", "filename");
        cmd.add(cache);

        cmd.parse(argc, argv);

        cache_file = cache.getValue();
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    using namespace hdcalib;

    Calib calib;

    /**
     * @brief image_scale f_x and f_y to be used in the simulation. Also half of the plotted image's width and height.
     */
    double const image_scale = 3240;

    /**
     * @brief marker_size value to be set for hdmarker::Corner::size, not to be confused with the physical dimension of a marker.
     */
    double const marker_size = 30;

    /**
     * @brief noise Gaussian noise added to the 2D location of the projected markers.
     */
    double const noise = 0.0;

    /**
     * @brief num_per_dir Number of markers per direction, total number is num_per_dir².
     */
    int const num_per_dir = 150;

    double const mean_depth = 50;

    calib.setImageSize(cv::Mat_<double>(2*image_scale,2*image_scale));
    cv::Size image_size(2*image_scale,2*image_scale);
    calib.setRecursionDepth(1);
    calib.setValidPages({0});



    bool has_cached_calib = false;
    if (fs::is_regular_file(cache_file)) {
        try {
            std::cout << "Reading cached calibration results..." << std::flush;
            cv::FileStorage fs(cache_file, cv::FileStorage::READ);
            cv::FileNode n = fs["calibration"];
            n >> calib;
            has_cached_calib = true;
            fs.release();
            std::cout << " done." << std::endl;
            calib.purgeInvalidPages();
        }
        catch (std::exception const& e) {
            std::cout << "Reading cache file failed with exception:" << std::endl
                      << e.what() << std::endl;
        }
        TIMELOG("Reading cached result");
    }

    cv::Mat_<double> const tvec_offset =
            .5 *
            cv::Mat_<double>({-1, -1, 0}) *
            calib.getMarkerSize()*num_per_dir/calib.getCornerIdFactor();

    std::random_device rd;
    std::default_random_engine engine(rd());
    std::normal_distribution<double> gauss;


    if (!has_cached_calib) {

        calib.initializeCameraMatrix(image_scale, image_scale, image_scale);

        runningstats::RunningStats global_stat_x, global_stat_y;

        for (size_t ii = 0; ii < 16; ++ii) {
            std::vector<hdmarker::Corner> markers;


            double const veclength = std::uniform_real_distribution<double>(.0001, .1)(engine);
            double const depth = std::uniform_real_distribution<double>(mean_depth*.9, mean_depth*1.1)(engine);

            cv::Mat_<double> rot_vec = {3*gauss(engine), 3*gauss(engine), gauss(engine)};
            rot_vec *= veclength / (std::sqrt(rot_vec.dot(rot_vec)));

            cv::Mat_<double> const t_vec = cv::Mat_<double>({gauss(engine), gauss(engine), depth}) + tvec_offset;

            runningstats::RunningStats stat_x, stat_y;

            for (int yy = 0; yy < num_per_dir; yy++) {
                for (int xx = 0; xx < num_per_dir; xx++) {
                    int const page = 0;
                    Corner current(cv::Point2f(), cv::Point2i(xx,yy), page);
                    current.size = marker_size;
                    cv::Vec3d loc3d = calib.get3DPoint(current, rot_vec, t_vec);

                    current.p = calib.project(loc3d);
                    current.p += noise * cv::Point2f(gauss(engine), gauss(engine));
                    if (Calib::validPixel(current.p, image_size)) {
                        markers.push_back(current);
                        stat_x.push(current.p.x);
                        stat_y.push(current.p.y);

                        global_stat_x.push(current.p.x);
                        global_stat_y.push(current.p.y);
                    }
                }
            }
            std::string filename = std::to_string(ii) + ".png";
            //calib.addInputImage(filename, markers, rot_vec, t_vec);
            calib.addInputImage(filename, markers);
            if (plot_synthetic_markers) {
                cv::Mat_<cv::Vec3b> paint(2*image_scale, 2*image_scale, cv::Vec3b(50,50,50));
                calib.paintSubmarkers(markers, paint, 1);
                cv::imwrite(filename, paint);
            }
            if (verbose2) {
                std::cout << "Marker position stats:" << std::endl
                          << "x: " << stat_x.print() << std::endl
                          << " y: " << stat_y.print() << std::endl;
            }
        }
        std::cout << "Global marker position stats:" << std::endl
                  << "x: " << global_stat_x.print() << std::endl
                  << " y: " << global_stat_y.print() << std::endl;

        TIMELOG("Setting up calibration images");

        //*
        calib.openCVCalib();

        TIMELOG("openCVCalib");

        calib.CeresCalib();

        TIMELOG("CeresCalib");
        // */

    }

    // Create 9x9 lightfield
    int const grid = 4;
    std::vector<std::string> lightfield;


    double const grid_angle = 10;
    double const grid_scale = 17.5;

    cv::Mat_<double> gt_rot_vec = {0,0,-std::sin(grid_angle*M_PI/180)};
    cv::Mat_<double> gt_row_vec{grid_scale*std::cos(grid_angle*M_PI/180),grid_scale*std::sin(grid_angle*M_PI/180),0};
    cv::Mat_<double> gt_col_vec{-grid_scale*std::sin(grid_angle*M_PI/180),grid_scale*std::cos(grid_angle*M_PI/180),0};
    runningstats::RunningStats global_stat_x, global_stat_y;
    size_t grid_counter = 0;
    std::map<std::string, cv::Mat> plots;
    for (int ii = -grid; ii <= grid; ++ii) {
        for (int jj = -grid; jj <= grid; ++jj, ++grid_counter) {
            std::vector<hdmarker::Corner> markers;

            cv::Mat_<double> const t_vec = cv::Mat_<double>({0,0, mean_depth}) + jj * gt_row_vec + ii * gt_col_vec + tvec_offset;

            runningstats::RunningStats stat_x, stat_y;

            for (int yy = 0; yy < num_per_dir; yy++) {
                for (int xx = 0; xx < num_per_dir; xx++) {
                    int const page = 0;
                    Corner current(cv::Point2f(), cv::Point2i(xx,yy), page);
                    current.size = marker_size;
                    cv::Vec3d loc3d = calib.get3DPoint(current, gt_rot_vec, t_vec);

                    current.p = calib.project(loc3d);
                    current.p += noise * cv::Point2f(gauss(engine), gauss(engine));

                    if (Calib::validPixel(current.p, image_size)) {
                        markers.push_back(current);
                        stat_x.push(current.p.x);
                        stat_y.push(current.p.y);

                        global_stat_x.push(current.p.x);
                        global_stat_y.push(current.p.y);
                    }
                }
            }
            std::string filename = "lf-" + Calib::tostringLZ(grid_counter, 2) + ".png";
            lightfield.push_back(filename);
            //calib.addInputImage(filename, markers, rot_vec, t_vec);
            calib.addInputImage(filename, markers);
            if (plot_synthetic_markers) {
                cv::Mat_<cv::Vec3b> paint(2*image_scale, 2*image_scale, cv::Vec3b(50,50,50));
                calib.paintSubmarkers(markers, paint, 1);
                plots[filename] = paint;
                cv::imwrite(filename, paint);
            }
            if (verbose2) {
                std::cout << "Marker position stats:" << std::endl
                          << "x: " << stat_x.print() << std::endl
                          << " y: " << stat_y.print() << std::endl;
            }
        }
    }
    std::cout << "Global marker position stats:" << std::endl
              << "x: " << global_stat_x.print() << std::endl
              << " y: " << global_stat_y.print() << std::endl;

    TIMELOG("Setting up lightfield");

    //*
    if (!has_cached_calib) {
        calib.CeresCalib();
        TIMELOG("Second CeresCalib");
    }
    // */

    cv::Vec3d col_vec(4,-6,9), row_vec(2,9,-5), rot_vec;
    calib.getGridVectors(2*grid+1, 2*grid+1, lightfield, row_vec, col_vec);

    clog::L("row-vector difference", 0) << row_vec + gt_col_vec << std::endl;
    clog::L("column-vector difference", 0) << col_vec + gt_row_vec << std::endl;

    TIMELOG("getGridVectors");

    calib.getRectificationRotation(2*grid+1, 2*grid+1, lightfield, rot_vec);

    TIMELOG("getRectificationRotation");

    if (!cache_file.empty()) {
        cv::FileStorage fs(cache_file, cv::FileStorage::WRITE);
        fs << "calibration" << calib;
        fs.release();
        TIMELOG("Writing cache file");
    }

    clog::L("result", 0) << "Ground truth rectification: " << gt_rot_vec << std::endl
                         << "Estimated rectification: " << rot_vec << std::endl
                         << "difference: " << (gt_rot_vec - rot_vec) << std::endl
                         << "normalized difference: " << 2*(gt_rot_vec - rot_vec)/(cv::norm(gt_rot_vec) + cv::norm(rot_vec)) << std::endl;

    cv::Mat_<cv::Vec2f> remap = calib.getCachedUndistortRectifyMap();

    TIMELOG("getCachedUndistortRectifyMap");

    for (auto const& it : plots) {
        cv::Mat remapped;
        cv::remap(it.second, remapped, remap, cv::Mat(), cv::INTER_LINEAR);
        clog::L("remapping", 3) << "Remapped image " << it.first;
        cv::imwrite(std::string("remapped-") + it.first, remapped);
        clog::L("remapping", 3) << "Saved remapped image";
    }

    TIMELOG("Remapping all synthetic images");

    std::cout << "Level 1 log entries: " << std::endl;
    clog::Logger::getInstance().printAll(std::cout, 1);

    std::cout << time_log.str() << std::endl;

    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
