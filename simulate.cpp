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

    ParallelTime t, total_time;
    std::stringstream time_log;

    try {
        TCLAP::CmdLine cmd("hdcalib simulation tool", ' ', "0.1");

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



    int const num_per_dir = 30;

    std::random_device rd;
    std::default_random_engine engine(rd());
    std::normal_distribution<double> gauss;

    double const image_scale = 500;

    double const noise = 0.001;

    for (size_t ii = 0; ii < 100; ++ii) {
        std::vector<hdmarker::Corner> markers;


        double const veclength = std::uniform_real_distribution<double>(.0001, .1)(engine);
        double const depth = std::uniform_real_distribution<double>(90,110)(engine);

        double rot_vec[3] = {gauss(engine), gauss(engine), gauss(engine)};
        double const tmp_vec_length = std::sqrt(rot_vec[0]*rot_vec[0] + rot_vec[1]*rot_vec[1] + rot_vec[2]*rot_vec[2]);
        for (size_t jj = 0; jj < 3; ++jj) {
            rot_vec[jj] *= veclength / tmp_vec_length;
        }

        double rot_mat[9];

        Calib::rot_vec2mat(rot_vec, rot_mat);

        runningstats::RunningStats stat_x, stat_y;

        for (int yy = 0; yy < num_per_dir; yy++) {
            for (int xx = 0; xx < num_per_dir; xx++) {
                cv::Point2i id(xx, yy);
                int const page = 0;
                Corner current(cv::Point2f(), id, page);
                cv::Point3f loc3d = calib.getInitial3DCoord(current, 0);
                loc3d = cv::Point3f(
                            loc3d.x * rot_mat[0] + loc3d.y * rot_mat[1] + loc3d.z * rot_mat[2],
                        loc3d.x * rot_mat[3] + loc3d.y * rot_mat[4] + loc3d.z * rot_mat[5],
                        loc3d.x * rot_mat[6] + loc3d.y * rot_mat[7] + loc3d.z * rot_mat[8] + depth
                        );

                current.p = image_scale * cv::Point2f(loc3d.x/loc3d.z, loc3d.y/loc3d.z);
                current.p += noise * cv::Point2f(gauss(engine), gauss(engine));
                markers.push_back(current);
                stat_x.push(current.p.x);
                stat_y.push(current.p.y);
            }
        }
        std::string filename = std::to_string(ii) + ".png";
        calib.addInputImage(filename, markers);
        cv::Mat paint(1100, 1100, CV_8UC3);
        calib.paintSubmarkers(markers, paint, 1);
        cv::imwrite(filename, paint);
        std::cout << "Marker position stats:" << std::endl
                  << "x: " << stat_x.print() << std::endl
                  << " y: " << stat_y.print() << std::endl;
    }

    calib.setImageSize(cv::Mat_<double>(10,10));
    calib.setRecursionDepth(0);
    calib.setValidPages({0});

    TIMELOG("Setting up calibration images");

    calib.openCVCalib();

    TIMELOG("openCVCalib");

    calib.CeresCalib();

    TIMELOG("CeresCalib");

    // Create 9x9 lightfield
    int const grid = 4;
    std::vector<std::string> lightfield;
    for (int ii = -grid; ii <= grid; ++ii) {
        for (int jj = -grid; jj <= grid; ++jj) {

            std::vector<hdmarker::Corner> markers;

            for (int xx = 0; xx < num_per_dir; xx++) {
                for (int yy = 0; yy < num_per_dir; yy++) {
                    cv::Point2i id(xx, yy);
                    int const page = 0;
                    Corner current(cv::Point2f(), id, page);
                    cv::Point3f loc3d = calib.getInitial3DCoord(current, 100);
                    loc3d.x += 10*jj;
                    loc3d.y += 10*ii;

                    current.p = image_scale * cv::Point2f(loc3d.x / loc3d.z, loc3d.y/loc3d.z);
                    current.p += noise * cv::Point2f(gauss(engine), gauss(engine));
                    markers.push_back(current);
                }
            }
            std::string filename = std::to_string(ii) + ";" + std::to_string(jj) + ".png";
            calib.addInputImageAfterwards(filename, markers);
            lightfield.push_back(filename);
        }
    }

    TIMELOG("Setting up lightfield")

    calib.CeresCalib();

    TIMELOG("Second CeresCalib");

    cv::Vec3d col_vec, row_vec, rot_vec;
    calib.getGridVectors(2*grid+1, 2*grid+1, lightfield, row_vec, col_vec);

    TIMELOG("getGridVectors");

    calib.getRectificationRotation(2*grid+1, 2*grid+1, lightfield, rot_vec);

    TIMELOG("getRectificationRotation");

    //  microbench_measure_output("app finish");
    return EXIT_SUCCESS;
}
