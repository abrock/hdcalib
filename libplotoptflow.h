#ifndef LIBPLOTOPTFLOW_H
#define LIBPLOTOPTFLOW_H

#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/optflow.hpp>

#include <runningstats/runningstats.h>

namespace hdflow {
cv::Mat_<cv::Vec3b> colorFlow(
        const cv::Mat_<cv::Vec2f>& flow,
        double& factor,
        const double scaleFactor);
cv::Mat_<cv::Vec3b> addArrows(cv::Mat_<cv::Vec3b> const& src,
                              cv::Mat_<cv::Vec2f> const& flow,
                              double factor = 1,
                              double const length_factor = 1,
                              cv::Scalar const& color = {0,0,255,0}
        );

void run(const std::string prefix,
        const cv::Mat_<cv::Vec2f> &flow,
        double factor = 1,
        const double length_factor = 1,
        const cv::Scalar &color = {0,0,0,0},
         const bool run_fit = false);

template<int NUM, int DEG>
void fitSpline(
        std::string const prefix,
        const cv::Mat_<cv::Vec2f> &flow,
        double factor = 1,
        const double length_factor = 1,
        const cv::Scalar &color = {0,0,0,0});


cv::Mat_<cv::Vec3b> plotWithArrows(
        const cv::Mat_<cv::Vec2f> &flow,
        double factor = 1.0,
        const double length_factor = 1,
        const cv::Scalar &color = {0,0,0,0});

void gnuplotWithArrows(const std::string &prefix,
        const cv::Mat_<cv::Vec2f> &flow,
        double factor = 1,
        const double arrow_factor = 1,
        const cv::Scalar &color = {0,0,0,0});



} // namespace hdflow


#endif // LIBPLOTOPTFLOW_H
