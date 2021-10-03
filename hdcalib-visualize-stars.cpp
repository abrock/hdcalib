/**
  This file was heavily inspired (read: mostly copied) from
  https://github.com/puzzlepaint/camera_calibration
  therefore we include their license:

Copyright 2019 ETH Zürich, Thomas Schöps

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
  may be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

  */
#undef NDEBUG
#include <cassert>
#include "hdcalib.h"

namespace hdcalib {

void Calib::CreateReprojectionErrorDirectionVisualization(
        std::vector<cv::Vec2f> const& points,
        std::vector<cv::Vec2f> const& residuals,
        cv::Mat_<cv::Vec3b> & visualization) {
    // Show all reprojection errors of a camera in one image, colored by direction (as in optical flow),
    // to see whether in certain areas all errors have the same direction.
    visualization = cv::Mat_<cv::Vec3b>(imageSize, cv::Vec3b(0, 0, 0));

    assert(points.size() == residuals.size());


    for (size_t ii = 0; ii < residuals.size(); ++ii) {
        Vec2d pixel;
        Vec2d reprojection_error = residuals[ii];

        int fx = std::round(points[ii][0]);
        int fy = std::round(points[ii][1]);

        if (fx >= 0 && fy >= 0 &&
                fx < static_cast<int>(visualization.cols) &&
                fy < static_cast<int>(visualization.rows)) {
            double dir = atan2(reprojection_error[1], reprojection_error[0]);  // from -M_PI to M_PI
            visualization(fx, fy) = cv::Vec3b(
                        127 + 127 * std::sin(dir),
                        127 + 127 * std::cos(dir),
                        127);
        }
    }
}
/*
void Calib::CreateReprojectionErrorMagnitudeVisualization(
        std::vector<cv::Vec2f> const& points,
        std::vector<cv::Vec2f> const& residuals,
        cv::Mat_<cv::Vec3b> & visualization) {
    Image<double> max_error_image(calibration.intrinsics[camera_index]->width(), calibration.intrinsics[camera_index]->height());
    max_error_image.SetTo(-1.f);

    for (int imageset_index = 0; imageset_index < dataset.ImagesetCount(); ++ imageset_index) {
        if (!calibration.image_used[imageset_index]) {
            continue;
        }

        const SE3d& image_tr_global = calibration.image_tr_global(camera_index, imageset_index);
        Mat3d image_r_global = image_tr_global.rotationMatrix();
        const Vec3d& image_t_global = image_tr_global.translation();

        shared_ptr<const Imageset> imageset = dataset.GetImageset(imageset_index);
        const vector<PointFeature>& features = imageset->FeaturesOfCamera(camera_index);

        for (const PointFeature& feature : features) {
            Vec3d local_point = image_r_global * calibration.points[feature.index] + image_t_global;
            Vec2d pixel;
            if (calibration.intrinsics[camera_index]->Project(local_point, &pixel)) {
                Vec2d reprojection_error = pixel - feature.xy.cast<double>();

                int fx = feature.xy.x();
                int fy = feature.xy.y();

                if (fx >= 0 && fy >= 0 &&
                        fx < static_cast<int>(max_error_image.width()) &&
                        fy < static_cast<int>(max_error_image.height())) {
                    // Update max error magnitude visualization
                    max_error_image(fx, fy) = std::max(max_error_image(fx, fy), reprojection_error.norm());
                }
            }
        }
    }

    double max_error_in_image = 0.f;
    for (u32 y = 0; y < max_error_image.height(); ++ y) {
        for (u32 x = 0; x < max_error_image.width(); ++ x) {
            if (max_error_image(x, y) >= 0) {
                max_error_in_image = std::max(max_error_in_image, max_error_image(x, y));
            }
        }
    }
    LOG(1) << "Maximum reprojection error: " << max_error_in_image;

    if (max_error <= 0) {
        max_error = max_error_in_image;
    }

    visualization->SetSize(max_error_image.width(), max_error_image.height());
    for (u32 y = 0; y < max_error_image.height(); ++ y) {
        for (u32 x = 0; x < max_error_image.width(); ++ x) {
            if (max_error_image(x, y) >= 0) {
                double factor = std::min(1., max_error_image(x, y) / max_error);
                visualization->at(x, y) = Vec3u8(255.99f * factor, 255.99f * (1 - factor), 0);
            } else {
                // No feature detection at this pixel.
                visualization->at(x, y) = Vec3u8(0, 0, 0);
            }
        }
    }
}
*/

}
