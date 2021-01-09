#include <iostream>

#include <opencv2/highgui.hpp>

#include <ceres/ceres.h>

#include <runningstats/runningstats.h>

static bool verbose = false;

static bool invert = false;

struct Gauss2D {
    double const x;
    double const y;
    double const val;

    Gauss2D(double const _x, double const _y, double const _val) : x(_x), y(_y), val(_val) {}

    template<class T>
    bool operator() (T const * const background,
                     T const * const amplitude,
                     T const * const m_x,
                     T const * const m_y,
                     T const * const s_x,
                     T const * const s_y,
                     T const * const shear,
                     T * residuals
                     ) const {
        if (s_x[0] <= T(0)) {
            return false;
        }
        if (s_y[0] <= T(0)) {
            return false;
        }
        // Check that determinant is positive
        T const det = s_x[0] * s_y[0] - shear[0] * shear[0];
        if (det <= T(0)) {
            return false;
        }
        residuals[0] = val - eval(background,
                                  amplitude,
                                  m_x,
                                  m_y,
                                  s_x,
                                  s_y,
                                  shear);
        return true;
    }

    template<class T>
    T eval(T const * const background,
           T const * const amplitude,
           T const * const m_x,
           T const * const m_y,
           T const * const s_x,
           T const * const s_y,
           T const * const shear) const {
        T const c_x = x - m_x[0];
        T const c_y = y - m_y[0];
        // Calculate determinant
        T const det = s_x[0] * s_y[0] - shear[0] * shear[0];
        // Intermediates, E^-1 * (x - mu)
        T const i_x = (s_y[0] * c_x - shear[0] * c_y) / det;
        T const i_y = (s_x[0] * c_y - shear[0] * c_x) / det;
        return background[0] + amplitude[0] * ceres::exp(-(c_x * i_x + c_y * i_y)/2.0);
    }
};

void analyze(std::string const& name) {
    cv::Mat_<uint8_t> img = cv::imread(name, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        return;
    }
    if (invert) {
        img = cv::Scalar(255) - img;
    }
    double background = cv::mean(img)[0];
    double amplitude = 255.0 - background;
    double m_x = double(img.cols)/2;
    double m_y = double(img.rows)/2;
    double s_x = double(img.cols)/20;
    double s_y = double(img.rows)*100;
    double shear = 0;

    runningstats::QuantileStats<float> val_stats;

    ceres::Problem problem;
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            uint8_t const val = img(row, col);
            val_stats.push_unsafe(val);
            problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<Gauss2D, 1, 1, 1, 1, 1, 1, 1, 1>(
                            new Gauss2D(col, row, val)
                            ),
                        nullptr,
                        &background,
                        &amplitude,
                        &m_x,
                        &m_y,
                        &s_x,
                        &s_y,
                        &shear
                        );
        }
    }

    background = val_stats.getMedian();
    if (background < val_stats.getMean()) { // White line on black background
        amplitude = val_stats.getMax() - background;
    }
    else { // Black line on white background
        amplitude = val_stats.getMin() - background;
    }

    runningstats::RunningStats x_estimate, y_estimate;
    double weight_sum = 0;
    double x_sum = 0;
    double y_sum = 0;
    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            double const weight = std::abs(double(img(row, col)) - background);
            weight_sum += weight;
            x_estimate.push_unsafe(double(col)*weight);
            x_sum += double(col)*weight;

            y_estimate.push_unsafe(double(row)*weight);
            y_sum += double(row)*weight;
        }
    }
    double const factor = weight_sum / (img.rows * img.cols);

    /*
    m_x = x_estimate.getMean()/factor;
    m_y = y_estimate.getMean()/factor;
    // */
    m_x = x_sum / weight_sum;
    m_y = y_sum / weight_sum;

    s_x = std::sqrt(x_estimate.getStddev()/factor);
    s_y = std::sqrt(y_estimate.getStddev()/factor);

    std::cout << "# initial values for " << name << " 1. background 2. amplitude 3. m_x 4. m_y 5. s_x 6. s_y 7. shear" << std::endl;
    std::cout << "# " << background << " " << amplitude << " " << m_x << " " << m_y << " " << s_x << " " << s_y << " " << shear << std::endl;

    double const ceres_tolerance = 1e-12;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = 150;
    options.function_tolerance = ceres_tolerance;
    options.gradient_tolerance = ceres_tolerance;
    options.parameter_tolerance = ceres_tolerance;
    options.minimizer_progress_to_stdout = verbose;

    ceres::Solver::Summary summary;
    Solve(options, &problem, &summary);

    if (verbose) {
        std::cout << summary.FullReport() << std::endl << std::endl;
        std::cout << summary.BriefReport() << std::endl << std::endl;
    }

    std::cout << "# " << name << " 1. background 2. amplitude 3. m_x 4. m_y 5. s_x 6. s_y 7. shear" << std::endl;
    std::cout << background << " " << amplitude << " " << m_x << " " << m_y << " " << s_x << " " << s_y << " " << shear << std::endl;

    for (int row = 0; row < img.rows; ++row) {
        for (int col = 0; col < img.cols; ++col) {
            uint8_t const val = img(row, col);
            Gauss2D f(col, row, val);
            img(row, col) = cv::saturate_cast<uint8_t>(f.eval(&background,
                                                              &amplitude,
                                                              &m_x,
                                                              &m_y,
                                                              &s_x,
                                                              &s_y,
                                                              &shear
                                                              ));

        }
    }
    cv::imwrite(name + "-fit.tif", img);
}

int main(int argc, char ** argv) {

    for (size_t ii = 1; ii < size_t(argc); ++ii) {
        if (std::string("-v") == std::string(argv[ii])) {
            verbose = true;
            continue;
        }
        if (std::string("--inv") == std::string(argv[ii])) {
            invert = true;
            continue;
        }
        analyze(argv[ii]);
    }
}
