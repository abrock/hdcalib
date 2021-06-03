#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/rational.hpp>

namespace fs = boost::filesystem;

//#include <hdmarker/hdmarker.hpp>
//#include <hdmarker/subpattern.hpp>

#include <ceres/ceres.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "gnuplot-iostream.h"

#include <runningstats/runningstats.h>

#include <ParallelTime/paralleltime.h>

//using namespace hdmarker;

namespace rs = runningstats;

static const float min_fit_contrast = 1.0;
static const float min_fitted_contrast = 3.0; //minimum amplitude of fitted gaussian
//static const float min_fit_contrast_gradient = 0.05;
static const float rms_use_limit = 30.0/255.0;
static const float recurse_min_len = 3.0;
static const int int_search_range = 2;
static const int int_extend_range = 2;
static const float extent_range_limit_size = 8;
static const double subfit_max_range = .2;
static const double fit_gauss_max_tilt = 0.1;
static const float max_size_diff = 1.0;
static const float max_sigma_diff = 4.0;
static const float gauss_sample_weight_crop = 0.2;
static const float sigma_anisotropy_penalty = 0.2;
static const double rms_size_mul_max = 20.0;

static int safety_border = 2;

static const double bg_weight = 0.0;
static const double mul_size_sigma = 0.125;
static const double tilt_max_rms_penalty = 10.0;
static const double border_frac = 0.15;

static const float max_retry_dist = 0.1;

static const float max_sigma_10 = 0.8;
static const float max_sigma_20 = 0.8;
static const float min_sigma_px = 0.6; //FIXME lower for non-bayer!
static const float min_sigma = 0.1;

//FIXME add possibility to reject too small sigma (less than ~one pixel (or two for bayer))

static const int min_fit_data_points = 9;

cv::Mat gt_c, gt_r, gt_t;
bool eval_gt = false;

#include "randutils.hpp"

boost::system::error_code ignore_error;

struct GenGauss2dPlaneDirectError {
    GenGauss2dPlaneDirectError(double val, int x, int y, double w, double h, double px, double py, double sw);

    /**
 * used function:
 */
    template <typename T>
    bool operator()(const T* const p,
                    T* residuals) const;

    template<typename T>
    T evaluateModel(const T * const p) const;

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(double val, int x, int y, double w, double h, double px, double py, double sw);

    int x_, y_;
    double w_, px_, py_, h_, sw_, val_;
};

template<typename T> T cos_sq(const T &a)
{
    return ceres::cos(a)*ceres::cos(a);
}

template<typename T> T sin_sq(const T &a)
{
    return ceres::sin(a)*ceres::sin(a);
}

/**
 * used function:
 */
template <typename T>
bool GenGauss2dPlaneDirectError::operator()(const T* const p,
                                            T* residuals) const {
    T x2 = T(x_) - (T(px_)+sin(p[0])*T(w_*subfit_max_range));
    T y2 = T(y_) - (T(py_)+sin(p[1])*T(h_*subfit_max_range));
    T dx = x2;
    T dy = y2;
    T sx2 = T(2.0)*p[3]*p[3];
    //max angle ~70°
    T sigma_y = abs(p[3])*(T(1.25)+T(0.75)*sin(p[7]));
    T sy2 = T(2.0)*sigma_y*sigma_y;
    T xy2 = x2*y2;
    x2 = x2*x2;
    y2 = y2*y2;

    T a = cos_sq(p[8])/sx2 + sin_sq(p[8])/sy2;
    T b = -sin_sq(T(2)*p[8])/(T(2)*sx2) + sin_sq(T(2)*p[8])/(T(2)*sy2);
    T c = sin_sq(p[8])/sx2 + cos_sq(p[8])/sy2;

    residuals[0] = (T(val_) - (p[4] + p[5]*dx + p[6]*dy +
            (p[2]-p[4])*exp(-(a*x2-T(2)*b*xy2+c*y2))))*(T(1)+T(sigma_anisotropy_penalty)*(std::max(abs(sigma_y/p[3]),abs(p[3]/sigma_y))))
            *T(sw_);

    return true;
}

/**
   * used function:
   */
template <typename T>
T GenGauss2dPlaneDirectError::evaluateModel(const T* const p) const {
    T x2 = T(x_) - (T(px_)+sin(p[0])*T(w_*subfit_max_range));
    T y2 = T(y_) - (T(py_)+sin(p[1])*T(h_*subfit_max_range));
    T dx = x2;
    T dy = y2;
    T sx2 = T(2.0)*p[3]*p[3];
    //max angle ~70°
    T sigma_y = abs(p[3])*(T(1.25)+T(0.75)*sin(p[7]));
    T sy2 = T(2.0)*sigma_y*sigma_y;
    T xy2 = x2*y2;
    x2 = x2*x2;
    y2 = y2*y2;

    T a = cos_sq(p[8])/sx2 + sin_sq(p[8])/sy2;
    T b = -sin_sq(T(2)*p[8])/(T(2)*sx2) + sin_sq(T(2)*p[8])/(T(2)*sy2);
    T c = sin_sq(p[8])/sx2 + cos_sq(p[8])/sy2;

    return ((p[4] + p[5]*dx + p[6]*dy +
            (p[2]-p[4])*exp(-(a*x2-T(2)*b*xy2+c*y2))))*(T(1)+T(sigma_anisotropy_penalty)*(std::max(abs(sigma_y/p[3]),abs(p[3]/sigma_y))))
            *T(sw_);

    return true;
}

GenGauss2dPlaneDirectError::GenGauss2dPlaneDirectError(double val, int x, int y, double w, double h, double px, double py, double sw)
    : val_(val), x_(x), y_(y), w_(w), h_(h), px_(px), py_(py), sw_(sw){}

ceres::CostFunction *GenGauss2dPlaneDirectError::Create(double val, int x, int y, double w, double h, double px, double py, double sw) {
    return (new ceres::AutoDiffCostFunction<GenGauss2dPlaneDirectError, 1, 9>(
                new GenGauss2dPlaneDirectError(val, x, y, w, h, px, py, sw)));
}

class FitExperiment {
public:
    int radius = 5;
    double sigma = 2;
    double scale = 128;

    bool verbose = false;

    double rms = -1;
    double snr = -1;

    std::vector<double> params = {0.01,0.01,scale,0.01,0.01,0.01,0.01,0.01,0.01,0.01};
    std::vector<double> gt_params = {0,0,scale,sigma,0,0,0,0,0,0};

    double fit_x = 100;
    double fit_y = 100;

    double gt_x = 0;
    double gt_y = 0;

    double diff_x = 100;
    double diff_y = 100;

    double noise_sigma = -1;

    double error_length = -1;

    std::string img_prefix = "";

    bool success = false;

    double getFitSigma() const {
        return params[3];
    }
    double getFitScale() const {
        return params[2];
    }
    rs::QuantileStats<float> signal_stats, noise_stats;

    void runFitImg(cv::Mat_<float> const& img) {
        signal_stats.clear();
        noise_stats.clear();

        ceres::Problem problem;
        radius = double(img.rows-1)/2;
        double max_img = 0;
        double min_img = 0;
        cv::minMaxIdx(img, &min_img, &max_img);
        double const median = (max_img + min_img)/2;
        double const center = img(img.rows/2, img.cols/2);
        scale = max_img - min_img;
        double background = min_img;
        if (center < median) {
            scale *= -1;
            background = max_img;
        }
        sigma = radius/2;
        params = {0,0,
                  scale,sigma,background,0,0,0,0,0};
        std::vector<double> initial_params = params;
        success = true;
        rs::QuantileStats<float> src_stats;
        double const model_radius = 2*radius;
        for (int yy = 0; yy < img.rows; ++yy) {
            for (int xx = 0; xx < img.cols; ++xx) {
                double const val = img(yy, xx);
                max_img = std::max(max_img, val);
                min_img = std::min(min_img, val);
                src_stats.push_unsafe(val);
                problem.AddResidualBlock(GenGauss2dPlaneDirectError::Create(
                                             val,
                                             double(xx) - radius, double(yy) - radius,
                                             model_radius, model_radius,
                                             0, 0, 1), nullptr, params.data());
            }
        }

        ceres::Solver::Options options;
        double const tolerance = 1e-12;
        options.function_tolerance = tolerance;
        options.parameter_tolerance = tolerance;
        options.gradient_tolerance = tolerance;
        options.minimizer_progress_to_stdout = verbose;
        options.max_num_iterations = 1'000;
        options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
        options.linear_solver_type = ceres::DENSE_QR;
        //options.preconditioner_type = ceres::IDENTITY;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (verbose) {
            std::cout << summary.FullReport() << std::endl;
            std::cout << summary.BriefReport() << std::endl;
            std::cout << "Fit vs. Initial parameters:" << std::endl;
            for (size_t ii = 0; ii < 10; ++ii) {
                std::cout << "#" << ii << ":\t" << params[ii] << "\t" << initial_params[ii] << std::endl;
            }
            std::cout << "Max image value: " << max_img << std::endl;
        }
        double sigma_y = abs(params[3])*(1.25+0.75*sin(params[7]));

        double contrast = abs(params[2]-params[4])*exp(-(0.25/(2.0*params[3]*params[3])+0.25/(2.0*params[3]*params[3])));
        contrast = std::min(255.0, contrast);

        /*
        for (GenGauss2dPlaneDirectError const& functor : functors) {
            double const signal = functor.evaluateModel(params.data());
            signal_stats.push_unsafe(signal);
            double const noise = functor.val_ - signal;
            noise_stats.push_unsafe(noise);
        }
        // */
        cv::Mat_<uint8_t> fit_img(img.size(), uint8_t(0)), residuals_img(img.size(), uint8_t(0));
        for (int yy = 0; yy < img.rows; ++yy) {
            for (int xx = 0; xx < img.cols; ++xx) {
                double const val = img(yy, xx);
                GenGauss2dPlaneDirectError model(val,
                                                 double(xx) - radius, double(yy) - radius,
                                                 model_radius, model_radius,
                                                 0, 0, 1);
                double const signal = model.evaluateModel(params.data());
                signal_stats.push_unsafe(signal);
                double const noise = val - signal;
                residuals_img(yy,xx) = cv::saturate_cast<uint8_t>(std::abs(noise*1.7));
                noise_stats.push_unsafe(noise);
                fit_img(yy,xx) = cv::saturate_cast<uint8_t>(signal);
            }
        }
        if (!img_prefix.empty()) {
            cv::imwrite(img_prefix + "src.png", cv::Mat_<uint8_t>(img));
            cv::imwrite(img_prefix + "fit.png", cv::Mat_<uint8_t>(fit_img));
            cv::imwrite(img_prefix + "res.png", cv::Mat_<uint8_t>(residuals_img));
        }

        cv::Point2f size(radius, radius);
        double scale_f =
                (std::max(abs(params[3]),sigma_y) / size.x - min_sigma*0.5)
                /contrast
                *(1.0+tilt_max_rms_penalty*(abs(params[5])+abs(params[6]))/fit_gauss_max_tilt);

        rms = sqrt(summary.final_cost/problem.NumResiduals())*scale_f;
        snr = signal_stats.getAccurateVariance() / noise_stats.getAccurateVariance();
        fit_x = std::sin(params[0])*(model_radius*subfit_max_range);
        fit_y = std::sin(params[1])*(model_radius*subfit_max_range);


        diff_x = fit_x - gt_x;
        diff_y = fit_y - gt_y;
        error_length = std::sqrt(diff_x*diff_x + diff_y*diff_y);

        for (double val : {
             fit_x, fit_y, gt_x, gt_y, diff_x, diff_y, error_length, rms, snr, scale_f,
             params[0],
             params[1],
             params[2],
             params[3],
             params[4],
             params[5],
             params[6],
             params[7],
             params[8],
    }) {
            success &= std::isfinite(val);
        }

        success &= rms > 0;
        success &= rms < 100;
        success &= snr > 0;


        if (verbose) {
            std::cout << "RMS: " << rms << std::endl;
            std::cout << "SNR: " << snr << std::endl;
            std::cout << "Signal: " << signal_stats.print() << std::endl;
            std::cout << "Noise: " << noise_stats.print() << std::endl;
            std::cout << "Src stats: " << src_stats.print() << std::endl;
            std::cout << "Fit x, y: " << fit_x << ", " << fit_y << std::endl;
            std::cout << "Localisation error: " << error_length << std::endl;
        }

    }

    void runFit() {
        static randutils::mt19937_rng rng;
        success = true;
        double const gt_range = 0.5;
        gt_x = rng.variate<double, std::uniform_real_distribution>(-gt_range, gt_range);
        gt_y = rng.variate<double, std::uniform_real_distribution>(-gt_range, gt_range);

        ceres::Problem problem;
        gt_params = params = {
                std::asin(gt_x/(double(radius)*subfit_max_range)),
                std::asin(gt_y/(double(radius)*subfit_max_range)),
                scale,sigma,0,0,0,0,0,0};

        rs::QuantileStats<float> src_stats, fit_stats;

        cv::Mat_<uint8_t> img(2*radius+1, 2*radius+1, uint8_t(0));
        double max_value = 0;
        std::vector<GenGauss2dPlaneDirectError> functors;
        for (int x = -radius; x <= radius; ++x) {
            for (int y = -radius; y <= radius; ++y) {
                GenGauss2dPlaneDirectError tmp_model(0, x, y, radius, radius, 0, 0, 1);
                double val = tmp_model.evaluateModel(gt_params.data());
                if (noise_sigma > 0) {
                    val += rng.variate<double, std::normal_distribution>(0, noise_sigma);
                }
                img(y+radius, x+radius) = cv::saturate_cast<uint8_t>(val);
                src_stats.push_unsafe(val);
                max_value = std::max(max_value, val);
                problem.AddResidualBlock(GenGauss2dPlaneDirectError::Create(val, x, y, radius, radius, 0, 0, 1), nullptr, params.data());
                GenGauss2dPlaneDirectError tmp_model2(val, x, y, radius, radius, 0, 0, 1);
                functors.push_back(tmp_model2);
            }
        }

        ceres::Solver::Options options;
        double const tolerance = 1e-12;
        options.function_tolerance = tolerance;
        options.parameter_tolerance = tolerance;
        options.gradient_tolerance = tolerance;
        options.minimizer_progress_to_stdout = verbose;
        options.max_num_iterations = 1'000;
        options.logging_type = ceres::LoggingType::PER_MINIMIZER_ITERATION;
        options.linear_solver_type = ceres::DENSE_QR;
        //options.preconditioner_type = ceres::IDENTITY;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        if (verbose) {
            std::cout << summary.FullReport() << std::endl;
            std::cout << summary.BriefReport() << std::endl;
            std::cout << "Fit vs. GT parameters:" << std::endl;
            for (size_t ii = 0; ii < 10; ++ii) {
                std::cout << "#" << ii << ":\t" << params[ii] << "\t" << gt_params[ii] << std::endl;
            }
            std::cout << "Max image value: " << max_value << std::endl;
        }
        double sigma_y = abs(params[3])*(1.25+0.75*sin(params[7]));

        double contrast = abs(params[2]-params[4])*exp(-(0.25/(2.0*params[3]*params[3])+0.25/(2.0*params[3]*params[3])));
        contrast = std::min(255.0, contrast);

        rs::QuantileStats<float> signal_stats, noise_stats;
        for (GenGauss2dPlaneDirectError const& functor : functors) {
            double const signal = functor.evaluateModel(params.data());
            signal_stats.push_unsafe(signal);
            double const noise = functor.val_ - signal;
            noise_stats.push_unsafe(noise);
        }

        cv::Point2f size(radius, radius);
        double scale_f =
                (std::max(abs(params[3]),sigma_y) / size.x - min_sigma*0.5)
                /contrast
                *(1.0+tilt_max_rms_penalty*(abs(params[5])+abs(params[6]))/fit_gauss_max_tilt);

        rms = sqrt(summary.final_cost/problem.NumResiduals())*scale_f;
        snr = signal_stats.getAccurateVariance() / noise_stats.getAccurateVariance();
        fit_x = std::sin(params[0])*(double(radius)*subfit_max_range);
        fit_y = std::sin(params[1])*(double(radius)*subfit_max_range);


        diff_x = fit_x - gt_x;
        diff_y = fit_y - gt_y;
        error_length = std::sqrt(diff_x*diff_x + diff_y*diff_y);

        for (double val : {
             fit_x, fit_y, gt_x, gt_y, diff_x, diff_y, error_length, rms, snr, scale_f,
             params[0],
             params[1],
             params[2],
             params[3],
             params[4],
             params[5],
             params[6],
             params[7],
             params[8],
    }) {
            success &= std::isfinite(val);
        }

        success &= rms > 0;
        success &= rms < 100;
        success &= snr > 0;


        if (verbose) {
            std::cout << "RMS: " << rms << std::endl;
            std::cout << "SNR: " << snr << std::endl;
            std::cout << "Signal: " << signal_stats.print() << std::endl;
            std::cout << "Noise: " << noise_stats.print() << std::endl;
            std::cout << "Src stats: " << src_stats.print() << std::endl;
            std::cout << "Fit x, y: " << fit_x << ", " << fit_y << std::endl;
            std::cout << "Localisation error: " << error_length << std::endl;
        }

        if (!img_prefix.empty()) {
            cv::imwrite(img_prefix + "src.png", img);
            cv::Mat_<uint8_t> fitted(img.size());
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    GenGauss2dPlaneDirectError tmp_model(0, x, y, radius, radius, 0, 0, 1);
                    double val = tmp_model.evaluateModel(params.data());
                    fitted(y+radius,x+radius) = cv::saturate_cast<uint8_t>(val);
                    success &= std::isfinite(val);
                }
            }
            cv::imwrite(img_prefix + "fit.png", fitted);
        }

    }
};

template<class T>
void plotWithErrors(
        std::string const& prefix,
        rs::HistConfig const& conf,
        std::map<T, rs::QuantileStats<float> > & stats) {
    gnuplotio::Gnuplot gpl;
    std::stringstream cmd;
    std::string data_file = prefix + ".data";
    std::vector<std::tuple<double, double, double> > values;
    size_t const n = stats.begin()->second.getCount();
    for (auto & it : stats) {
        values.push_back({it.first, it.second.getTrimmedMean(.5), it.second.getStddev()});
    }
    cmd << "set term svg enhanced background rgb \"white\";\n"
        << "set output \"" << prefix + ".svg\"; \n"
        << conf.toString() << ";\n";
    if (conf.title.empty()) {
        cmd << "set title \"n=" << n << "\"; \n";
    }
    fs::copy_file(data_file, data_file + ".old", fs::copy_option::overwrite_if_exists, ignore_error);

    cmd << "plot " << gpl.file(values, data_file) << " u 1:2:3 with yerrorbars notitle; \n";

    gpl << cmd.str();

    std::ofstream out(prefix + ".gpl");
    out << cmd.str();
}

void rms_image() {
    std::cout << "rms_image" << std::endl;
    std::map<size_t, rs::QuantileStats<float> > rms, snr, error;

    for (size_t kk = 0; kk < 10000; ++kk) {
        for (size_t jj = 0; jj < 500; ++jj) {
            for (size_t scale = 10; scale < 256; scale += 10) {
                FitExperiment f;
                f.radius = 4;
                f.verbose = false;
                f.scale = scale;
                f.sigma = 1.5;
                f.noise_sigma = f.scale/20;
                f.runFit();
                rms[scale].push_unsafe(f.rms);
                snr[scale].push_unsafe(f.snr);
                error[scale].push_unsafe(f.error_length);
            }
        }
        rs::HistConfig conf;
        conf.setXLabel("Scale");
        plotWithErrors("scale-rms", conf.setYLabel("RMS"), rms);
        plotWithErrors("scale-snr", conf.setYLabel("SNR"), snr);
        plotWithErrors("scale-error", conf.setYLabel("Error"), error);
        std::cout << "." << std::flush;
    }
    std::cout << std::endl;
}

template<class T>
void plot2D(
        std::string const& prefix,
        rs::HistConfig const& conf,
        rs::Image2D<rs::RunningStats> const& stats) {
    gnuplotio::Gnuplot gpl;
    std::stringstream cmd;
    std::string data_file = prefix + ".data";
    std::vector<std::tuple<double, double, double> > values;
    cmd << "set term svg enhanced background rgb \"white\";\n"
        << "set output \"" << prefix + ".svg\"; \n"
        << conf.toString() << ";\n";
    if (conf.title.empty()) {
        //cmd << "set title \"n=" << n << "\"; \n";
    }
    fs::copy_file(data_file, data_file + ".old", fs::copy_option::overwrite_if_exists);

    cmd << "plot " << gpl.file(values, data_file) << " u 1:2:3 with yerrorbars notitle; \n";

    gpl << cmd.str();

    std::ofstream out(prefix + ".gpl");
    out << cmd.str();
}

void rms_2d() {
    std::cout << "rms_2d" << std::endl;
    double const scale_step = 10;
    double const noise_step = 1;
    rs::Image2D<rs::QuantileStats<float> >
            rms(scale_step, noise_step),
            snr(scale_step, noise_step),
            error(scale_step, noise_step);

    rs::Stats2D<double> rms_vs_error, snr_vs_error;

    size_t n = 0;
    for (size_t kk = 0; kk < 100000; ++kk) {
        for (size_t jj = 0; jj < 50; ++jj) {
            for (double scale = 10; scale < 256; scale += scale_step) {
                for (double src_noise = 1; src_noise <= 15; src_noise += noise_step) {
                    FitExperiment f;
                    f.radius = 4;
                    f.verbose = false;
                    f.scale = scale;
                    f.sigma = 1;
                    f.noise_sigma = src_noise * (f.scale/100.0);
                    f.runFit();
                    if (f.success) {
                        rms[scale][src_noise].push_unsafe(f.rms);
                        snr[scale][src_noise].push_unsafe(f.snr);
                        error[scale][src_noise].push_unsafe(f.error_length);
                        rms_vs_error.push_unsafe(f.rms, f.error_length);
                        snr_vs_error.push_unsafe(f.snr, f.error_length);
                    }
                    n = std::numeric_limits<size_t>::max();
                    n = std::min(n, rms[scale][src_noise].getCount());
                    n = std::min(n, snr[scale][src_noise].getCount());
                    n = std::min(n, error[scale][src_noise].getCount());
                }
            }
        }
        {
            rs::HistConfig conf;
            conf.setXLabel("Scale").setYLabel("Noise std. dev.").setTitle(std::to_string(n)).extractTrimmedMean(.5);
            rms.plot("scale-noise-rms", conf);
            snr.plot("scale-noise-snr", conf);
            error.plot("scale-noise-error", conf);
        }
        {
            rs::HistConfig conf;
            conf.setYLabel("error").setTitle(std::to_string(n));
            rms_vs_error.plotHist("scale-noise-rms-vs-error", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist("scale-noise-snr-vs-error", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
            conf.setLogCB();
            rms_vs_error.plotHist("scale-noise-rms-vs-error-log", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist("scale-noise-snr-vs-error-log", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
        }
        std::cout << "." << std::flush;
    }
    std::cout << std::endl;
}

void dot_size_vs_noise() {
    std::cout << "dot_size_vs_noise" << std::endl;
    double const dotsize_step = 0.125;
    double const noise_step = 1;
    rs::Image2D<rs::QuantileStats<float> >
            rms(dotsize_step, noise_step),
            snr(dotsize_step, noise_step),
            error(dotsize_step, noise_step);

    rs::Stats2D<double> rms_vs_error, snr_vs_error;

    for (size_t kk = 0; kk < 100000; ++kk) {
        size_t n = 0;
        for (size_t jj = 0; jj < 50; ++jj) {
            for (double dot_size = 1; dot_size <= 3; dot_size += dotsize_step) {
                for (double src_noise = 1; src_noise <= 15; src_noise += noise_step) {
                    FitExperiment f;
                    f.radius = 6;
                    f.verbose = false;
                    f.scale = 100;
                    f.sigma = dot_size;
                    f.noise_sigma = src_noise * (f.scale/100.0);
                    f.runFit();
                    if (f.success) {
                        rms[dot_size][src_noise].push_unsafe(f.rms);
                        snr[dot_size][src_noise].push_unsafe(f.snr);
                        error[dot_size][src_noise].push_unsafe(f.error_length);
                        rms_vs_error.push_unsafe(f.rms, f.error_length);
                        snr_vs_error.push_unsafe(f.snr, f.error_length);
                    }
                    n = std::numeric_limits<size_t>::max();
                    n = std::min(n, rms[dot_size][src_noise].getCount());
                    n = std::min(n, snr[dot_size][src_noise].getCount());
                    n = std::min(n, error[dot_size][src_noise].getCount());
                }
            }
        }
        {
            rs::HistConfig conf;
            conf.setXLabel("Dot size").setYLabel("Noise std. dev.").setTitle(std::to_string(n));
            rms.plot("dotsize-noise-rms", conf);
            snr.plot("dotsize-noise-snr", conf);
            error.plot("dotsize-noise-error", conf);
        }
        {
            rs::HistConfig conf;
            conf.setYLabel("error").setTitle(std::to_string(n));
            rms_vs_error.plotHist("dotsize-noise-rms-vs-error", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist("dotsize-noise-snr-vs-error", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
            conf.setLogCB();
            rms_vs_error.plotHist("dotsize-noise-rms-vs-error-log", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist("dotsize-noise-snr-vs-error-log", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
        }
        std::cout << "." << std::flush;
    }
    std::cout << std::endl;
}
static bool save_highres_square = true;
#include <boost/integer/common_factor.hpp>
cv::Mat_<float> renderSquare(int const radius, double const point_scale = 100, bool const invert = false) {
    int target_width = 2*radius+1;
    float const background = invert ? 1.0 : 1.0/6.0;
    float const foreground = invert ? 1.0/6.0 : 1.0;
    int intermediate_width = std::min(5*7*9, boost::integer::lcm(target_width, 5*7*9));
    double square_size = 0.2*(double(intermediate_width) * point_scale) / 100.0;
    square_size = std::round((square_size-1)/2)*2+1;
    cv::Mat_<float> result(intermediate_width, intermediate_width, background);
    result.setTo(background);
    int const left = (intermediate_width/2) - int(square_size)/2;
    cv::Rect dot(left, left, square_size, square_size);
    cv::rectangle(result, dot, cv::Scalar(foreground), cv::FILLED);
    if (save_highres_square) {
        cv::imwrite("highres-square.tif", result);
        save_highres_square = false;
    }
    double const blur_sigma = intermediate_width/20;
    cv::Mat_<float> _result;
    cv::GaussianBlur(result, _result, cv::Size(), blur_sigma, blur_sigma);
    cv::resize(_result, result, cv::Size(target_width, target_width), 0, 0, cv::INTER_LINEAR);
    return result.clone();
}

void addNoise(cv::Mat_<float>& img) {
    static randutils::mt19937_rng rng;
    for (float& val : img) {
        val += rng.variate<double, std::normal_distribution>(0, 0.62*std::sqrt(val));
    }
}

void single_square(int radius, double scale) {
    std::string const prefix = "2dgauss_" + std::to_string(radius) + "_" + std::to_string(int(std::round(scale))) + "_";
    {
        std::cout << "Single square" << std::endl;
        cv::Mat_<float> img = renderSquare(radius, scale, false)*255;
        std::cout << "Rendering done: " << img.size() << std::endl;
        addNoise(img);
        double min = 0;
        double max = 0;
        cv::minMaxIdx(img, &min, &max);
        std::cout << "Min/max: " << min << " / " << max << std::endl;
        FitExperiment f;
        f.verbose = true;
        f.img_prefix = prefix + "black_";
        f.runFitImg(img);
        std::cout << "Success: " << f.success << std::endl;
    }
    {
        std::cout << "Single inverted square" << std::endl;
        cv::Mat_<float> img = renderSquare(radius, scale, true)*255;
        std::cout << "Rendering done: " << img.size() << std::endl;
        addNoise(img);
        double min = 0;
        double max = 0;
        cv::minMaxIdx(img, &min, &max);
        std::cout << "Min/max: " << min << " / " << max << std::endl;
        FitExperiment f;
        f.verbose = true;
        f.img_prefix = prefix + "white_";
        f.runFitImg(img);
        std::cout << "Success: " << f.success << std::endl;
    }
}

void squares(int const radius = 2, bool const noise_only = false) {
    std::cout << "square_size_vs_noise" << std::endl;
    double const dotsize_step = 7;
    double const underexposure_step = .25;
    std::string const width = std::to_string(2*radius+1);
    std::string const prefix = noise_only ? "noise-squaresize-" + width : "squaresize-" + width;

    rs::Image2D<rs::QuantileStats<float> >
            rms(dotsize_step, underexposure_step),
            snr(dotsize_step, underexposure_step),
            snr_times_sigma(dotsize_step, underexposure_step),
            error(dotsize_step, underexposure_step);

    rs::Image2D<std::vector<rs::QuantileStats<float> > > combined_stats(dotsize_step, underexposure_step);

    rs::Stats2D<double> rms_vs_error, snr_vs_error;
    rs::QuantileStats<float> x_res, y_res;

    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d(1), rms_vs_error_1d(.05);
    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d_res(1), rms_vs_error_1d_res(.05);
    rs::BinaryStats success_count;

    rs::Histogram fail_by_exposure(underexposure_step);

    size_t total_count = 0;
    for (size_t kk = 0; kk < 100000; ++kk) {
        size_t n = 0;
        ParallelTime runtime;
        for (size_t jj = 0; jj < 10; ++jj) {
            for (double dot_size = 100; dot_size <= 170; dot_size += dotsize_step) {
                for (double underexposure = 0; underexposure <= 5; underexposure += underexposure_step) {
                    for (bool invert : {true, false}) {
                        FitExperiment f;
                        f.verbose = false;
                        cv::Mat_<float> img = noise_only ?
                                    cv::Mat_<float>(2*radius+1, 2*radius+1, float(invert ? 1.0 : 1.0/6.0))
                                  : renderSquare(radius, dot_size, invert);
                        img *= 4095.0 * std::pow(2.0, -1.0/2.0) * std::pow(2.0, -underexposure);
                        addNoise(img);
                        img *= 255.0 / 4095.0;
                        f.runFitImg(img);
                        success_count.push(f.success);
                        if (f.success) {
                            total_count++;
                            rms[dot_size][underexposure].push_unsafe(f.rms);
                            snr[dot_size][underexposure].push_unsafe(f.snr);
                            error[dot_size][underexposure].push_unsafe(f.error_length);
                            snr_times_sigma[dot_size][underexposure].push_unsafe(f.snr * std::sqrt(f.getFitSigma()));
                            combined_stats.push_unsafe(dot_size,underexposure,
                                                       {
                                                           f.rms, // 3
                                                           f.snr, // 4
                                                           f.error_length, // 5
                                                           f.getFitSigma(), // 6
                                                           f.getFitScale() // 7
                                                       });

                            rms_vs_error.push_unsafe(f.rms, f.error_length);
                            snr_vs_error.push_unsafe(f.snr, f.error_length);
                            x_res.push_unsafe(f.diff_x);
                            y_res.push_unsafe(f.diff_y);
                            snr_vs_error_1d[f.snr].push_unsafe(f.error_length);
                            rms_vs_error_1d[f.rms].push_unsafe(f.error_length);

                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_x);
                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_y);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_x);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_y);
                        }
                        else {
                            fail_by_exposure.push_unsafe(underexposure);
                        }
                        n = std::numeric_limits<size_t>::max();
                        n = std::min(n, rms[dot_size][underexposure].getCount());
                        n = std::min(n, snr[dot_size][underexposure].getCount());
                        n = std::min(n, error[dot_size][underexposure].getCount());
                    }
                }
            }
        }
        runtime.stop();
        ParallelTime plottime;
        std::cout << "Plotting..." << std::endl;
        {
            fail_by_exposure.plotHist(prefix + "-failcount-by-underexposure");

            std::ofstream out(prefix + "-combined.data");
            combined_stats.data2file(out, rs::HistConfig().extractTrimmedMean(.5));
        }
#if 0
        {
            rs::HistConfig conf;
            conf
                    .setYLabel("Localization Error [px]")
                    .extractMeanAndStddev()
                    .setTitle("");
            snr_vs_error_1d.plot(prefix + "-snr-vs-error", conf);
            rms_vs_error_1d.plot(prefix + "-rms-vs-error", conf);

            conf.extractMedianAndIQR();
            snr_vs_error_1d.plot(prefix + "-snr-vs-error-median", conf);
            rms_vs_error_1d.plot(prefix + "-rms-vs-error-median", conf);

            conf.extractQuantile(.9);
            snr_vs_error_1d.plot(prefix + "-snr-vs-error-quantile-.9", conf);
            rms_vs_error_1d.plot(prefix + "-rms-vs-error-quantile-.9", conf);
        }
        {
            rs::HistConfig conf;
            conf
                    .setYLabel("Localization Residual [px]")
                    .extractMeanAndStddev()
                    .setTitle("");
            snr_vs_error_1d_res.plot(prefix + "-snr-vs-res", conf);
            rms_vs_error_1d_res.plot(prefix + "-rms-vs-res", conf);

            conf.extractMedianAndIQR();
            snr_vs_error_1d_res.plot(prefix + "-snr-vs-res-median", conf);
            rms_vs_error_1d_res.plot(prefix + "-rms-vs-res-median", conf);
        }
        {
            rs::HistConfig conf;
            conf.setYLabel("error").setTitle(std::to_string(n));
            rms_vs_error.plotHist(prefix + "-noise-rms-vs-error", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist(prefix + "-noise-snr-vs-error", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
            conf.setLogCB();
            rms_vs_error.plotHist(prefix + "-noise-rms-vs-error-log", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist(prefix + "-noise-snr-vs-error-log", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
        }
#endif
        {
            rs::HistConfig conf;
            conf
                    .setXLabel("Square width [\\\\%]")
                    .setYLabel("Underexposure [stops]")
                    .setTitle(std::to_string(n))
                    .extractTrimmedMean();
            rms.plot(prefix + "-noise-rms", conf);
            snr.plot(prefix + "-noise-snr", conf);
            error.plot(prefix + "-noise-error", conf);
            snr_times_sigma.plot(prefix + "-noise-snr_sigma", conf);
        }
        {
            std::cout << "x residuals: " << x_res.print() << std::endl;
            std::cout << "y residuals: " << y_res.print() << std::endl;
        }
        std::cout << "Success stats: " << success_count.print() << std::endl;
        std::cout << "Total count: " << total_count << std::endl;
        std::cout << "Per-pixel count: " << n << std::endl;
        std::cout << "Runtime: " << runtime.print() << std::endl
                  << "Plot time: " << plottime.print() << std::endl;
        std::cout << std::endl << std::endl;
    }
    std::cout << std::endl;
}

void exposure(size_t const radius) {
    std::cout << "square_size_vs_noise" << std::endl;
    double const dotsize_step = 30;
    double const underexposure_step = 1;
    std::string const width = std::to_string(2*radius+1);

    rs::Image2D<rs::QuantileStats<float> >
            rms(dotsize_step, underexposure_step),
            snr(dotsize_step, underexposure_step),
            snr_times_sigma(dotsize_step, underexposure_step);

    std::map<double, std::map<double, rs::QuantileStats<double> > > error;

    rs::Image2D<std::vector<rs::QuantileStats<float> > > combined_stats(dotsize_step, underexposure_step);

    rs::Stats2D<double> rms_vs_error, snr_vs_error;
    rs::QuantileStats<float> x_res, y_res;

    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d(1), rms_vs_error_1d(.05);
    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d_res(1), rms_vs_error_1d_res(.05);
    rs::BinaryStats success_count;

    rs::Histogram fail_by_exposure(underexposure_step);

    size_t total_count = 0;
    for (size_t kk = 0; kk < 100'000; ++kk) {
        size_t n = 0;
        ParallelTime runtime;
        for (size_t jj = 0; jj < 100; ++jj) {
            for (double dot_size = 100; dot_size <= 170; dot_size += dotsize_step) {
                for (double underexposure = 0; underexposure <= 5; underexposure += underexposure_step) {
                    for (bool invert : {true, false}) {
                        FitExperiment f;
                        f.verbose = false;
                        cv::Mat_<float> img = renderSquare(radius, dot_size, invert);
                        img *= 4095.0 * std::pow(2.0, -1.0/2.0) * std::pow(2.0, -underexposure);
                        addNoise(img);
                        img *= 255.0 / 4095.0;
                        f.runFitImg(img);
                        success_count.push(f.success);
                        if (f.success) {
                            total_count++;
                            error[dot_size][underexposure].push_unsafe(f.error_length);
                            /*
                            rms[dot_size][underexposure].push_unsafe(f.rms);
                            snr[dot_size][underexposure].push_unsafe(f.snr);
                            snr_times_sigma[dot_size][underexposure].push_unsafe(f.snr * std::sqrt(f.getFitSigma()));
                            combined_stats.push_unsafe(dot_size,underexposure,
                                                       {
                                                           f.rms, // 3
                                                           f.snr, // 4
                                                           f.error_length, // 5
                                                           f.getFitSigma(), // 6
                                                           f.getFitScale() // 7
                                                       });

                            rms_vs_error.push_unsafe(f.rms, f.error_length);
                            snr_vs_error.push_unsafe(f.snr, f.error_length);
                            x_res.push_unsafe(f.diff_x);
                            y_res.push_unsafe(f.diff_y);
                            snr_vs_error_1d[f.snr].push_unsafe(f.error_length);
                            rms_vs_error_1d[f.rms].push_unsafe(f.error_length);

                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_x);
                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_y);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_x);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_y);
                            */
                        }
                        else {
                            fail_by_exposure.push_unsafe(underexposure);
                        }
                        n = error[dot_size][underexposure].getCount();
                    }
                }
            }
        }
        runtime.stop();
        ParallelTime plottime;
        std::cout << "Plotting " << width << "..." << std::endl;

        for (auto& it1 : error) {
            double const dotsize = it1.first;
            std::cout << "Dot size: " << dotsize << std::endl;
            for (auto& it2 : it1.second) {
                double const under_exp = it2.first;
                std::cout << under_exp << ": " << it2.second.getMedian();
                auto const next_exp = it1.second.find(under_exp + underexposure_step);
                if (next_exp == it1.second.end()) {
                    std::cout << ", ...";
                }
                else {
                    std::cout << ", " << (1.0-it2.second.getMedian()/next_exp->second.getMedian())*100;
                }
                auto const prev_size = error.find(dotsize - dotsize_step);
                if (prev_size != error.end()) {
                    std::cout << ", " << (1.0 - it2.second.getMedian()/prev_size->second[under_exp].getMedian())*100;
                }
                std::cout << std::endl;
            }
        }

        std::cout << "Success stats: " << success_count.print() << std::endl;
        std::cout << "Total count: " << total_count << std::endl;
        std::cout << "Per-pixel count: " << n << std::endl;
        std::cout << "Runtime: " << runtime.print() << std::endl
                  << "Plot time: " << plottime.print() << std::endl;
        std::cout << std::endl << std::endl;
    }
    std::cout << std::endl;
}

void addGradient(cv::Mat_<float>& img, double const gradient, double const angle = 45) {
    double const sin = std::sin(angle*M_PI/180);
    double const cos = std::cos(angle*M_PI/180);

    int const rows = img.rows;
    int const cols = img.cols;
    if (rows < 2 || cols < 2) {
        throw std::runtime_error("rows or cols < 2");
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            img(row, col) += gradient*(row*sin + col*cos)/(rows-1);
        }
    }
}

template<class T>
void plotTable(T & table, double const a_step, double const b_step) {
    for (auto& it1 : table) {
        double const dotsize = it1.first;
        std::cout << "Dot size: " << dotsize << std::endl;
        for (auto& it2 : it1.second) {
            double const under_exp = it2.first;
            std::cout << std::setw(7) << under_exp << " &" << std::setw(14) << 1000*std::abs(it2.second.getMedian());
            auto const next_exp = it1.second.find(under_exp + b_step);
            if (next_exp == it1.second.end()) {
                std::cout << " &" << std::setw(14) << "...";
            }
            else {
                std::cout << " &" << std::setw(14) << (1.0-std::abs(it2.second.getMedian()/next_exp->second.getMedian()))*100;
            }
            auto const prev_size = table.find(dotsize - a_step);
            if (prev_size != table.end()) {
                std::cout << " &" << std::setw(14) << (1.0 - std::abs(it2.second.getMedian())/std::abs(prev_size->second[under_exp].getMedian()))*100;
            }
            std::cout << " \\\\" << std::endl;
        }
    }
    std::cout << std::endl;

}

void backgroundGradient(size_t const radius) {
    std::cout << "backgroundGradient" << std::endl;
    double const dotsize_step = 30;
    double const gradient_step = .2;
    std::string const width = std::to_string(2*radius+1);

    rs::Image2D<rs::QuantileStats<float> >
            rms(dotsize_step, gradient_step),
            snr(dotsize_step, gradient_step),
            snr_times_sigma(dotsize_step, gradient_step);

    std::map<double, std::map<double, rs::QuantileStats<double> > > error;
    std::map<double, std::map<double, rs::QuantileStats<double> > > bias_x;
    std::map<double, std::map<double, rs::QuantileStats<double> > > bias_y;

    rs::Image2D<std::vector<rs::QuantileStats<float> > > combined_stats(dotsize_step, gradient_step);

    rs::Stats2D<double> rms_vs_error, snr_vs_error;
    rs::QuantileStats<float> x_res, y_res;

    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d(1), rms_vs_error_1d(.05);
    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d_res(1), rms_vs_error_1d_res(.05);
    rs::BinaryStats success_count;

    rs::Histogram fail_by_exposure(gradient_step);

    size_t total_count = 0;
    for (size_t kk = 0; kk < 100'000; ++kk) {
        size_t n = 0;
        ParallelTime runtime;
        for (size_t jj = 0; jj < 100; ++jj) {
            for (double dot_size = 100; dot_size <= 170; dot_size += dotsize_step) {
                for (double gradient = 0; gradient <= 1; gradient += gradient_step) {
                    for (bool invert : {true, false}) {
                        FitExperiment f;
                        f.verbose = false;
                        cv::Mat_<float> img = renderSquare(radius, dot_size, invert);
                        addGradient(img, gradient);
                        img *= 4095.0 * std::pow(2.0, -1.0/2.0);
                        addNoise(img);
                        img *= 255.0 / 4095.0;
                        f.runFitImg(img);
                        success_count.push(f.success);
                        if (f.success) {
                            total_count++;
                            error[dot_size][gradient].push_unsafe(f.error_length);
                            bias_x[dot_size][gradient].push_unsafe((f.diff_x));
                            bias_y[dot_size][gradient].push_unsafe((f.diff_y));
                            /*
                            rms[dot_size][underexposure].push_unsafe(f.rms);
                            snr[dot_size][underexposure].push_unsafe(f.snr);
                            snr_times_sigma[dot_size][underexposure].push_unsafe(f.snr * std::sqrt(f.getFitSigma()));
                            combined_stats.push_unsafe(dot_size,underexposure,
                                                       {
                                                           f.rms, // 3
                                                           f.snr, // 4
                                                           f.error_length, // 5
                                                           f.getFitSigma(), // 6
                                                           f.getFitScale() // 7
                                                       });

                            rms_vs_error.push_unsafe(f.rms, f.error_length);
                            snr_vs_error.push_unsafe(f.snr, f.error_length);
                            x_res.push_unsafe(f.diff_x);
                            y_res.push_unsafe(f.diff_y);
                            snr_vs_error_1d[f.snr].push_unsafe(f.error_length);
                            rms_vs_error_1d[f.rms].push_unsafe(f.error_length);

                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_x);
                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_y);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_x);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_y);
                            */
                        }
                        else {
                            fail_by_exposure.push_unsafe(gradient);
                        }
                        n = error[dot_size][gradient].getCount();
                    }
                }
            }
        }
        runtime.stop();

        ParallelTime plottime;
        std::cout << "Plotting " << width << "..." << std::endl;
        plotTable(error, dotsize_step, gradient_step);
        std::cout << "Bias x, " << width << "..." << std::endl;
        plotTable(bias_x, dotsize_step, gradient_step);
        std::cout << "Bias y, " << width << "..." << std::endl;
        plotTable(bias_y, dotsize_step, gradient_step);

        std::cout << "Success stats: " << success_count.print() << std::endl;
        std::cout << "Total count: " << total_count << std::endl;
        std::cout << "Per-pixel count: " << n << std::endl;
        std::cout << "Runtime: " << runtime.print() << std::endl
                  << "Plot time: " << plottime.print() << std::endl;
        std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
    }
    std::cout << std::endl;
}

void backgroundGradientByAngle(size_t const radius) {
    std::cout << "backgroundGradientByAngle" << std::endl;
    double const dotsize_step = 30;
    double const angle_step = 9;
    std::string const width = std::to_string(2*radius+1);

    rs::Image2D<rs::QuantileStats<float> >
            rms(dotsize_step, angle_step),
            snr(dotsize_step, angle_step),
            snr_times_sigma(dotsize_step, angle_step);

    std::map<double, std::map<double, rs::QuantileStats<double> > > error;
    std::map<double, std::map<double, rs::QuantileStats<double> > > bias_x;
    std::map<double, std::map<double, rs::QuantileStats<double> > > bias_y;

    rs::Image2D<std::vector<rs::QuantileStats<float> > > combined_stats(dotsize_step, angle_step);

    rs::Stats2D<double> rms_vs_error, snr_vs_error;
    rs::QuantileStats<float> x_res, y_res;

    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d(1), rms_vs_error_1d(.05);
    rs::Image1D<rs::QuantileStats<float> > snr_vs_error_1d_res(1), rms_vs_error_1d_res(.05);
    rs::BinaryStats success_count;

    rs::Histogram fail_by_exposure(angle_step);

    size_t total_count = 0;
    for (size_t kk = 0; kk < 100'000; ++kk) {
        size_t n = 0;
        ParallelTime runtime;
        for (size_t jj = 0; jj < 100; ++jj) {
            for (double dot_size = 100; dot_size <= 170; dot_size += dotsize_step) {
                for (double angle = 0; angle <= 45; angle += angle_step) {
                    for (bool invert : {true, false}) {
                        FitExperiment f;
                        f.verbose = false;
                        cv::Mat_<float> img = renderSquare(radius, dot_size, invert);
                        addGradient(img, 1.0, angle);
                        img *= 4095.0 * std::pow(2.0, -1.0/2.0);
                        addNoise(img);
                        img *= 255.0 / 4095.0;
                        f.runFitImg(img);
                        success_count.push(f.success);
                        if (f.success) {
                            total_count++;
                            error[dot_size][angle].push_unsafe(f.error_length);
                            bias_x[dot_size][angle].push_unsafe((f.diff_x));
                            bias_y[dot_size][angle].push_unsafe((f.diff_y));
                            /*
                            rms[dot_size][underexposure].push_unsafe(f.rms);
                            snr[dot_size][underexposure].push_unsafe(f.snr);
                            snr_times_sigma[dot_size][underexposure].push_unsafe(f.snr * std::sqrt(f.getFitSigma()));
                            combined_stats.push_unsafe(dot_size,underexposure,
                                                       {
                                                           f.rms, // 3
                                                           f.snr, // 4
                                                           f.error_length, // 5
                                                           f.getFitSigma(), // 6
                                                           f.getFitScale() // 7
                                                       });

                            rms_vs_error.push_unsafe(f.rms, f.error_length);
                            snr_vs_error.push_unsafe(f.snr, f.error_length);
                            x_res.push_unsafe(f.diff_x);
                            y_res.push_unsafe(f.diff_y);
                            snr_vs_error_1d[f.snr].push_unsafe(f.error_length);
                            rms_vs_error_1d[f.rms].push_unsafe(f.error_length);

                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_x);
                            snr_vs_error_1d_res[f.snr].push_unsafe(f.diff_y);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_x);
                            rms_vs_error_1d_res[f.rms].push_unsafe(f.diff_y);
                            */
                        }
                        else {
                            fail_by_exposure.push_unsafe(angle);
                        }
                        n = error[dot_size][angle].getCount();
                    }
                }
            }
        }
        runtime.stop();

        ParallelTime plottime;
        std::cout << "Plotting " << width << "..." << std::endl;
        plotTable(error, dotsize_step, angle_step);
        std::cout << "Bias x, " << width << "..." << std::endl;
        plotTable(bias_x, dotsize_step, angle_step);
        std::cout << "Bias y, " << width << "..." << std::endl;
        plotTable(bias_y, dotsize_step, angle_step);

        std::cout << "Success stats: " << success_count.print() << std::endl;
        std::cout << "Total count: " << total_count << std::endl;
        std::cout << "Per-pixel count: " << n << std::endl;
        std::cout << "Runtime: " << runtime.print() << std::endl
                  << "Plot time: " << plottime.print() << std::endl;
        std::cout << std::endl << std::endl << std::endl << std::endl << std::endl << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char ** argv) {

    for (int radius = 2; radius <= 19; ++radius) {
        double dot_size = 100;
        bool invert = false;
        double gradient = 1;
        cv::Mat_<float> img = renderSquare(radius, dot_size, invert);
        addGradient(img, gradient);
        img *= 4095.0 * std::pow(2.0, -1.0/2.0);
        addNoise(img);
        img *= 255.0 / 4095.0;
        cv::imwrite(std::string("gradient-1-radius-") + std::to_string(radius) + ".png", img);
    }


    if (argc < 2) {
        FitExperiment f;
        f.verbose = true;
        f.scale = 250;
        f.sigma = 1;
        f.noise_sigma = f.scale/20;
        f.radius = 4;
        f.img_prefix = "2dgauss_";
        f.runFit();

        for (double sigma = 1; sigma <= 3; sigma += .25) {
            FitExperiment f;
            f.verbose = false;
            f.scale = 250;
            f.sigma = sigma;
            f.noise_sigma = f.scale/20;
            f.radius = 6;
            f.img_prefix = "2dgauss_" + std::to_string(sigma) + "_";
            f.runFit();
        }

        return EXIT_SUCCESS;
    }
    std::string const action = argv[1];

    int scale = 2;
    if (argc > 2) {
        scale = std::stoi(argv[2]);
    }

    if ("1D" == action) {
        rms_image();
    }

    if ("2D" == action) {
        rms_2d();
    }
    if ("dotsize-vs-noise" == action) {
        dot_size_vs_noise();
    }
    if ("squares" == action) {
        squares(scale);
    }
    if ("noise-squares" == action) {
        squares(scale, true);
    }
    if ("single-square" == action) {
        for (size_t radius = 2; radius <= 5; ++radius) {
            for (double scale = 100; scale <= 160; scale += 30) {
                single_square(radius, scale);
            }
        }
    }
    if ("exposure" == action) {
        exposure(std::round(scale));
    }
    if ("bg-grad" == action) {
        backgroundGradient(scale);
    }
    if ("bg-grad-angle" == action) {
        backgroundGradientByAngle(scale);
    }
}
