#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/rational.hpp>

namespace fs = boost::filesystem;

#include <hdmarker/hdmarker.hpp>
#include <hdmarker/subpattern.hpp>

#include <ceres/ceres.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "gnuplot-iostream.h"

#include <runningstats/runningstats.h>

using namespace hdmarker;

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

    void runFitImg(cv::Mat_<float> const& img) {
        ceres::Problem problem;
        radius = double(img.rows-1)/2;
        double max_img = 0;
        sigma = radius/2;
        success = true;
        runningstats::QuantileStats<float> src_stats;
        for (int xx = 0; xx <= img.rows; ++xx) {
            for (int yy = 0; yy <= img.cols; ++yy) {
                double const val = img(yy, xx);
                max_img = std::max(max_img, val);
                src_stats.push_unsafe(val);
                problem.AddResidualBlock(GenGauss2dPlaneDirectError::Create(val, double(xx) - radius, double(yy) - radius, 2*radius, 2*radius, 0, 0, 1), nullptr, params.data());
            }
        }
        scale = max_img;
        params = {0,0,
                  scale,sigma,0,0,0,0,0,0};

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
            std::cout << "Max image value: " << max_img << std::endl;
        }
        double sigma_y = abs(params[3])*(1.25+0.75*sin(params[7]));

        double contrast = abs(params[2]-params[4])*exp(-(0.25/(2.0*params[3]*params[3])+0.25/(2.0*params[3]*params[3])));
        contrast = std::min(255.0, contrast);

        runningstats::QuantileStats<float> signal_stats, noise_stats;
        /*
        for (GenGauss2dPlaneDirectError const& functor : functors) {
            double const signal = functor.evaluateModel(params.data());
            signal_stats.push_unsafe(signal);
            double const noise = functor.val_ - signal;
            noise_stats.push_unsafe(noise);
        }
        // */
        cv::Mat_<uint8_t> fit_img(img.size(), uint8_t(0));
        for (int xx = 0; xx <= img.rows; ++xx) {
            for (int yy = 0; yy <= img.cols; ++yy) {
                double const val = img(yy, xx);
                GenGauss2dPlaneDirectError model(val, double(xx) - radius, double(yy) - radius, 2*radius, 2*radius, 0, 0, 1);
                double const signal = model.evaluateModel(params.data());
                signal_stats.push_unsafe(signal);
                double const noise = val - signal;
                noise_stats.push_unsafe(noise);
                fit_img(yy,xx) = cv::saturate_cast<uint8_t>(signal);
            }
        }
        if (!img_prefix.empty()) {
            cv::imwrite(img_prefix + "src.png", img);
            cv::imwrite(img_prefix + "fit.png", fit_img);
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

        runningstats::QuantileStats<float> src_stats, fit_stats;

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

        runningstats::QuantileStats<float> signal_stats, noise_stats;
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
        runningstats::HistConfig const& conf,
        std::map<T, runningstats::QuantileStats<float> > & stats) {
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
    std::map<size_t, runningstats::QuantileStats<float> > rms, snr, error;

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
        runningstats::HistConfig conf;
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
        runningstats::HistConfig const& conf,
        runningstats::Image2D<runningstats::RunningStats> const& stats) {
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
    runningstats::Image2D<runningstats::QuantileStats<float> >
            rms(scale_step, noise_step),
            snr(scale_step, noise_step),
            error(scale_step, noise_step);

    runningstats::Stats2D<double> rms_vs_error, snr_vs_error;

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
            runningstats::HistConfig conf;
            conf.setXLabel("Scale").setYLabel("Noise std. dev.").setTitle(std::to_string(n)).extractTrimmedMean(.5);
            rms.plot("scale-noise-rms", conf);
            snr.plot("scale-noise-snr", conf);
            error.plot("scale-noise-error", conf);
        }
        {
            runningstats::HistConfig conf;
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
    runningstats::Image2D<runningstats::QuantileStats<float> >
            rms(dotsize_step, noise_step),
            snr(dotsize_step, noise_step),
            error(dotsize_step, noise_step);

    runningstats::Stats2D<double> rms_vs_error, snr_vs_error;

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
            runningstats::HistConfig conf;
            conf.setXLabel("Dot size").setYLabel("Noise std. dev.").setTitle(std::to_string(n));
            rms.plot("dotsize-noise-rms", conf);
            snr.plot("dotsize-noise-snr", conf);
            error.plot("dotsize-noise-error", conf);
        }
        {
            runningstats::HistConfig conf;
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
#include <boost/integer/common_factor.hpp>
cv::Mat_<float> renderSquare(int radius, double point_scale = 1) {
    int target_width = 2*radius+1;
    int intermediate_width = boost::integer::lcm(target_width, 5*7*9);
    double scale = intermediate_width / target_width;
    double square_size = (double(radius)/2) * scale * point_scale;
    square_size = std::round((square_size-1)/2)*2+1;
    cv::Mat_<float> result(intermediate_width, intermediate_width, float(0));
    int const left = (intermediate_width/2) - int(square_size)/2;
    cv::Rect dot(left, left, square_size, square_size);
    cv::rectangle(result, dot, cv::Scalar(255), cv::FILLED);
    double const blur_sigma = scale/5;
    cv::GaussianBlur(result, result, cv::Size(), blur_sigma, blur_sigma);
    cv::resize(result, result, cv::Size(target_width, target_width), 0, 0, cv::INTER_AREA);
    return result;
}

void addNoise(cv::Mat_<float>& img, double stddev) {
    static randutils::mt19937_rng rng;
    for (float& val : img) {
        val += rng.variate<double, std::normal_distribution>(0, stddev);
    }
}

void single_square() {
    cv::Mat_<float> img = renderSquare(3, 1);
    cv::Mat_<float> rand(img.size(), float(0));
    addNoise(img, 3);
    double min = 0;
    double max = 0;
    cv::minMaxIdx(img, &min, &max);
    std::cout << "Min/max: " << min << " / " << max << std::endl;
    FitExperiment f;
    f.verbose = true;
    f.img_prefix = "2dgauss_";
    f.runFitImg(img);
    std::cout << "Success: " << f.success << std::endl;
}

void squares(int radius = 2) {
    std::cout << "square_size_vs_noise" << std::endl;
    double const dotsize_step = 0.05;
    double const noise_step = 1;
    std::string const width = std::to_string(2*radius+1);
    runningstats::Image2D<runningstats::QuantileStats<float> >
            rms(dotsize_step, noise_step),
            snr(dotsize_step, noise_step),
            error(dotsize_step, noise_step);

    runningstats::Stats2D<double> rms_vs_error, snr_vs_error;
    runningstats::QuantileStats<float> x_res, y_res;

    for (size_t kk = 0; kk < 100000; ++kk) {
        size_t n = 0;
        for (size_t jj = 0; jj < 5; ++jj) {
            for (double dot_size = 1; dot_size <= 1.66; dot_size += dotsize_step) {
                for (double src_noise = 1; src_noise <= 15; src_noise += noise_step) {
                    FitExperiment f;
                    f.verbose = false;
                    cv::Mat_<float> img = renderSquare(radius, dot_size);
                    img *= 100.0/255.0;
                    addNoise(img, src_noise);
                    f.runFitImg(img);
                    if (f.success) {
                        rms[dot_size][src_noise].push_unsafe(f.rms);
                        snr[dot_size][src_noise].push_unsafe(f.snr);
                        error[dot_size][src_noise].push_unsafe(f.error_length);
                        rms_vs_error.push_unsafe(f.rms, f.error_length);
                        snr_vs_error.push_unsafe(f.snr, f.error_length);
                        x_res.push_unsafe(f.diff_x);
                        y_res.push_unsafe(f.diff_y);
                    }
                    n = std::numeric_limits<size_t>::max();
                    n = std::min(n, rms[dot_size][src_noise].getCount());
                    n = std::min(n, snr[dot_size][src_noise].getCount());
                    n = std::min(n, error[dot_size][src_noise].getCount());
                }
            }
        }
        std::cout << "Plotting..." << std::flush;
        {
            runningstats::HistConfig conf;
            conf
                    .setXLabel("Square scale factor")
                    .setYLabel("Noise std. dev.")
                    .setTitle(std::to_string(n))
                    .extractTrimmedMean();
            rms.plot("squaresize-" + width + "-noise-rms", conf);
            snr.plot("squaresize-" + width + "-noise-snr", conf);
            error.plot("squaresize-" + width + "-noise-error", conf);
        }
        {
            runningstats::HistConfig conf;
            conf.setYLabel("error").setTitle(std::to_string(n));
            rms_vs_error.plotHist("squaresize-" + width + "-noise-rms-vs-error", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist("squaresize-" + width + "-noise-snr-vs-error", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
            conf.setLogCB();
            rms_vs_error.plotHist("squaresize-" + width + "-noise-rms-vs-error-log", rms_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("RMS"));
            snr_vs_error.plotHist("squaresize-" + width + "-noise-snr-vs-error-log", snr_vs_error.FreedmanDiaconisBinSize(), conf.setXLabel("SNR"));
        }
        {
            std::cout << "x residuals: " << x_res.print() << std::endl;
            std::cout << "y residuals: " << y_res.print() << std::endl;
        }
        std::cout << "done." << std::endl;
    }
    std::cout << std::endl;
}


int main(int argc, char ** argv) {



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
        single_square();
        squares();
    }
    if ("single-square" == action) {
        single_square();
    }
}
