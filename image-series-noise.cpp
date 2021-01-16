#include <iostream>
#include <fstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <runningstats/runningstats.h>

#include "randutils.hpp"

cv::Mat_<float> imread_12(std::string const& filename) {
    return cv::Mat_<float>(cv::imread(filename, cv::IMREAD_UNCHANGED))/16;
}

size_t stochastic_round(double const val) {
    static randutils::mt19937_rng rng;
    return std::round(val + rng.variate<double, std::uniform_real_distribution>(-.5,.5));

    size_t const rounded = std::round(val);
    double const diff = std::abs(double(rounded) - val);
    if (std::abs(diff - 0.5) < 0.1) {
        return std::floor(val) + rng.pick({size_t(0), size_t(1)});
    }
    return rounded;
}

int main(int argc, char ** argv) {
    /*
    randutils::mt19937_rng rng;
    runningstats::QuantileStats<float> test_stats;
    for (size_t ii = 0; ii < 10'000'000; ++ii) {
        test_stats.push_unsafe(M_SQRT1_2*(rng.variate<double, std::normal_distribution>(0, 1) - rng.variate<double, std::normal_distribution>(0, 1)));
    }
    std::cout << "Test stats: " << test_stats.print() << std::endl;
    */

    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " output-prefix image1 image2 ... image_n" << std::endl;
        return EXIT_FAILURE;
    }
    //runningstats::Stats2D<float> diff_by_mean;
    cv::Mat_<float> img1 = imread_12(argv[2]);
    runningstats::RunningStats rel_noise_stats, abs_noise_stats, mean_diff_stats, mean_img_stats;
    std::vector<runningstats::RunningStats> diff_by_value(4096);
    mean_img_stats.push_unsafe(cv::mean(img1)[0]);
    runningstats::BinaryStats stochastic_round_stats;
    runningstats::Histogram round_error(0.01);
    for (size_t ii = 3; ii < size_t(argc); ++ii) {
        runningstats::RunningStats mean_stats, diff_stats;
        cv::Mat_<float> const _img2 = imread_12(argv[ii]);
        if (img1.size() != _img2.size()) {
            std::cout << "Error: image sizes don't match: " << img1.size() << " vs. " << _img2.size() << std::endl;
            continue;
        }
        double const mean1 = cv::mean(img1)[0];
        double const mean2 = cv::mean(_img2)[0];
        if (mean1 / mean2 < 1.2 && mean2 / mean1 < 1.2) {

            mean_img_stats.push_unsafe(mean2);
            cv::Mat_<float> img2 = (mean1 / mean2) * _img2.clone();
            for (int ii = 0; ii < img1.rows; ++ii) {
                for (int jj = 0; jj < img1.cols; ++jj) {
                    float const diff = M_SQRT1_2 * (img1(ii, jj) - img2(ii, jj));
                    float const mean = 0.5 * (img1(ii, jj) + img2(ii, jj));
                    mean_stats.push_unsafe(mean);
                    diff_stats.push_unsafe(diff);
                    //diff_by_mean.push_unsafe(diff, mean);


                    size_t index_old = std::round(mean);
                    size_t const index = stochastic_round(mean);
                    stochastic_round_stats.push(index_old == index);
                    if (index >= diff_by_value.size()) {
                        diff_by_value.resize(index+1);
                    }
                    diff_by_value[index].push_unsafe(diff);
                }
            }
            double const rel_noise = 100.0 * diff_stats.getStddev() / mean_stats.getMean();
            rel_noise_stats.push_unsafe(rel_noise);
            abs_noise_stats.push_unsafe(diff_stats.getStddev());
            double const mean_diff = 100.0 * (mean1/mean2 - 1.0);
            mean_diff_stats.push_unsafe(mean_diff);
            std::cout << "pair # " << ii-2 << ":" << std::endl
                      << "mean: " << mean_stats.print() << std::endl
                      << "diff: " << diff_stats.print() << std::endl
                      << "diff_of_means_%: " << mean_diff << std::endl
                      << "Relative_noise_stddev%: " << rel_noise << std::endl
                      << std::endl;
        }

        img1 = _img2.clone();
    }
    std::cout << "stochastic round stats: " << stochastic_round_stats.getPercent() << "% identical" << std::endl;
    /*
    diff_by_mean.plotHist(std::string(argv[1]) + "-diff-by-mean", diff_by_mean.FreedmanDiaconisBinSize(),
            runningstats::HistConfig().setLogCB().setXLabel("diff").setYLabel("mean").setMinMaxX(-3000, 3000));
    */
    /*
    rel_noise_stats.plotHistAndCDF(std::string(argv[1]) + "-rel-noise", rel_noise_stats.FreedmanDiaconisBinSize(),
            runningstats::HistConfig().setDataLabel("Relative Noise [%]"));

    mean_diff_stats.plotHistAndCDF(std::string(argv[1]) + "-mean-diff", mean_diff_stats.FreedmanDiaconisBinSize(),
            runningstats::HistConfig().setDataLabel("Difference of Mean Values [%]"));
    */
    {
        std::ofstream out(std::string(argv[1]) + "-rel-noise-stats.txt");
        out << "# 1. mean 2. stddev" << std::endl
            << rel_noise_stats.getMean() << "\t" << rel_noise_stats.getStddev() << std::endl;
    }
    {
        std::ofstream out(std::string(argv[1]) + "-mean-diff-stats.txt");
        out << "# 1. mean 2. stddev" << std::endl
            << mean_diff_stats.getMean() << "\t" << mean_diff_stats.getStddev() << std::endl;
    }
    {
        std::ofstream out(std::string(argv[1]) + "-stats-by-value.txt");
        out << "# 1. pixel value " << std::endl
            << "# 2. Mean diff for that pixel value " << std::endl
            << "# 3. Stddev for that pixel value " << std::endl
            << "# 4. count for that value" << std::endl;
        for (size_t ii = 0; ii < diff_by_value.size(); ++ii) {
            out << ii << "\t"
                << diff_by_value[ii].getMean() << "\t"
                << diff_by_value[ii].getStddev() << "\t"
                << diff_by_value[ii].getCount() << std::endl;
        }
    }
    {
        std::ofstream out(std::string(argv[1]) + "-all-stats.txt");
        out << "# 1. mean image value" << std::endl
            << "# 2-3 mean diff [%] mean and stddev" << std::endl
            << "# 4-5 abs. noise mean and stddev" << std::endl
            << "# 6-7 rel. noise mean and stddev" << std::endl
            << mean_img_stats.getMean() << "\t"
            << mean_diff_stats.getMean() << "\t"
            << mean_diff_stats.getStddev() << "\t"
            << abs_noise_stats.getMean() << "\t"
            << abs_noise_stats.getStddev() << "\t"
            << rel_noise_stats.getMean() << "\t"
            << rel_noise_stats.getStddev() << "\t"
            << std::endl;
    }
    return EXIT_SUCCESS;
}
