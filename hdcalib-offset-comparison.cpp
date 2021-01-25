#undef NDEBUG
#include <cassert>

#include <iostream>
#include <vector>

#include <tclap/CmdLine.h>
#include <ParallelTime/paralleltime.h>
#include <boost/filesystem.hpp>

#include "hdcalib.h"
#include "cornercolor.h"

#include "gnuplot-iostream.h"

#include <regex>

namespace fs = boost::filesystem;

namespace rs = runningstats;

namespace hdm = hdmarker;

class CornerCache {
public:
    static CornerCache& getInstance() {
        static CornerCache instance;
        return instance;
    }

    hdcalib::CornerStore & operator[](std::string const& filename) {
        {
            std::lock_guard<std::mutex> guard(access_mutex);
            std::lock_guard<std::mutex> guard2(access_mutex_by_file[filename]);
            std::map<std::string, hdcalib::CornerStore>::iterator it = data.find(filename);
            if (it != data.end()) {
                return it->second;
            }
            data[filename];
        }
        std::lock_guard<std::mutex> guard(access_mutex_by_file[filename]);
        hdcalib::Calib c;
        ParallelTime t;
        std::vector<hdm::Corner> corners = c.getSubMarkers(filename);
        std::cout << "Reading " << filename << ": " << t.print() << std::endl;
        t.start();
        data[filename] = corners;
        std::cout << "Creating CornerStore for " << filename << ": " << t.print() << std::endl;
        return data[filename];
    }

private:
    std::mutex access_mutex;
    std::map<std::string, std::mutex> access_mutex_by_file;
    CornerCache() {}
    CornerCache(CornerCache const&) = delete;
    void operator=(CornerCache const&) = delete;

    std::map<std::string, hdcalib::CornerStore> data;
};

boost::system::error_code ignore_error_code;

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

bool stringEndsWith(std::string const& str, std::string const& search) {
    if (search.size() > str.size()) {
        return false;
    }
    return str.substr(str.size() - search.size()) == search;
}

void plotOffsets(
        std::set<std::string> & suffixes,
        rs::Stats2D<float> &local_stat_2d,
        rs::Stats2D<float> local_stat_2d_color[3][4],
std::string const& result_prefix
) {

    rs::HistConfig conf2d;
    conf2d.setMinMaxX(-2,2)
            .setMinMaxY(-2,2)
            .setXLabel("x offset [px]")
            .setYLabel("y offset [px]")
            .addVerticalLine(0, "white")
            .addHorizontalLine(0, "white")
            .setFixedRatio();

    rs::HistConfig conf1d;
    conf1d.setMinMaxX(-2,2)
            .setMinMaxY(-2,2)
            .addVerticalLine(0, "black");

    rs::Stats2D<float> stat_2d_level[4];

#pragma omp parallel for
    for (size_t color_index = 0; color_index < 3; ++color_index) {
#pragma omp parallel for
        for (size_t level = 0; level < 4; ++level) {
            rs::Stats2D<float> & stats = local_stat_2d_color[color_index][level];
            stat_2d_level[level].push(stats);
            if (stats.empty()) {
                continue;
            }
            size_t const color = (level == 0 ? color_index + 2 : color_index);
            std::string suffix = "-offset2d-c" + std::to_string(color) + "-l" + std::to_string(level);
            stats.saveSummary(result_prefix + suffix + ".summary");
            suffixes.insert(suffix);
            stats.plotHist(result_prefix + suffix,
                           stats.FreedmanDiaconisBinSize(),
                           conf2d);
            suffix = "-offset2d-log-c" + std::to_string(color) + "-l" + std::to_string(level);
            suffixes.insert(suffix);
            stats.plotHist(result_prefix + suffix,
                           stats.FreedmanDiaconisBinSize(),
                           conf2d.clone()
                           .setLogCB());
            for (std::string const axis : {"x", "y"}) {
                rs::QuantileStats<float> axis_stats = stats.get(axis);
                suffix = "-offset-" + axis + "-c" + std::to_string(color) + "-l" + std::to_string(level);
                suffixes.insert(suffix);
                axis_stats.plotHistAndCDF(result_prefix + suffix,
                                          axis_stats.FreedmanDiaconisBinSize(),
                                          conf1d.clone()
                                          .setDataLabel(axis + " offset [px]"));
            }
        }
    }

#pragma omp parallel for
    for (size_t level = 0; level < 4; ++level) {
        rs::Stats2D<float> & stats = stat_2d_level[level];
        if (stats.empty()) {
            continue;
        }
        std::string suffix = "-offset2d-l" + std::to_string(level);
        stats.saveSummary(result_prefix + suffix + ".summary");
        suffixes.insert(suffix);
        stats.plotHist(result_prefix + suffix,
                       stats.FreedmanDiaconisBinSize(),
                       conf2d);
        suffix = "-offset2d-log-l" + std::to_string(level);
        suffixes.insert(suffix);
        stats.plotHist(result_prefix + suffix,
                       stats.FreedmanDiaconisBinSize(),
                       conf2d.clone()
                       .setLogCB());
        for (std::string const axis : {"x", "y"}) {
            rs::QuantileStats<float> axis_stats = stats.get(axis);
            suffix = "-offset-" + axis + "-l" + std::to_string(level);
            suffixes.insert(suffix);
            axis_stats.plotHistAndCDF(result_prefix + suffix,
                                      axis_stats.FreedmanDiaconisBinSize(),
                                      conf1d.clone()
                                      .setDataLabel(axis + " offset [px]"));
        }
    }


    std::string suffix = "-offset2d";
    suffixes.insert(suffix);
    suffix = "-offset2d-log";
    suffixes.insert(suffix);
    for (std::string const axis : {"x", "y"}) {
        suffix = "-offset-" + axis + "-all";
        suffixes.insert(suffix);
    }

#pragma omp parallel sections
    {
#pragma omp section
        local_stat_2d.plotHist(result_prefix + suffix,
                               local_stat_2d.FreedmanDiaconisBinSize(),
                               conf2d);
#pragma omp section
        local_stat_2d.plotHist(result_prefix + suffix,
                               local_stat_2d.FreedmanDiaconisBinSize(),
                               conf2d.clone()
                               .setLogCB());
#pragma omp section
        {
            for (std::string const axis : {"x", "y"}) {
                rs::QuantileStats<float> axis_stats = local_stat_2d.get(axis);
                suffix = "-offset-" + axis + "-all";
                suffixes.insert(suffix);
                axis_stats.plotHistAndCDF(result_prefix + suffix,
                                          axis_stats.FreedmanDiaconisBinSize(),
                                          conf1d.clone()
                                          .setDataLabel(axis + " offset [px]"));
            }
        }
    }
}

void analyzeOffsets(
        hdcalib::CornerStore const& a,
        hdcalib::CornerStore const& b,
        std::set<std::string> & suffixes,
        std::string const& result_prefix,
        int const recursion,
        rs::Stats2D<float> &local_stat_2d,
        rs::Stats2D<float> local_stat_2d_color[3][4],
std::vector<cv::Point2d> & src,
std::vector<cv::Point2d> & dst
) {
    for (size_t ii = 0; ii < a.size(); ++ii) {
        hdm::Corner const& _a = a.get(ii);
        hdm::Corner _b;
        if (b.hasIDLevel(_a, _b, _a.layer)) {
            size_t const color = CornerColor::getColor(_a, recursion);
            if (color > 3) {
                throw std::runtime_error("Color value unexpectedly >3, aborting.");
            }
            std::pair<double, double> const val(_b.p.x - _a.p.x, _b.p.y - _a.p.y);
            local_stat_2d.push_unsafe(val);
            if (_a.layer == 0) { // main marker
                if (color < 2) {
                    throw std::runtime_error("Error: Main marker with color < 2");
                }
                if (color > 4) {
                    throw std::runtime_error("Error: Main marker with color > 4");
                }
            }
            else {
                if (color > 1) {
                    throw std::runtime_error("Error: Submarker with color > 1");
                }
            }
            size_t const color_index = (_a.layer == 0 ? color-2 : color);
            local_stat_2d_color[2][_a.layer].push_unsafe(val);
            local_stat_2d_color[color_index][_a.layer].push_unsafe(val);
            src.push_back(_a.p);
            dst.push_back(_b.p);
        }
    }

    plotOffsets(
                suffixes,
                local_stat_2d,
                local_stat_2d_color,
                result_prefix
                );
}

void analyzeFileList(std::vector<std::string> const& files, std::string const& prefix, int const recursion) {
    if (files.size() < 2) {
        std::cout << "Number of files given is smaller than two." << std::endl;
        return;
    }
    hdcalib::Calib calib;
    calib.setRecursionDepth(recursion);
    CornerCache & cache = CornerCache::getInstance();
    hdcalib::CornerStore & first_corners = cache[files.front()];
    rs::Stats2D<float> stat_2d, stat_2d_color[3][4];
    std::set<std::string> suffixes;
    rs::Ellipses ellipses;

    std::vector<hdcalib::Similarity2D> fits(files.size());
#pragma omp parallel for
    for (size_t ii = 1; ii < files.size(); ++ii) {
        rs::Stats2D<float> local_stat_2d, local_stat_2d_color[3][4];
        hdcalib::CornerStore &  cmp_corners = cache[files[ii]];
        std::vector<cv::Point2d> src, dst;
        analyzeOffsets(first_corners, cmp_corners, suffixes, prefix + files[ii], recursion, local_stat_2d, local_stat_2d_color, src, dst);
        stat_2d.push(local_stat_2d);
        local_stat_2d.getQuantileEllipse(ellipses, .5);
        for (size_t ii = 0; ii < 3; ++ii) {
            for (size_t jj = 0; jj < 4; ++jj) {
                stat_2d_color[ii][jj].push(local_stat_2d_color[ii][jj]);
            }
        }
        hdcalib::Similarity2D & fit = fits[ii];
        fit.src = src;
        fit.dst = dst;
        fit.runFit();
    }

    plotOffsets(
                suffixes,
                stat_2d,
                stat_2d_color,
                prefix + "all"
                );

    ellipses.plot(prefix + "all-ellipses", rs::HistConfig()
                  .setTitle("50%-ellipses of the offsets of all tested images relative to the first.")
                  .setXLabel("x offset [px]")
                  .setYLabel("y offset [px]")
                  .setFixedRatio());

    for (std::string const& suffix : suffixes) {
        std::string const dirname = "suffix-" + suffix;
        std::cout << "mkdir \"" << dirname << "\"" << std::endl;
        std::cout << "for i in *" << suffix << "*; do ln -s $(pwd)/$i \"" << dirname << "\"/$i; done " << std::endl;
    }
}

using namespace hdcalib;

class RMS_Eval {
public:
    cv::Mat_<uint8_t> valid_mask;

    std::unordered_map<size_t, std::vector<hdm::Corner> > valid_data;
    std::unordered_map<size_t, std::vector<hdm::Corner> > invalid_data;

    cv::Point2f getMedian(std::vector<hdm::Corner> const& vec) {
        rs::QuantileStats<float> x, y;
        if (vec.empty()) {
            return {0,0};
        }
        hdm::Corner const& ref = vec.front();
        for (hdm::Corner const& c : vec) {
            x.push_unsafe(c.p.x);
            y.push_unsafe(c.p.y);
            if (ref.id != c.id || ref.page != c.page || ref.layer != c.layer) {
                std::stringstream msg;
                msg << "IDs don't match: " << ref.id << "/" << ref.page << ", l " << ref.layer
                    << " vs. " << c.id << "/" << c.page << ", l " << c.layer;
                throw std::runtime_error(msg.str());
            }
        }
        return {x.getMedian(), y.getMedian()};
    }

    void plot(std::string const& prefix) {
        if (invalid_data.size() > 0) {
            removeUncertain(invalid_data);
            plot(invalid_data, prefix + "invalid");
        }
        removeUncertain(valid_data);
        plot(valid_data, prefix + "valid");
    }

    void writePlot(std::string const& prefix,
                   rs::Stats2D<float> const& data,
                   rs::HistConfig const conf) {
        data.plotHist(prefix, data.FreedmanDiaconisBinSize(), conf);
    }

    bool validRMS(hdm::Corner const& c) {
        int32_t fails = hdm::Corner::unsetFail(c.fails, hdm::Corner::Fail::rms);
        return c.rms > 0 && c.snr > 0 && (fails == 0);
    }

    void removeUncertain(std::unordered_map<size_t, std::vector<hdm::Corner> > & data) {
        rs::QuantileStats<float> sizes;
        for (auto const & it : data) {
            sizes.push_unsafe(it.second.size());
        }
        for (auto & it : data) {
            if (it.second.size() < sizes.getMedian()/4) {
                it.second.clear();
            }
        }
    }

    void plot(std::unordered_map<size_t, std::vector<hdm::Corner> > const& data, std::string const& prefix) {
        std::unordered_map<std::string, std::unordered_map<int, rs::Stats2D<float> > >
                layer;
        std::unordered_map<std::string, std::unordered_map<int, std::unordered_map<int, rs::Stats2D<float> > > >
                layer_color;

        std::unordered_map<int, rs::QuantileStats<float> > errors_by_color;
        rs::QuantileStats<float> errors_total;

        rs::QuantileStats<float> id_counts;

        rs::ThresholdErrorMean<float> rms_vs_error, snr_sigma_vs_error;

        ParallelTime t;
        rs::BinaryStats ignored;
        for (auto const& it : data) {
            cv::Point2f median = getMedian(it.second);
            id_counts.push_unsafe(it.second.size());
            for (hdm::Corner const& c : it.second) {
                if (!validRMS(c)) {
                    ignored.pushTrue(1);
                    continue;
                }
                ignored.pushFalse(1);
                cv::Point2f diff = median - c.p;
                double length = log10(0.001 + std::min<double>(10.0, std::sqrt(diff.dot(diff))));
                double const rms = log10(0.001 + std::min<double>(c.rms, 2.0));
                double const snr = std::min<double>(c.rms, 50.0);
                double const snr_sigma = std::min<double>(std::abs(c.snr * c.getSigma()), 50.0);
                layer["RMS"][c.layer].push_unsafe(rms, length);
                layer["SNR"][c.layer].push_unsafe(snr, length);
                layer["SNR x Sigma"][c.layer].push_unsafe(snr_sigma, length);

                layer_color["RMS"][c.layer][c.color].push_unsafe(rms, length);
                layer_color["SNR"][c.layer][c.color].push_unsafe(snr, length);
                layer_color["SNR x Sigma"][c.layer][c.color].push_unsafe(snr_sigma, length);

                rms_vs_error.push_unsafe(rms, length);
                snr_sigma_vs_error.push_unsafe(1.0/(1+snr_sigma), length);

                errors_by_color[c.color].push_unsafe(length);
                errors_total.push_unsafe(length);
            }
        }
        std::cout << "Filling Stats2D for " << prefix << ": " << t.print() << std::endl;
        std::cout << "Ignored: " << ignored.print() << std::endl;
        t.start();

#pragma omp parallel sections
        {
#pragma omp section
            rms_vs_error.save(prefix + "-rms-vs-error.bin");
#pragma omp section
            snr_sigma_vs_error.save(prefix + "-snr_sigma-vs-error.bin");
#pragma omp section
            rms_vs_error.plot(prefix + "-rms-vs-error", rs::HistConfig().setMaxPlotPts(10000));
#pragma omp section
            snr_sigma_vs_error.plot(prefix + "-snr_sigma-vs-error", rs::HistConfig().setMaxPlotPts(10000));
#pragma omp section
            id_counts.plotHistAndCDF(prefix + "-id-counts", 1,
                                     rs::HistConfig().setDataLabel("# samples"));
#pragma omp section
            errors_by_color[0].plotHistAndCDF(prefix + "-errors-c0", -1, rs::HistConfig()
                                              .setMaxBins(1000, 1000).setMaxPlotPts(1000).setDataLabel("Error"));
#pragma omp section
            errors_by_color[1].plotHistAndCDF(prefix + "-errors-c1", -1, rs::HistConfig()
                                              .setMaxBins(1000, 1000).setMaxPlotPts(1000).setDataLabel("Error"));
#pragma omp section
            errors_total.plotHistAndCDF(prefix + "-errors-total", -1, rs::HistConfig()
                                        .setMaxBins(1000, 1000).setMaxPlotPts(1000).setDataLabel("Error"));
        }

        rs::HistConfig conf;
        conf
                .setMaxBins(50,50)
                .setIgnoreAmount(1e-4);

        for (auto const& it : layer) {
            std::string const& type = it.first;
            for (auto const& it2 : it.second) {
                std::string const layer = "-l" + std::to_string(it2.first);
                conf.setXLabel(type).setYLabel("Error");
#pragma omp parallel sections
                {
#pragma omp section
                    it2.second.plotHist(prefix + layer + "-" + type + "-linear",
                                        it2.second.FreedmanDiaconisBinSize(),
                                        conf);
#pragma omp section
                    it2.second.get1().plotHist(prefix + layer + "-" + type + "-hist",
                                               -1,
                                               conf.clone().setDataLabel(type));
#pragma omp section
                    it2.second.plotHist(prefix + layer + "-" + type + "-linear-norm",
                                        it2.second.FreedmanDiaconisBinSize(),
                                        conf.clone().setNormalizeX());
#pragma omp section
                    it2.second.plotHist(prefix + layer + "-" + type + "-log",
                                        it2.second.FreedmanDiaconisBinSize(),
                                        conf.clone().setLogCB());
#pragma omp section
                    it2.second.plotHist(prefix + layer + "-" + type + "-log-norm",
                                        it2.second.FreedmanDiaconisBinSize(),
                                        conf.clone().setLogCB().setNormalizeX());
                }
            }
        }

        for (auto const& it : layer_color) {
            std::string const& type = it.first;
            for (auto const& it2 : it.second) {
                std::string const layer = "-l" + std::to_string(it2.first);
                for (auto const& it3 : it2.second) {
                    std::string const color = "-c" + std::to_string(it3.first);
                    conf.setXLabel(type).setYLabel("Error");
#pragma omp parallel sections
                    {
#pragma omp section
                        it3.second.plotHist(prefix + layer + color + "-" + type + "-linear",
                                            it3.second.FreedmanDiaconisBinSize(),
                                            conf);
#pragma omp section
                        it3.second.get1().plotHist(prefix + layer + color + "-" + type + "-hist",
                                                   -1,
                                                   conf.clone().setDataLabel(type));
#pragma omp section
                        it3.second.plotHist(prefix + layer + color + "-" + type + "-linear-norm",
                                            it3.second.FreedmanDiaconisBinSize(),
                                            conf.clone().setNormalizeX());
#pragma omp section
                        it3.second.plotHist(prefix + layer + color + "-" + type + "-log",
                                            it3.second.FreedmanDiaconisBinSize(),
                                            conf.clone().setLogCB());
#pragma omp section
                        it3.second.plotHist(prefix + layer + color + "-" + type + "-log-norm",
                                            it3.second.FreedmanDiaconisBinSize(),
                                            conf.clone().setLogCB().setNormalizeX());
                    }
                }
            }
        }
        std::cout << "Plotting Stats2D for " << prefix << ": " << t.print() << std::endl;

    }

    static double getFactor(std::string const& name) {
        std::regex expression("exposure-([0-9]*)");
        std::smatch match;
        std::regex_search (name,match,expression);
        if (match.size() == 2) {
            return std::stod(match[1]) / 2048.0;
        }

        return 1;
    }

    void addList(std::vector<std::string> const& files) {
        ParallelTime t;
#pragma omp parallel for
        for (size_t ii = 0; ii < files.size(); ++ii) {
            CornerCache::getInstance()[files[ii]];
        }
        std::cout << "Reading " << files.size() << " files: " << t.print() << std::endl;
        t.start();
        std::vector<size_t> file_counts(files.size());
        for (size_t ii = 0; ii < files.size(); ++ii) {
            double factor = getFactor(files[ii]);
            for (hdm::Corner c : CornerCache::getInstance()[files[ii]].getCorners()) {
                //c.rms *= factor;
                if (valid_mask.empty() || valid_mask(c.p) > 127) {
                    valid_data[Calib::getIdHash(c)].push_back(c);
                }
                else {
                    invalid_data[Calib::getIdHash(c)].push_back(c);
                }
                if (RMS_Eval::validRMS(c)) {
                    file_counts[ii]++;
                }
            }
            std::cout << "File count for " << files[ii] << ": " << file_counts[ii] << std::endl;
        }
        std::cout << "Putting into multimap: " << t.print() << std::endl;
        t.start();
    }
};

int main(int argc, char ** argv) {

    assert(1 == RMS_Eval::getFactor("exposure-2048/hzgvbhjii87z"));
    assert(.5 == RMS_Eval::getFactor("exposure-1024/hzgvbhjii87z"));
    assert(.25 == RMS_Eval::getFactor("exposure-512/hzgvbhjii87z"));
    assert(.125 == RMS_Eval::getFactor("exposure-256/hzgvbhjii87z"));
    assert(.125 == RMS_Eval::getFactor("target2-warmup/exposure-256/2c_dd_a3_31_87_d7-31417-7459610237144-1607811636128.buf-0.tif"));
    assert(.0625 == RMS_Eval::getFactor("exposure-128/hzgvbhjii87z"));
    assert(.0625/2 == RMS_Eval::getFactor("exposure-64/hzgvbhjii87z"));
    assert(.0625/4 == RMS_Eval::getFactor("exposure-32/hzgvbhjii87z"));

    clog::Logger::getInstance().addListener(std::cout);

    ParallelTime t, total_time;
    std::stringstream time_log;

    std::map<std::string, std::vector<std::string> > files;

    int recursion = 0;

    bool rms_eval = false;

    cv::Mat_<uint8_t> mask;

    try {
        TCLAP::CmdLine cmd("hdcalib tool for analyzing offsets of detected markers between images", ' ', "0.1");

        TCLAP::MultiArg<std::string> files_arg("f",
                                               "file",
                                               "Text file containing image filenames. The first image is the reference and is compared to all other images.",
                                               false,
                                               "text file containing image filenames");
        cmd.add(files_arg);

        TCLAP::ValueArg<int> recursive_depth_arg("r", "recursion",
                                                 "Recursion depth of the sub-marker detection. Set this to the actual recursion depth of the used target.",
                                                 true, -1, "int");
        cmd.add(recursive_depth_arg);

        TCLAP::ValueArg<std::string> mask_arg("", "mask",
                                              "mask image marking the valid parts (containig target) white",
                                              false, "", "int");
        cmd.add(mask_arg);

        TCLAP::SwitchArg rms_eval_arg("", "rms", "Only plot RMS / SNR evaluation");
        cmd.add(rms_eval_arg);


        cmd.parse(argc, argv);

        recursion = recursive_depth_arg.getValue();
        rms_eval = rms_eval_arg.getValue();

        if (mask_arg.isSet()) {
            mask = cv::imread(mask_arg.getValue(), cv::IMREAD_GRAYSCALE);
        }

        for (std::string const& src : files_arg.getValue()) {
            std::ifstream in(src);
            std::vector<std::string> local_files;
            std::string name;
            while (in >> name) {
                trim(name);
                if (!name.empty() && fs::is_regular_file(name)) {
                    local_files.push_back(name);
                }
            }
            if (local_files.size() > 1) {
                files[src] = local_files;
            }
        }
    }
    catch (TCLAP::ArgException const & e) {
        std::cerr << e.what() << std::endl;
        return 0;
    }
    catch (...) {
        std::cerr << "Unknown exception" << std::endl;
        return 0;
    }

    //fs::current_path("./plots", ignore_error_code);
    std::cout << fs::current_path() << std::endl;

    if (rms_eval) {
        for (auto const& it : files) {
            RMS_Eval eval;
            eval.valid_mask = mask;
            std::string plots_name = "plots-" + it.first;
            fs::create_directories(plots_name, ignore_error_code);
            eval.addList(it.second);
            eval.plot(plots_name + "/");
        }
        return EXIT_SUCCESS;
    }

    for (auto const& it : files) {
        std::string plots_name = "plots-" + it.first;
        fs::create_directories(plots_name, ignore_error_code);
        analyzeFileList(it.second, plots_name, recursion);
    }


    std::cout << "Total time: " << total_time.print() << std::endl;

    return EXIT_SUCCESS;
}
